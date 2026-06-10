from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
import subprocess
import tempfile
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# ============ 状态定义 ============

class State(TypedDict):
    user_request: str       # 用户原始需求
    plan: str               # PlannerAgent 输出的任务计划
    code_result: str        # CodeAgent 输出的代码和执行结果
    review: str             # ReviewAgent 输出的评审意见
    messages: list          # CodeAgent 的对话历史（上下文隔离）
    finished: bool

# ============ 工具定义 ============

def execute_code(code: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ['python', tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
            encoding='utf-8'
        )
        if result.returncode == 0:
            return f"执行成功:\n{result.stdout}"
        else:
            return f"执行报错:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "执行超时（超过10秒）"
    finally:
        os.unlink(tmp_path)


WORKSPACE = "workspace"

def read_file(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件内容:\n{content}"
    except FileNotFoundError:
        return f"错误: 文件 {filepath} 不存在"
    except Exception as e:
        return f"读取失败: {str(e)}"


def write_file(filepath: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文件已保存: {filepath}"
    except Exception as e:
        return f"写入失败: {str(e)}"


def list_files() -> str:
    try:
        if not os.path.exists(WORKSPACE):
            return "workspace 目录为空"
        files = os.listdir(WORKSPACE)
        if not files:
            return "workspace 目录为空"
        return "workspace 目录下的文件:\n" + "\n".join(files)
    except Exception as e:
        return f"列举失败: {str(e)}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "执行Python代码，返回执行结果或错误信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的Python代码"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "文件路径"}
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入本地文件，文件不存在则创建",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"}
                },
                "required": ["filepath", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出workspace目录下的所有文件",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# ============ PlannerAgent ============

def planner_node(state: State):
    print("\n[PlannerAgent] 规划任务中...")

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {
                "role": "system",
                "content": """你是一个任务规划专家。
将用户的编程需求拆解为具体步骤，并以JSON格式输出任务DAG。

输出格式（只输出JSON，不要其他内容）：
{
  "goal": "任务总目标",
  "tasks": [
    {
      "id": "task_1",
      "name": "步骤名称",
      "description": "具体描述",
      "depends_on": []
    },
    {
      "id": "task_2", 
      "name": "步骤名称",
      "description": "具体描述",
      "depends_on": ["task_1"]
    },
    {
      "id": "task_3",
      "name": "步骤名称", 
      "description": "具体描述",
      "depends_on": ["task_1"]
    },
    {
      "id": "task_4",
      "name": "步骤名称",
      "description": "具体描述",
      "depends_on": ["task_2", "task_3"]
    }
  ]
}

规则：
- depends_on 为空表示该步骤无前置依赖，可以最先执行
- depends_on 列出必须先完成的步骤id
- 最多5个步骤
- 只输出JSON，不要任何解释"""
            },
            {
                "role": "user",
                "content": state["user_request"]
            }
        ]
    )

    raw = response.choices[0].message.content.strip()
    # 去掉可能的markdown代码块
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    dag = json.loads(raw)
    print(f"任务DAG: 共{len(dag['tasks'])}个步骤")
    for task in dag['tasks']:
        deps = dag['tasks'] and task['depends_on']
        print(f"  {task['id']}: {task['name']} 依赖:{task['depends_on']}")

    return {"plan": json.dumps(dag, ensure_ascii=False)}


# ============ CodeAgent ============

def code_node(state: State):
    print("\n[CodeAgent] 按DAG顺序执行任务...")

    dag = json.loads(state["plan"])
    tasks = dag["tasks"]

    def compress_messages(messages: list) -> list:
        """超过10条消息时，压缩中间历史为摘要"""
        if len(messages) <= 10:
            return messages

        print("\n[上下文压缩] 消息过多，压缩历史...")

        # 保留system消息和最近2条
        system_msg = messages[0]
        recent_msgs = messages[-2:]
        middle_msgs = messages[1:-2]

        # 把中间消息拼成文本摘要
        history_text = ""
        for msg in middle_msgs:
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if role == "tool":
                history_text += f"[工具结果] {content[:200]}\n"
            elif role == "assistant":
                history_text += f"[Assistant] {content[:200]}\n"

        summary_response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {
                    "role": "user",
                    "content": f"请用3句话总结以下对话历史：\n{history_text}"
                }
            ]
        )
        summary = summary_response.choices[0].message.content
        print(f"压缩摘要: {summary[:100]}...")

        # 重新组合：system + 摘要 + 最近2条
        compressed = [
            system_msg,
            {"role": "user", "content": f"[历史摘要] {summary}"},
            *recent_msgs
        ]
        print(f"消息数: {len(messages)} → {len(compressed)}")
        return compressed

    def code_node(state: State):
        print("\n[CodeAgent] 按DAG顺序执行任务...")

        dag = json.loads(state["plan"])
        tasks = dag["tasks"]

        # 如果是工具调用返回，继续当前messages
        if state["messages"]:
            # 上下文压缩
            state["messages"] = compress_messages(state["messages"])

    # 如果是工具调用返回，继续当前messages
    if state["messages"]:
        last_msg = state["messages"][-1]
        if last_msg.get("role") == "tool":
            # 工具执行完，继续让LLM处理
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=state["messages"],
                tools=tools,
                tool_choice="auto"
            )
            msg = response.choices[0].message
            new_messages = list(state["messages"])

            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                new_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                return {"messages": new_messages, "finished": False}
            else:
                new_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
                return {
                    "messages": new_messages,
                    "code_result": msg.content,
                    "finished": False
                }

    # 第一次进入：按拓扑顺序构建执行指令
    completed = []
    execution_order = []

    # 拓扑排序
    remaining = list(tasks)
    while remaining:
        for task in remaining:
            if all(dep in completed for dep in task["depends_on"]):
                execution_order.append(task)
                completed.append(task["id"])
                remaining.remove(task)
                break

    # 构建任务描述
    task_desc = f"目标：{dag['goal']}\n\n按以下顺序执行任务：\n"
    for i, task in enumerate(execution_order):
        deps_str = f"（依赖：{', '.join(task['depends_on'])}）" if task["depends_on"] else "（无前置依赖）"
        task_desc += f"\n{i+1}. [{task['id']}] {task['name']} {deps_str}\n   {task['description']}\n"

    print(f"执行顺序: {[t['id'] for t in execution_order]}")

    messages = [
        {
            "role": "system",
            "content": """你是一个Python编程专家。
按照任务顺序逐步编写并执行代码。
要求：
- 每次只调用一次工具
- 执行报错则修复后重试
- 最终代码保存到workspace目录"""
        },
        {
            "role": "user",
            "content": task_desc
        }
    ]

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message
    new_messages = list(messages)

    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        new_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }]
        })
        return {"messages": new_messages, "finished": False}
    else:
        new_messages.append({
            "role": "assistant",
            "content": msg.content
        })
        return {
            "messages": new_messages,
            "code_result": msg.content,
            "finished": False
        }


# ============ ToolNode ============

def tool_node(state: State):
    print("\n[工具节点] 执行工具...")

    last_msg = state["messages"][-1]
    tool_call = last_msg["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])

    print(f"调用工具: {tool_name}")

    if tool_name == "execute_code":
        result = execute_code(args["code"])
    elif tool_name == "read_file":
        result = read_file(args["filepath"])
    elif tool_name == "write_file":
        result = write_file(args["filepath"], args["content"])
    elif tool_name == "list_files":
        result = list_files()
    else:
        result = f"未知工具: {tool_name}"

    print(f"结果: {result}")

    new_messages = list(state["messages"])
    new_messages.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": result
    })

    return {"messages": new_messages, "finished": False}


# ============ ReviewAgent ============

def review_node(state: State):
    print("\n[ReviewAgent] 审查代码质量...")

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {
                "role": "system",
                "content": """你是一个代码审查专家。
根据原始需求和执行结果，评审代码质量。
输出格式：
✅ 完成情况：xxx
📊 代码质量：xxx
⚠️ 潜在问题：xxx（如果没有写"无"）
💡 优化建议：xxx（如果没有写"无"）"""
            },
            {
                "role": "user",
                "content": f"原始需求：{state['user_request']}\n\n任务计划：{state['plan']}\n\n执行结果：{state['code_result']}"
            }
        ]
    )

    review = response.choices[0].message.content
    print(f"\n审查结果:\n{review}")

    return {"review": review, "finished": True}


# ============ 路由函数 ============

def after_code(state: State):
    """CodeAgent之后的路由"""
    last_msg = state["messages"][-1]
    if last_msg.get("tool_calls"):
        return "tool"
    # 没有工具调用说明代码写完了，进入ReviewAgent
    return "review"


# ============ 构建图 ============

graph_builder = StateGraph(State)

graph_builder.add_node("planner", planner_node)
graph_builder.add_node("code", code_node)
graph_builder.add_node("tool", tool_node)
graph_builder.add_node("review", review_node)

graph_builder.set_entry_point("planner")
graph_builder.add_edge("planner", "code")

graph_builder.add_conditional_edges(
    "code",
    after_code,
    {"tool": "tool", "review": "review"}
)

graph_builder.add_edge("tool", "code")
graph_builder.add_edge("review", END)

graph = graph_builder.compile()


# ============ 运行 ============

def run(user_request: str):
    print(f"\n用户需求: {user_request}")
    print("=" * 50)

    final_state = graph.invoke({
        "user_request": user_request,
        "plan": "",
        "code_result": "",
        "review": "",
        "messages": [],
        "finished": False
    })

    print("\n" + "=" * 50)
    print("任务完成！")
    return final_state


if __name__ == "__main__":
    run("写一个学生成绩管理系统，支持添加学生、录入成绩、计算平均分，保存到workspace/students.py")