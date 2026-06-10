from fastapi import FastAPI
from pydantic import BaseModel
from main import graph
import uvicorn

app = FastAPI(title="CodePilot")

# 存储每个会话的对话历史
sessions = {}

SYSTEM_PROMPT = """你是一个Python编程助手。
当用户提出编程需求时，你需要：
1. 编写Python代码
2. 调用execute_code工具执行代码
3. 如果报错，分析错误并修复，重新执行
4. 直到代码成功运行为止
每次只调用一次工具。"""


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id

    # 初始化新会话
    if session_id not in sessions:
        sessions[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # 追加用户消息
    sessions[session_id].append({
        "role": "user",
        "content": request.message
    })

    # 调用 graph
    state = graph.invoke({
        "messages": sessions[session_id],
        "finished": False
    })

    # 更新会话历史
    sessions[session_id] = state["messages"]

    # 取最后一条 assistant 消息作为回复
    reply = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "assistant" and msg.get("content"):
            reply = msg["content"]
            break

    return ChatResponse(reply=reply, session_id=session_id)


@app.get("/sessions/{session_id}/files")
def list_session_files(session_id: str):
    """列出 workspace 下的文件"""
    import os
    workspace = "workspace"
    if not os.path.exists(workspace):
        return {"files": []}
    return {"files": os.listdir(workspace)}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """清除会话历史"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": f"会话 {session_id} 已清除"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)