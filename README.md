# CodePilot 🤖

基于 LangGraph 构建的多 Agent 自动化编程系统。用自然语言描述需求，系统自动规划、编写、执行、修复代码，直到任务完成。

## 系统架构
用户需求
↓
PlannerAgent（DAG任务规划）
↓
CodeAgent（编写+执行+自我修复）←→ ToolNode（代码执行/文件读写）
↓
ReviewAgent（代码质量审查）
↓
输出结果
## 核心特性

- **Multi-Agent 协作**：PlannerAgent、CodeAgent、ReviewAgent 分工协作，职责清晰
- **DAG 任务调度**：PlannerAgent 将需求拆解为有依赖关系的任务图，按拓扑顺序执行
- **自我修复**：CodeAgent 执行报错后自动分析原因，重写代码直至成功
- **SubAgent 上下文隔离**：每个 Agent 只接收必要信息，避免上下文污染
- **上下文压缩**：对话超过 10 轮自动摘要压缩，防止 Token 溢出
- **工具调用**：支持代码执行、文件读写、目录列举四种工具

## 技术栈

- **框架**：LangGraph、LangChain
- **LLM**：DeepSeek-V3
- **后端**：FastAPI、Uvicorn
- **工具**：Python subprocess、文件系统

## 评估结果

在 10 道标准编程题测试中：

| 指标 | 结果 |
|------|------|
| 任务成功率 | 100% |
| 一次成功率 | 100% |
| 平均修复次数 | 0.0 次 |
| 平均响应时间 | 47.69 秒 |

## 快速开始

**1. 安装依赖**
```bash
pip install langchain langgraph langchain-openai fastapi uvicorn python-dotenv
```

**2. 配置环境变量**
```bash
# 新建 .env 文件
SILICONFLOW_API_KEY=你的key
```

**3. 启动 API 服务**
```bash
python api.py
```

**4. 访问接口文档**
```bash
http://localhost:8000/docs
```
## API 接口

### POST /chat
发送编程需求，返回执行结果。

```json
{
  "session_id": "test001",
  "message": "写一个快速排序函数并保存到本地"
}
```

### GET /sessions/{session_id}/files
列出当前会话生成的文件。

### DELETE /sessions/{session_id}
清除会话历史。

## 项目结构
codepilot/
├── main.py        # Agent 核心逻辑（LangGraph 状态机）
├── api.py         # FastAPI 接口
├── eval.py        # 评估脚本
├── .env           # API Key 配置
└── workspace/     # Agent 生成的代码文件