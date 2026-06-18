from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import graph
import uvicorn
import sqlite3
import json
import os

app = FastAPI(title="CodePilot")

DB_PATH = "sessions.db"

# ============ SQLite 持久化 ============

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            messages TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def load_session(session_id: str) -> list:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT messages FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return []

def save_session(session_id: str, messages: list):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO sessions (session_id, messages) VALUES (?, ?)",
        (session_id, json.dumps(messages, ensure_ascii=False))
    )
    conn.commit()
    conn.close()

init_db()

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
    messages = load_session(session_id)

    if not messages:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages.append({"role": "user", "content": request.message})

    try:
        # 补全 State 所有必要字段
        state = graph.invoke({
            "user_request": request.message,
            "plan": "",
            "code_result": "",
            "review": "",
            "messages": messages,
            "finished": False
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {str(e)}")

    updated_messages = state.get("messages", messages)
    save_session(session_id, updated_messages)

    reply = ""
    for msg in reversed(updated_messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            reply = msg["content"]
            break

    if not reply:
        raise HTTPException(status_code=500, detail="Agent 未返回有效回复")

    return ChatResponse(reply=reply, session_id=session_id)


@app.get("/sessions/{session_id}/files")
def list_session_files(session_id: str):
    workspace = "workspace"
    if not os.path.exists(workspace):
        return {"files": []}
    return {"files": os.listdir(workspace)}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"message": f"会话 {session_id} 已清除"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)