"""
FastAPI 应用入口
单端点 /chat，零业务逻辑，纯透传
启动: .venv/Scripts/uvicorn interface.fastapi_app:app --reload
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from interface.api_routes import ChatRequest, handle_chat

app = FastAPI(title="Agent Runtime API", version="1.0")

@app.post("/chat")
def chat(req: ChatRequest):
    """Agent 对话接口：接收问题 → 运行完整状态机 → 返回治理结果"""
    return handle_chat(req)
