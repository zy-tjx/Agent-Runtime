"""
会话状态持久化工具（context_store）
保存/恢复会话中间状态，供 checkpointer 使用
"""
from typing import Optional
from pydantic import BaseModel, Field
from tools.tool_executor import ToolOutput


class ContextStoreInput(BaseModel):
    """context_store 输入参数"""
    action: str = Field(..., description="操作类型：save / load / list")
    session_id: str = Field(..., description="会话 ID")
    data: Optional[dict] = Field(default=None, description="待保存的状态数据（save 时必填）")


def run(input_data: ContextStoreInput) -> ToolOutput:
    """会话状态持久化（待 Memory Phase 实现）"""
    return ToolOutput(
        tool_name="context_store",
        status="success",
        result={"action": input_data.action, "session_id": input_data.session_id},
        error=None,
        duration_ms=0,
    )
