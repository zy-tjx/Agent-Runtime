"""
统一兜底工具（fallback）
所有降级路径的最终出口，返回预设的安全回答
"""
from typing import Optional
from pydantic import BaseModel, Field
from tools.tool_executor import ToolOutput


class FallbackInput(BaseModel):
    """fallback 输入参数"""
    reason: str = Field(..., description="触发降级的原因")
    context: Optional[dict] = Field(default=None, description="当前错误上下文")


def run(input_data: FallbackInput) -> ToolOutput:
    """返回兜底回答（待 Recovery Phase 实现）"""
    return ToolOutput(
        tool_name="fallback",
        status="success",
        result={"message": "抱歉，当前无法处理您的请求。请稍后重试。"},
        error=None,
        duration_ms=0,
    )
