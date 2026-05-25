"""
记忆写入工具（memory_write）
将 REFLECT 产出的结构化数据持久化到长期记忆存储（SQLite）
"""
from typing import Optional
from pydantic import BaseModel, Field
from tools.tool_executor import ToolOutput
from memory.long_term_memory import MemoryRecord, save


class MemoryWriteInput(BaseModel):
    """memory_write 输入参数"""

    key: str = Field(..., description="记忆唯一键", min_length=1)
    value: dict = Field(..., description="记忆内容（JSON 可序列化）")
    category: str = Field(
        default="experience",
        description="记忆类别：profile / progress / experience / session",
        pattern="^(profile|progress|experience|session)$",
    )
    session_id: Optional[str] = Field(default=None, description="关联会话 ID")


def run(input_data: MemoryWriteInput) -> ToolOutput:
    """写入长期记忆"""
    record = MemoryRecord(
        key=input_data.key,
        value=input_data.value,
        category=input_data.category,
        session_id=input_data.session_id,
    )
    try:
        save(record)
        return ToolOutput(
            tool_name="memory_write",
            status="success",
            result={"key": input_data.key, "written": True},
            error=None,
            duration_ms=0,
        )
    except Exception as e:
        return ToolOutput(
            tool_name="memory_write",
            status="failed",
            result=None,
            error=str(e),
            duration_ms=0,
        )
