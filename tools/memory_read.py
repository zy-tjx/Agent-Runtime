"""
记忆读取工具（memory_read）
从长期记忆检索用户画像、学习进度、历史经验等数据
"""
from typing import Optional
from pydantic import BaseModel, Field
from tools.tool_executor import ToolOutput
from memory.long_term_memory import load


class MemoryReadInput(BaseModel):
    """memory_read 输入参数"""

    key: Optional[str] = Field(default=None, description="记忆键名，为空则按 category 筛选或返回全部")
    category: Optional[str] = Field(
        default=None,
        description="记忆类别：profile / progress / experience / session",
    )


def run(input_data: MemoryReadInput) -> ToolOutput:
    """读取长期记忆"""
    try:
        rows = load(key=input_data.key, category=input_data.category)

        if input_data.key and rows:
            data = rows[0]  # 按 key 查，返回单条
        else:
            data = rows  # 按 category 或全部，返回列表

        return ToolOutput(
            tool_name="memory_read",
            status="success",
            result={"data": data, "count": len(rows) if isinstance(data, list) else 1, "found": len(rows) > 0},
            error=None,
            duration_ms=0,
        )
    except Exception as e:
        return ToolOutput(
            tool_name="memory_read",
            status="failed",
            result=None,
            error=str(e),
            duration_ms=0,
        )
