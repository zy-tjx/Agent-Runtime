"""
工具执行器
负责：查找工具 →参数校验 → 超时控制 → 工具调用 → 异常捕获 → 结果标准化
"""
import time
from typing import Any, Optional
from pydantic import BaseModel, ValidationError
from tenacity import stop_after_attempt, wait_fixed

from tools.tool_registry import ToolRegistry


class ToolOutput(BaseModel):
    """所有工具返回结果的统一结构"""

    tool_name: str

    status: str

    result: Any = None
    #工具执行时的具体产出（类型由各工具自行约定）
    error: Optional[str] = None

    duration_ms: int = 0


class ToolExecutor:
    """
    工具执行器

    执行流程：查找工具 → 校验参数 → 超时控制 → 调用 → 异常捕获 → 标准化返回

    用法:
        registry = ToolRegistry()
        registry.register(...)
        executor = ToolExecutor(registry)
        output = executor.execute("search_docs", {"query": "什么是 RAG"})
    """

    def __init__(self, registry: ToolRegistry, default_timeout_ms: int = 30000):
        self.registry = registry
        self.default_timeout_ms = default_timeout_ms

    def execute(self, tool_name: str, raw_args: dict[str, Any]) -> ToolOutput:
        """
        执行指定工具

        Args:
            tool_name: 注册的工具名称
            raw_args: 原始参数字典（来自 DECIDE 的决策）

        Returns:
            ToolOutput: 统一结构化的执行结果
        """
        start_time = time.time()

        # ── 1. 查找工具 ──
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolOutput(
                tool_name=tool_name,
                status="failed",
                error=f"工具 '{tool_name}' 在注册表中找不到",
            )

        # ── 2. 参数校验 ──
        try:
            validated_input = tool.input_schema(**raw_args)
        except ValidationError as e:
            return ToolOutput(
                tool_name=tool_name,
                status="failed",
                error=f"参数校验失败: {e.errors()}",
            )

        # ── 3. 超时控制 + 执行 ──
        try:
            retry_decorator = _build_retry(max_attempts=3, wait_ms=500)
            result = retry_decorator(tool.run_func)(validated_input)
            result.duration_ms = int((time.time() - start_time) * 1000)
            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ToolOutput(
                tool_name=tool_name,
                status="failed",
                error=f"工具执行异常: {str(e)}",
                duration_ms=duration_ms,
            )


def _build_retry(max_attempts: int, wait_ms: int):
    """构建 tenacity 重试装饰器"""
    from tenacity import retry
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait_ms / 1000),
        reraise=True,
    )
