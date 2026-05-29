"""
错误分类器
纯函数：输入 state 片段，输出结构化错误诊断，不依赖 LLM
"""
from typing import Any


def analyze(
    error: str | None = None,
    tool_results: list[dict] | None = None,
    retrieved_context: list[dict] | None = None,
    retrieval_attempted: bool = False,
    retry_count: int = 0,
    max_retries: int = 3,
    fallback_triggered: bool = False,   # 是否触发LLM降级兜底
    fallback_reason: str | None = None, # 引发LLM降级兜底的原因
) -> dict[str, Any]:
    """
    对错误信号做结构化分类

    Returns:
        {
            "error_type": str,     # none / llm_error / tool_failure / retrieval_empty
                                   # / max_retries_exhausted / unknown
            "recoverable": bool,   # 是否建议重试
            "severity": str,       # none / low / medium / high
            "summary": str,        # 中文摘要
        }
    """
    # ── 优先级 1：已达最大重试次数 ──
    if retry_count >= max_retries:
        return {
            "error_type": "max_retries_exhausted",
            "recoverable": False,
            "severity": "high",
            "summary": f"已达最大重试次数（{retry_count}/{max_retries}），强制终止",
        }

    # ── 优先级 2：LLM 降级 ──
    if fallback_triggered:
        return {
            "error_type": "llm_error",
            "recoverable": True,
            "severity": "medium",
            "summary": f"LLM 调用失败触发降级: {fallback_reason or '未知原因'}",
        }

    # ── 优先级 3：显式 error 字段 ──
    if error:
        return {
            "error_type": _classify_error_string(error),
            "recoverable": True,
            "severity": "medium",
            "summary": error,
        }

    # ── 优先级 4：工具执行失败 ──
    tool_results = tool_results or []
    failures = [r for r in tool_results if r.get("status") != "success"]
    if failures:
        failure_names = [r.get("tool_name", "unknown") for r in failures]
        return {
            "error_type": "tool_failure",
            "recoverable": True,
            "severity": "medium",
            "summary": f"工具执行失败: {', '.join(failure_names)}",
        }

    # ── 优先级 5：检索为空（仅在确实尝试过检索时判断） ──
    retrieved_context = retrieved_context or []
    if retrieval_attempted and len(retrieved_context) == 0:
        return {
            "error_type": "retrieval_empty",
            "recoverable": True,
            "severity": "low",
            "summary": "检索未返回任何文档",
        }

    # ── 无错误 ──
    return {
        "error_type": "none",
        "recoverable": False,
        "severity": "none",
        "summary": "未检测到错误",
    }


def _classify_error_string(error: str) -> str:
    """从错误字符串推断错误类型"""
    lower = error.lower()
    if any(kw in lower for kw in ["timeout", "timed out", "connection", "connect"]):
        return "llm_error"
    if any(kw in lower for kw in ["tool", "execute", "executor"]):
        return "tool_failure"
    if any(kw in lower for kw in ["retriev", "search", "vector"]):
        return "retrieval_empty"
    if any(kw in lower for kw in ["parse", "json", "schema", "validation", "structured"]):
        return "llm_error"
    return "unknown"
