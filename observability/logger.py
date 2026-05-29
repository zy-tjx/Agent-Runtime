"""
结构化日志系统
统一 JSON 输出，提供 node_start / node_end / node_error 契约
"""
import json
import time
from typing import Any, Optional

# ── 治理字段列表（从 AgentState 提取打日志） ──
_GOVERNANCE_FIELDS = [
    "mode", "retrieval_score", "groundedness_score",
    "completeness_score", "answer_source", "retry_reason",
    "retry_count", "error",
    "fallback_triggered", "fallback_reason",
]

def _summarize(state: dict) -> dict[str, Any]:
    """从 AgentState 中提取关键字段作为日志摘要"""
    summary = {}
    for key in _GOVERNANCE_FIELDS:
        if key in state:
            summary[key] = state[key]
    # 提取state中需要的字段的值
    # 追加非治理但有助于 trace 的字段
    if "current_step" in state:
        summary["current_step"] = state["current_step"]
    if "user_input" in state:
        val = state["user_input"]
        summary["user_input"] = val[:80] if len(val) > 80 else val
    return summary

class NodeLogger:
    """节点级结构化日志"""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._start_times: dict[str, float] = {}

    def info(self, message: str, **kwargs) -> None:
        """通用信息日志"""
        self._emit(event="info", message=message, **kwargs)

    # ── 内部 ──

    def _emit(self, event: str, **fields) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(time.time()*1000)%1000:03d}",
            "session_id": self.session_id,
            "event": event,
            **fields,
        }
        print(json.dumps(record, ensure_ascii=False))

    # ── 契约方法 ──

    def node_start(self, name: str, state: dict[str, Any]) -> None:
        """节点开始执行"""
        self._start_times[name] = time.time()
        self._emit(
            event="node_start",
            node=name,
            state_summary=_summarize(state),
        )

    def node_end(
        self, name: str, state: dict[str, Any], duration_ms: Optional[int] = None
    ) -> None:
        """节点执行完成"""
        if duration_ms is None:
            start = self._start_times.pop(name, time.time())
            duration_ms = int((time.time() - start) * 1000)

        self._emit(
            event="node_end",
            node=name,
            duration_ms=duration_ms,
            state_summary=_summarize(state),
        )

    def node_error(self, name: str, error: str) -> None:
        """节点执行异常"""
        self._emit(
            event="node_error",
            node=name,
            error=error,
        )

# ── 全局单例 ──
_logger: Optional[NodeLogger] = None


def get_logger(session_id: str = "default") -> NodeLogger:
    global _logger
    if _logger is None or _logger.session_id != session_id:
        _logger = NodeLogger(session_id)
    return _logger
