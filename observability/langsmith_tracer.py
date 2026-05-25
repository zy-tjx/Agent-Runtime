"""
最小化追踪器
记录每次节点执行的耗时、产出字段、工具调用，输出结构化 trace
"""
import time
import json as _json
from typing import Any, Callable


class TraceCollector:
    """收集每次节点执行的追踪数据"""

    def __init__(self):
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        node: str,
        duration_ms: int,
        fields_produced: list[str], # 产生新字段
        state_diff: dict[str, Any], # 状态变化
        tool_calls: list[dict] | None = None,  # 调用的工具记录
    ) -> None:
        """记录一次节点追踪"""
        rec = {
            "node": node,
            "duration_ms": duration_ms,
            "fields_produced": fields_produced,
            "state_diff": state_diff,
            "tool_calls": tool_calls or [],
        }
        self._records.append(rec)
        print(_json.dumps(rec, ensure_ascii=False))

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def summary(self) -> dict[str, Any]:
        """返回整条 trace 的汇总"""
        total_ms = sum(r["duration_ms"] for r in self._records)
        nodes_visited = [r["node"] for r in self._records]
        tool_calls_total = sum(len(r["tool_calls"]) for r in self._records)
        return {
            "total_duration_ms": total_ms,
            "nodes_visited": nodes_visited,
            "node_count": len(self._records),
            "tool_calls_total": tool_calls_total,
        }


# ── 全局单例 ──

_tracer: TraceCollector | None = None


def get_tracer() -> TraceCollector:
    global _tracer
    if _tracer is None:
        _tracer = TraceCollector()
    return _tracer


def reset_tracer() -> None:
    """每次图运行前重置（避免跨会话污染）"""
    global _tracer
    _tracer = TraceCollector()


# ── 状态差异计算 ──

def _compute_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """计算 before → after 中发生变化的字段"""
    diff: dict[str, Any] = {}
    all_keys = set(before.keys()) | set(after.keys())
    for key in all_keys:
        b_val = before.get(key)
        a_val = after.get(key)
        if key == "messages":
            b_len = len(b_val) if b_val else 0
            a_len = len(a_val) if a_val else 0
            if b_len != a_len:
                diff[key] = f"{b_len} → {a_len} 条消息"
        elif b_val != a_val:
            if isinstance(a_val, (list, dict)):
                diff[key] = "updated"
            else:
                diff[key] = {"before": b_val, "after": a_val}
    return diff


def _extract_tool_calls(result: dict[str, Any]) -> list[dict]:
    """从节点返回值中提取工具调用记录"""
    calls = result.get("tool_calls")
    if calls and isinstance(calls, list):
        return [
            {
                "tool_name": c.get("tool_name", "unknown"),
                "called_at": c.get("called_at", ""),
            }
            for c in calls
        ]
    return []


# ── 节点包装器 ──

def trace_node(name: str, node_func: Callable) -> Callable:
    """
    包装节点函数，自动记录追踪数据

    用法:
        graph.add_node("planner", trace_node("planner", planner_node))
    """

    def wrapper(state: dict[str, Any]) -> dict[str, Any]:
        tracer = get_tracer()
        start = time.time()

        try:
            result = node_func(state)
        except Exception:
            raise

        duration_ms = int((time.time() - start) * 1000)

        # 合并完整状态以计算 diff
        merged = {**state, **result}
        diff = _compute_diff(state, merged)
        fields_produced = list(result.keys())
        tool_calls = _extract_tool_calls(result)

        tracer.record(
            node=name,
            duration_ms=duration_ms,
            fields_produced=fields_produced,
            state_diff=diff,
            tool_calls=tool_calls,
        )
        return result

    return wrapper
