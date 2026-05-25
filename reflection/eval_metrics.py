"""
确定性评估指标计算
基于最终 AgentState 计算 RAG/工具/回答/流程/治理 5 类指标，不依赖 LLM
"""
from typing import Any


def compute_metrics(state: dict[str, Any]) -> dict[str, Any]:
    """
    从最终 AgentState 一次性计算所有确定性指标

    Args:
        state: 图运行结束后的完整 AgentState

    Returns:
        {
            "rag": {...},           # RAG 检索指标
            "tool": {...},          # 工具执行指标
            "answer": {...},        # 回答质量指标
            "flow": {...},          # 流程指标
            "governance": {...},    # 治理字段指标
        }
    """
    return {
        "rag": _rag_metrics(state),
        "tool": _tool_metrics(state),
        "answer": _answer_metrics(state),
        "flow": _flow_metrics(state),
        "governance": _governance_metrics(state),
    }


# ── RAG 检索指标 ──

def _rag_metrics(state: dict[str, Any]) -> dict[str, Any]:
    retrieved = state.get("retrieved_context", [])
    score = state.get("retrieval_score")
    metadata = state.get("rag_metadata") or {}

    return {
        "docs_retrieved": len(retrieved),
        "retrieval_score": score,
        "has_context": len(retrieved) > 0,
        "total_docs_after_rerank": metadata.get("total_docs_after_rerank", len(retrieved)),
        "vector_search_latency_ms": metadata.get("vector_search_latency_ms", 0),
        "rerank_latency_ms": metadata.get("rerank_latency_ms", 0),
    }


# ── 工具执行指标 ──

def _tool_metrics(state: dict[str, Any]) -> dict[str, Any]:
    results = state.get("tool_results", [])
    calls = state.get("tool_calls", [])

    success_count = sum(1 for r in results if r.get("status") == "success")
    total_duration = sum(r.get("duration_ms", 0) for r in results)
    tool_names = list({c.get("tool_name", "unknown") for c in calls})

    return {
        "tool_calls_total": len(calls),
        "tool_success_count": success_count,
        "tool_failure_count": len(results) - success_count,
        "tool_success_rate": success_count / len(results) if results else None,
        "total_tool_duration_ms": total_duration,
        "tools_used": tool_names,
    }


# ── 回答质量指标（确定性） ──

def _answer_metrics(state: dict[str, Any]) -> dict[str, Any]:
    output = state.get("final_output") or ""
    source = state.get("answer_source")

    return {
        "has_answer": bool(output),
        "answer_length_chars": len(output),
        "answer_source": source,
        "is_rag_sourced": source == "rag",
        "is_fallback": source == "llm_fallback",
    }


# ── 流程指标 ──

def _flow_metrics(state: dict[str, Any]) -> dict[str, Any]:
    mode = state.get("mode", "learn")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    error = state.get("error")

    # 根据 state 字段是否存在，推断访问过的节点
    steps_seen: list[str] = _infer_nodes_visited(state)

    return {
        "mode": mode,
        "nodes_visited": steps_seen,
        "nodes_visited_count": len(steps_seen),
        "retry_count": retry_count,
        "max_retries": max_retries,
        "did_retry": retry_count > 1,
        "retries_exhausted": retry_count >= max_retries,
        "has_error": error is not None,
        "error": error,
    }


def _infer_nodes_visited(state: dict[str, Any]) -> list[str]:
    """根据 state 中已填充的字段，反推断访问过的节点（按执行顺序）"""
    order = ["PLANNER", "DECIDE", "RETRIEVE", "EXECUTE", "REFLECT"]
    indicators = {
        "PLANNER": lambda s: s.get("plan") is not None,
        "DECIDE": lambda s: s.get("decision") is not None,
        "RETRIEVE": lambda s: (
            s.get("retrieval_score") is not None
            or bool(s.get("vector_search_results"))
            or s.get("rewritten_query") is not None
        ),
        "EXECUTE": lambda s: bool(s.get("tool_results")) or s.get("final_output") is not None,
        "REFLECT": lambda s: s.get("reflection") is not None,
    }
    visited: list[str] = []
    for step in order:
        if indicators.get(step, lambda _: False)(state):
            visited.append(step)
    return visited


# ── 治理字段透传 ──

def _governance_metrics(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": state.get("mode"),
        "retrieval_score": state.get("retrieval_score"),
        "groundedness_score": state.get("groundedness_score"),
        "completeness_score": state.get("completeness_score"),
        "answer_source": state.get("answer_source"),
        "retry_reason": state.get("retry_reason"),
        "fallback_triggered": state.get("fallback_triggered", False),
        "fallback_reason": state.get("fallback_reason"),
    }
