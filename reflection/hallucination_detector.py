"""
幻觉检测器（规则驱动）
基于治理字段和检索上下文做确定性推理，不依赖 LLM
"""
from typing import Any


# ── 默认阈值 ──

DEFAULT_GROUNDEDNESS_THRESHOLD = 0.4
DEFAULT_COMPLETENESS_THRESHOLD = 0.5


def detect(state: dict[str, Any]) -> dict[str, Any]:
    """
    检测本轮回答是否存在幻觉嫌疑

    Args:
        state: 最终 AgentState（含治理字段 + retrieved_context + final_output）

    Returns:
        {
            "flag": bool,           # 是否触发幻觉告警
            "reason": str,          # 触发原因（中文描述）
            "severity": str,        # "high" | "medium" | "low" | "none"
            "rules_triggered": [],  # 触发了哪些规则
            "evidence": {...},      # 具体证据字段
        }
    """
    groundedness = state.get("groundedness_score")
    completeness = state.get("completeness_score")
    answer_source = state.get("answer_source")
    retrieved_context = state.get("retrieved_context", [])
    final_output = state.get("final_output") or ""

    results: list[dict] = []

    # ── 规则 1（主规则） ──
    results.append(_rule_rag_hallucination(
        groundedness, completeness, answer_source, retrieved_context
    ))

    # ── 规则 2：声称 RAG 但无检索上下文 ──
    results.append(_rule_empty_context(answer_source, retrieved_context))

    # ── 规则 3：回答很长但接地极低 ──
    results.append(_rule_long_answer_low_groundedness(
        groundedness, final_output
    ))

    # ── 汇总 ──
    triggered = [r for r in results if r["triggered"]]
    if not triggered:
        return {
            "flag": False,
            "reason": "未检测到幻觉嫌疑",
            "severity": "none",
            "rules_triggered": [],
            "evidence": {
                "groundedness_score": groundedness,
                "completeness_score": completeness,
                "answer_source": answer_source,
                "context_docs_count": len(retrieved_context),
            },
        }

    # 取最高严重度
    severity_order = {"high": 0, "medium": 1, "low": 2}
    worst = min(triggered, key=lambda r: severity_order.get(r["severity"], 99))

    return {
        "flag": True,
        "reason": "；".join(r["reason"] for r in triggered),
        "severity": worst["severity"],
        "rules_triggered": [r["name"] for r in triggered],
        "evidence": {
            "groundedness_score": groundedness,
            "completeness_score": completeness,
            "answer_source": answer_source,
            "context_docs_count": len(retrieved_context),
            "answer_length_chars": len(final_output),
        },
    }


# ── 规则实现 ──

def _rule_rag_hallucination(
    groundedness: float | None,
    completeness: float | None,
    answer_source: str | None,
    retrieved_context: list,
) -> dict[str, Any]:
    """
    规则 1（主规则）：
    groundedness 低 + answer_source=rag + 有检索上下文 + completeness 尚可
    → 模型可能脱离文档内容编造回答
    """
    if (
        groundedness is not None
        and groundedness < DEFAULT_GROUNDEDNESS_THRESHOLD
        and answer_source == "rag"
        and len(retrieved_context) > 0
        and (completeness is None or completeness >= DEFAULT_COMPLETENESS_THRESHOLD)
    ):
        return {
            "name": "rag_hallucination",
            "triggered": True,
            "severity": "high",
            "reason": (
                f"RAG 回答接地分低（{groundedness:.2f}<{DEFAULT_GROUNDEDNESS_THRESHOLD}）"
                f"但完整度正常（{completeness:.2f}），疑似脱离检索内容编造"
            ),
        }
    return {"name": "rag_hallucination", "triggered": False, "severity": "none", "reason": ""}


def _rule_empty_context(
    answer_source: str | None,
    retrieved_context: list,
) -> dict[str, Any]:
    """
    规则 2：声称来源 RAG 但 retrieved_context 为空
    → 来源标记与实际情况矛盾
    """
    if answer_source == "rag" and len(retrieved_context) == 0:
        return {
            "name": "empty_context_mismatch",
            "triggered": True,
            "severity": "medium",
            "reason": "answer_source=rag 但 retrieved_context 为空，来源标记矛盾",
        }
    return {"name": "empty_context_mismatch", "triggered": False, "severity": "none", "reason": ""}


def _rule_long_answer_low_groundedness(
    groundedness: float | None,
    final_output: str,
) -> dict[str, Any]:
    """
    规则 3：回答长度 > 200 字符但 groundedness 极低（< 0.2）
    → 长篇回答几乎没有文档支撑
    """
    if (
        groundedness is not None
        and groundedness < 0.2
        and len(final_output) > 200
    ):
        return {
            "name": "long_answer_low_groundedness",
            "triggered": True,
            "severity": "medium",
            "reason": (
                f"回答较长（{len(final_output)} 字符）"
                f"但接地分极低（{groundedness:.2f}<0.2），大段内容可能无据可查"
            ),
        }
    return {"name": "long_answer_low_groundedness", "triggered": False, "severity": "none", "reason": ""}
