"""
置信度评估器
纯函数：4 因子加权计算综合信心分，每个因子独立可追溯，不依赖 LLM
"""
from typing import Any


# ── 评估权重（可调） ──

_W_GROUNDEDNESS = 0.4
_W_COMPLETENESS = 0.3
_PENALTY_FALLBACK = 0.2
_PENALTY_HALLUCINATION = 0.3


def evaluate(
    groundedness_score: float | None = None,
    completeness_score: float | None = None,
    fallback_triggered: bool = False,
    hallucination_flag: bool = False,
) -> dict[str, Any]:
    """
    计算综合置信度

    Args:
        groundedness_score: 接地分 0~1（REFLECT QA 评估写入）
        completeness_score: 完整度 0~1（REFLECT QA 评估写入）
        fallback_triggered: 是否触发过 LLM 降级
        hallucination_flag: 幻觉检测是否告警

    Returns:
        {
            "confidence": float,         # 综合信心分 0~1
            "factors": {
                "groundedness_contrib": float,   # 接地分贡献
                "completeness_contrib": float,   # 完整度贡献
                "fallback_penalty": float,       # 降级惩罚（≥0）
                "hallucination_penalty": float,  # 幻觉惩罚（≥0）
            },
            "hallucination_penalty_applied": bool,
        }
    """
    # ── 缺失时用中性默认值，不惩罚未评估的场景 ──
    g = groundedness_score if groundedness_score is not None else 0.5
    c = completeness_score if completeness_score is not None else 0.5

    # ── 正向贡献 ──
    groundedness_contrib = round(g * _W_GROUNDEDNESS, 3)
    completeness_contrib = round(c * _W_COMPLETENESS, 3)

    # ── 惩罚项 ──
    fallback_penalty = _PENALTY_FALLBACK if fallback_triggered else 0.0
    hallucination_penalty = _PENALTY_HALLUCINATION if hallucination_flag else 0.0

    # ── 综合 ──
    raw = groundedness_contrib + completeness_contrib - fallback_penalty - hallucination_penalty
    confidence = round(max(0.0, min(1.0, raw)), 3)

    return {
        "confidence": confidence,
        "factors": {
            "groundedness_contrib": groundedness_contrib,
            "completeness_contrib": completeness_contrib,
            "fallback_penalty": -fallback_penalty,
            "hallucination_penalty": -hallucination_penalty,
        },
        "hallucination_penalty_applied": hallucination_flag,
    }
