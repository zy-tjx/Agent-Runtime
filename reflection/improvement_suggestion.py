"""
建议生成器
纯函数：输入错误诊断 + 治理信号，输出 next_action 建议，不依赖 LLM
"""
from typing import Any


# ── reason_code 枚举 ──

LOW_GROUNDEDNESS = "LOW_GROUNDEDNESS"   #低接地分
EMPTY_RETRIEVAL = "EMPTY_RETRIEVAL"     #检索为空
LLM_ERROR = "LLM_ERROR"                 #大模型错误
TOOL_FAILURE = "TOOL_FAILURE"           #工具错误
MAX_RETRIES = "MAX_RETRIES"             #最大重试次数
NONE = "NONE"                           #无错误


def suggest(
    error_analysis: dict[str, Any],
    mode: str = "learn",
    groundedness_score: float | None = None,
    completeness_score: float | None = None,
    hallucination_flag: bool = False,   #是否发生幻觉告警
    fallback_reason: str | None = None, #触发兜底的原因
) -> dict[str, Any]:
    """
    基于错误诊断生成下一步行动建议

    Args:
        error_analysis: error_analysis.analyze() 的输出
        mode: 当前模式 learn / qa
        groundedness_score: 接地分 0~1
        completeness_score: 完整度 0~1
        hallucination_flag: 幻觉检测是否触发
        fallback_reason: LLM 降级原因（含节点名）

    Returns:
        {
            "next_action": "end" | "retry",
            "retry_target_node": "PLANNER" | "RETRIEVE" | "EXECUTE" | None,
            "reason_code": str,
            "rationale": str,
        }
    """
    error_type = error_analysis.get("error_type", "none")
    recoverable = error_analysis.get("recoverable", False)

    # ── 有错误且不可恢复 → 直接结束 ──
    if error_type != "none" and not recoverable:
        return _end(MAX_RETRIES, "错误不可恢复，停止重试")

    # ── 可恢复错误，按类型路由重试目标 ──
    if error_type == "llm_error":
        target = _infer_target_from_fallback(fallback_reason)
        return _retry(LLM_ERROR, target, f"LLM 调用失败，建议重试 {target}")

    if error_type == "tool_failure":
        return _retry(TOOL_FAILURE, "EXECUTE", "工具执行失败，建议重试执行节点")

    if error_type == "retrieval_empty":
        return _retry(EMPTY_RETRIEVAL, "RETRIEVE", "检索为空，建议重试检索节点")

    # ── 无显式错误，检查质量信号是否需要主动重试 ──
    if groundedness_score is not None and groundedness_score < 0.4:
        if mode == "qa" and hallucination_flag:
            return _end(LOW_GROUNDEDNESS, f"接地分低({groundedness_score:.2f})且幻觉告警，避免继续编造")
        return _retry(LOW_GROUNDEDNESS, "RETRIEVE",
                      f"接地分低({groundedness_score:.2f})，建议重试检索获取更相关内容")

    return _end(NONE, "无错误信号，建议结束")


# ── 内部辅助 ──

def _infer_target_from_fallback(fallback_reason: str | None) -> str:
    """从降级原因推断应重试哪个节点"""
    if not fallback_reason:
        return "PLANNER"
    reason = fallback_reason.upper()
    if "PLANNER" in reason:
        return "PLANNER"
    if "DECIDE" in reason:
        return "PLANNER"  # DECIDE 不是合法重试目标，回退到 PLANNER
    if "REFLECT" in reason:
        return "PLANNER"  # REFLECT 自身失败，保守回退
    return "PLANNER"


def _end(code: str, rationale: str) -> dict[str, Any]:
    return {
        "next_action": "end",
        "retry_target_node": None,
        "reason_code": code,
        "rationale": rationale,
    }


def _retry(code: str, target: str, rationale: str) -> dict[str, Any]:
    return {
        "next_action": "retry",
        "retry_target_node": target,
        "reason_code": code,
        "rationale": rationale,
    }
