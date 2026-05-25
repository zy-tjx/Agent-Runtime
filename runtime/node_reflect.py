"""
REFLECT 节点
负责自我反思与评估，产出 reflection 字段及治理字段
retry_count 在此处自增（每次进入 REFLECT +1）

执行策略：
  1. 规则引擎预分析（error_analysis + improvement_suggestion + self_reflection）
  2. LLM 路径：接收规则建议作为上下文，可覆盖但需说明理由
  3. 降级路径：LLM 失败时使用规则引擎结果（而非硬编码 end）
"""
import json
from typing import Any
from pydantic import BaseModel, Field

from engine.prompt_manager import render
from engine.model_manager import ModelManager
from runtime.node_decide import _parse_json
from tools.memory_write import run as memory_write_run
from tools.memory_write import MemoryWriteInput
from observability.logger import get_logger
from reflection.error_analysis import analyze
from reflection.improvement_suggestion import suggest
from reflection.self_reflection import evaluate
from memory.long_term_memory import load_experience_summaries


# ── Structured Output 模型 ──

class ReflectOutput(BaseModel):
    """learn 模式的反思评估输出"""
    confidence: float = Field(ge=0.0, le=1.0)
    is_satisfactory: bool
    error_root_cause: str | None = None
    improvement_suggestion: str | None = None
    next_action: str  # "end" | "retry"
    retry_target_node: str | None = None
    hallucination_flag: bool = False


class QAReflectOutput(BaseModel):
    """qa 模式的反思评估输出"""
    groundedness_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    is_satisfactory: bool
    next_action: str  # "end" | "retry"
    retry_target_node: str | None = None
    retry_reason: str | None = None


# ── 模块级单例 ──

_model_manager = None


def _get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# ── 规则引擎降级输出构建 ──

def _build_from_rule(
    mode: str,
    retry_count: int,
    err: dict[str, Any],
    rule_suggestion: dict[str, Any],
    confidence: dict[str, Any],
) -> dict[str, Any]:
    """用规则引擎结果构建 REFLECT 输出（LLM 不可用时的降级路径）"""
    if mode == "qa":
        reflection = {
            "groundedness_score": 0.5,
            "completeness_score": 0.5,
            "confidence": confidence["confidence"],
            "is_satisfactory": rule_suggestion["next_action"] == "end",
            "next_action": rule_suggestion["next_action"],
            "retry_target_node": rule_suggestion["retry_target_node"],
            "retry_reason": rule_suggestion["rationale"],
        }
        return _build_qa_result(reflection, retry_count, via="规则引擎")
    else:
        reflection = {
            "confidence": confidence["confidence"],
            "is_satisfactory": rule_suggestion["next_action"] == "end",
            "error_root_cause": err["summary"] if err["error_type"] != "none" else None,
            "improvement_suggestion": rule_suggestion["rationale"],
            "next_action": rule_suggestion["next_action"],
            "retry_target_node": rule_suggestion["retry_target_node"],
            "hallucination_flag": False,
        }
        return _build_learn_result(reflection, retry_count, via="规则引擎")


def _make_rule_context(
    err: dict[str, Any],
    suggestion: dict[str, Any],
    confidence: dict[str, Any],
) -> str:
    """将规则引擎输出格式化为 Prompt 可嵌入文本"""
    return (
        f"错误类型={err['error_type']}, "
        f"是否可恢复={err['recoverable']}, "
        f"建议动作={suggestion['next_action']}, "
        f"建议目标={suggestion['retry_target_node']}, "
        f"原因码={suggestion['reason_code']}, "
        f"置信度={confidence['confidence']}"
    )


# ── 节点主函数 ──

def reflect_node(state: dict[str, Any]) -> dict[str, Any]:
    """自我反思与评估（规则预分析 → LLM 覆盖 → 规则降级）"""
    logger = get_logger()
    logger.node_start("REFLECT", state)

    retry_count = state.get("retry_count", 0)
    new_retry_count = retry_count + 1
    max_retries = state.get("max_retries", 3)
    mode = state.get("mode", "learn")
    user_input = state.get("user_input", "")

    # ── 步骤 1: 规则引擎预分析 ──
    err = analyze(
        error=state.get("error"),
        tool_results=state.get("tool_results"),
        retrieval_score=state.get("retrieval_score"),
        retrieved_context=state.get("retrieved_context"),
        retrieval_attempted=state.get("retrieval_score") is not None
                            or bool(state.get("vector_search_results")),
        retry_count=new_retry_count,
        max_retries=max_retries,
        fallback_triggered=state.get("fallback_triggered", False),
        fallback_reason=state.get("fallback_reason"),
    )
    rule_suggestion = suggest(
        error_analysis=err,
        mode=mode,
        groundedness_score=state.get("groundedness_score"),
        completeness_score=state.get("completeness_score"),
        hallucination_flag=False,
        fallback_reason=state.get("fallback_reason"),
    )
    confidence = evaluate(
        groundedness_score=state.get("groundedness_score"),
        completeness_score=state.get("completeness_score"),
        fallback_triggered=state.get("fallback_triggered", False),
        hallucination_flag=False,
    )

    # ── 步骤 2: LLM 路径（规则建议作为上下文） ──
    import os
    if os.getenv("DECIDE_USE_KEYWORD", "") == "1":
        result = _build_from_rule(mode, new_retry_count, err, rule_suggestion, confidence)
    else:
        try:
            if mode == "qa":
                result = _reflect_qa_path(
                    state, new_retry_count, max_retries, rule_suggestion, confidence
                )
            else:
                result = _reflect_learn_path(
                    state, new_retry_count, max_retries, rule_suggestion, confidence
                )
        except Exception as e:
            logger.info(f"LLM 反思路径失败，降级到规则引擎: {e}")
            result = _build_from_rule(mode, new_retry_count, err, rule_suggestion, confidence)
            result["fallback_triggered"] = True
            result["fallback_reason"] = f"REFLECT: {type(e).__name__}"

    # ── 写入长期记忆（单写源） ──
    _write_to_memory(mode, user_input, result.get("reflection", {}), new_retry_count)
    logger.node_end("REFLECT", {**state, **result})
    return result


def _write_to_memory(mode: str, user_input: str, reflection: dict, retry_count: int) -> None:
    """将反思结果写入长期记忆（fire-and-forget，失败不阻流程）"""
    import time
    try:
        memory_write_run(MemoryWriteInput(
            key=f"reflect_{time.strftime('%Y%m%dT%H%M%S')}_{retry_count}",
            value={
                "mode": mode,
                "user_input": user_input,
                "reflection": reflection,
            },
            category="experience",
        ))
    except Exception:
        pass  # 写入失败不阻断状态机流转


def _reflect_learn_path(
    state: dict, retry_count: int, max_retries: int,
    rule_suggestion: dict[str, Any],
    confidence: dict[str, Any],
) -> dict[str, Any]:
    """learn 模式 LLM 路径（含规则建议上下文）"""
    user_input = state.get("user_input", "")
    tool_results = state.get("tool_results", [])
    error = state.get("error") or "无"

    model = _get_model_manager()
    tool_results_text = json.dumps(tool_results, ensure_ascii=False, indent=2)

    # 提取错误诊断中的 summary 作为规则上下文
    err = analyze(
        error=state.get("error"),
        tool_results=state.get("tool_results"),
        retry_count=retry_count,
        max_retries=max_retries,
        fallback_triggered=state.get("fallback_triggered", False),
    )
    rule_context = _make_rule_context(err, rule_suggestion, confidence)
    experience_context = "\n".join(
        f"- {s}" for s in load_experience_summaries(limit=3, mode="learn")
    )

    prompt = render(
        "reflect_eval",
        user_input=user_input,
        tool_results=tool_results_text,
        error=error,
        retry_info=f"{retry_count} / {max_retries}",
        rule_context=rule_context,
        experience_context=experience_context,
    )
    result_text = model.generate(prompt)
    reflection = _parse_json(result_text, ReflectOutput)
    return _build_learn_result(reflection.model_dump(), retry_count, via="LLM")


def _reflect_qa_path(
    state: dict, retry_count: int, max_retries: int,
    rule_suggestion: dict[str, Any],
    confidence: dict[str, Any],
) -> dict[str, Any]:
    """qa 模式 LLM 路径：评估 groundedness + completeness（含规则建议上下文）"""
    user_input = state.get("user_input", "")
    retrieved_context = state.get("retrieved_context", [])
    final_output = state.get("final_output", "")
    answer_source = state.get("answer_source", "rag")

    model = _get_model_manager()
    context_text = json.dumps(retrieved_context, ensure_ascii=False, indent=2)

    err = analyze(
        error=state.get("error"),
        retrieval_score=state.get("retrieval_score"),
        retrieved_context=retrieved_context,
        retrieval_attempted=state.get("retrieval_score") is not None,
        retry_count=retry_count,
        max_retries=max_retries,
        fallback_triggered=state.get("fallback_triggered", False),
    )
    rule_context = _make_rule_context(err, rule_suggestion, confidence)
    experience_context = "\n".join(
        f"- {s}" for s in load_experience_summaries(limit=3, mode="qa")
    )

    prompt = render(
        "reflect_qa_eval",
        user_input=user_input,
        retrieved_context=context_text,
        final_output=final_output,
        answer_source=answer_source,
        retry_info=f"{retry_count} / {max_retries}",
        rule_context=rule_context,
        experience_context=experience_context,
    )
    result_text = model.generate(prompt)
    qa_reflect = _parse_json(result_text, QAReflectOutput)
    return _build_qa_result(qa_reflect.model_dump(), retry_count, via="LLM")


# ── 组装返回 ──

def _build_learn_result(
    reflection: dict[str, Any],
    retry_count: int,
    via: str,
) -> dict[str, Any]:
    """组装 learn 模式 REFLECT 返回"""
    is_satisfactory = reflection.get("is_satisfactory", True)
    next_action = reflection.get("next_action", "end")

    return {
        "current_step": "REFLECT",
        "retry_count": retry_count,
        "reflection": reflection,
        "final_output": (
            f"学习计划已完成。评估置信度：{reflection.get('confidence', 0.5)}"
            if is_satisfactory or next_action == "end"
            else f"执行未达预期。{reflection.get('improvement_suggestion', '建议重试')}"
        ),
        "messages": [
            {"role": "ai", "content": f"反思完成（{via}）：满意度={is_satisfactory}, 决策={next_action}"}
        ],
    }


def _build_qa_result(
    reflection: dict[str, Any],
    retry_count: int,
    via: str,
) -> dict[str, Any]:
    """组装 qa 模式 REFLECT 返回，含治理字段"""
    next_action = reflection.get("next_action", "end")

    return {
        "current_step": "REFLECT",
        "retry_count": retry_count,
        "reflection": reflection,
        "groundedness_score": reflection.get("groundedness_score"),
        "completeness_score": reflection.get("completeness_score"),
        "retry_reason": reflection.get("retry_reason"),
        "messages": [
            {
                "role": "ai",
                "content": (
                    f"QA 反思完成（{via}）："
                    f"groundedness={reflection.get('groundedness_score')}, "
                    f"completeness={reflection.get('completeness_score')}, "
                    f"决策={next_action}"
                ),
            }
        ],
    }
