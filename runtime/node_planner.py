"""
PLANNER 节点
根据用户输入判断意图（learn / qa），产出 plan 和 mode 字段

执行策略：
  1. 主路径：LLM 判断意图 + 生成计划
  2. 降级路径：LLM 失败时关键词匹配 + 内置模板
"""
from typing import Any
from pydantic import BaseModel, Field

from engine.prompt_manager import render
from engine.model_manager import ModelManager
from runtime.node_decide import _parse_json
from observability.logger import get_logger


# ── Structured Output 模型 ──

class PlanModule(BaseModel):
    """学习计划中的单个模块"""
    title: str
    content: str
    duration_minutes: int = Field(ge=1, le=180)


class PlanOutput(BaseModel):
    """LLM 规划的输出结构"""
    mode: str = "learn"  # "learn" | "qa"
    topic: str
    goals: list[str] = []
    modules: list[PlanModule] = []
    estimated_total_minutes: int = 0


# ── 模块级单例 ──

_model_manager = None


def _get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# ── 降级路径：关键词匹配 ──

_QA_KEYWORDS = ["什么是", "介绍一下", "如何", "怎么", "是什么", "有什么用", "区别", "对比"]


def _detect_mode_keyword(user_input: str) -> str:
    """关键词判断意图：learn 或 qa"""
    for kw in _QA_KEYWORDS:
        if kw in user_input:
            return "qa"
    return "learn"


def _plan_fallback(user_input: str, user_level: str, goals: list[str] | None) -> dict[str, Any]:
    """LLM 不可用时返回模板化计划"""
    mode = _detect_mode_keyword(user_input)
    level_modules = {
        "beginner": [
            {"title": "概念入门", "content": "核心概念与术语解释", "duration_minutes": 20},
            {"title": "环境搭建", "content": "开发环境配置与第一个示例", "duration_minutes": 25},
            {"title": "基础实战", "content": "简单场景的动手练习", "duration_minutes": 15},
        ],
        "intermediate": [
            {"title": "原理深入", "content": "底层机制与架构分析", "duration_minutes": 25},
            {"title": "进阶实战", "content": "复杂场景的工程实践", "duration_minutes": 30},
            {"title": "性能优化", "content": "瓶颈分析与调优策略", "duration_minutes": 20},
        ],
        "advanced": [
            {"title": "源码剖析", "content": "核心模块源码解读", "duration_minutes": 30},
            {"title": "系统设计", "content": "大规模系统架构设计", "duration_minutes": 35},
            {"title": "前沿探索", "content": "最新论文与研究方向", "duration_minutes": 25},
        ],
    }
    modules = level_modules.get(user_level, level_modules["beginner"])
    if mode == "qa":
        modules = []
    return {
        "mode": mode,
        "topic": user_input,
        "goals": goals or ["理解核心概念", "掌握实践技能", "能够独立应用"],
        "modules": modules,
        "estimated_total_minutes": sum(m["duration_minutes"] for m in modules) if modules else 0,
    }


# ── 节点主函数 ──

def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """基于用户输入判断意图并生成计划"""
    logger = get_logger()
    logger.node_start("PLANNER", state)

    user_input = state.get("user_input", "")
    plan = state.get("plan") or {}
    user_level = plan.get("user_level", "beginner") if isinstance(plan, dict) else "beginner"
    goals = plan.get("goals", []) if isinstance(plan, dict) else []

    # ── 环境变量控制：测试时跳过 LLM ──
    import os
    if os.getenv("DECIDE_USE_KEYWORD", "") == "1":
        plan_data = _plan_fallback(user_input, user_level, goals)
        result = _build_plan_result(plan_data, via="模板降级")
        logger.node_end("PLANNER", {**state, **result})
        return result

    # ── 提取对话历史 ──
    stm = state.get("short_term_memory", {}) or {}
    history = stm.get("buffer", [])
    conversation_history = _format_history(history)

    # ── 优先走 LLM 路径 ──
    try:
        plan_data = _plan_via_llm(user_input, user_level, goals, conversation_history)
        result = _build_plan_result(plan_data, via="LLM")
        logger.node_end("PLANNER", {**state, **result})
        return result
    except Exception as e:
        logger.info(f"LLM 路径失败，降级到模板匹配: {e}")
        fallback_error = f"PLANNER: {type(e).__name__}"

    # ── 降级 ──
    plan_data = _plan_fallback(user_input, user_level, goals)
    result = _build_plan_result(plan_data, via="模板降级")
    result["fallback_triggered"] = True
    result["fallback_reason"] = fallback_error
    logger.node_end("PLANNER", {**state, **result})
    return result


def _plan_via_llm(
    user_input: str, user_level: str, goals: list[str] | None,
    conversation_history: str = "（无历史对话）",
) -> dict[str, Any]:
    """LLM 驱动路径"""
    model = _get_model_manager()
    goals_text = ", ".join(goals) if goals else "未指定，请根据主题自动生成"

    prompt = render(
        "planner_plan",
        user_input=user_input,
        user_level=user_level,
        goals=goals_text,
        conversation_history=conversation_history,
    )
    result_text = model.generate(prompt)
    plan_obj = _parse_json(result_text, PlanOutput)
    return plan_obj.model_dump()


def _build_plan_result(plan_data: dict[str, Any], via: str) -> dict[str, Any]:
    """组装 plan 节点返回结构"""
    import time
    plan_data["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    mode = plan_data.pop("mode", "learn")
    return {
        "current_step": "PLANNER",
        "mode": mode,
        "plan": plan_data,
        "messages": [
            {"role": "ai", "content": f"意图={mode}，已生成计划（{via}）"}
        ],
    }


def _format_history(buffer: list[dict]) -> str:
    """将对话历史列表格式化为 prompt 可读文本"""
    if not buffer:
        return "（无历史对话）"
    lines = []
    for msg in buffer[-10:]:  # 最多 10 轮
        role = "用户" if msg.get("role") == "human" else "AI"
        content = msg.get("content", "")
        lines.append(f"{role}: {content[:200]}")
    return "\n".join(lines)
