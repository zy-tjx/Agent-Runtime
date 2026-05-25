"""
DECIDE 节点
基于用户输入和可用工具列表做策略决策，产出 decision 字段

执行策略：
  1. 主路径：LLM 驱动（工具选择 + 参数生成），输出受 Pydantic 约束
  2. 降级路径：LLM 失败时回退到关键词规则匹配
"""
from typing import Any
from pydantic import BaseModel, Field

from tools.tool_registry import create_default_registry
from engine.prompt_manager import render, format_tools_list, format_parameters_schema
from engine.model_manager import ModelManager
from observability.logger import get_logger


# ── Structured Output 模型 ──

class DecideToolSelection(BaseModel):
    """LLM 工具选择的输出结构"""

    tool_name: str = Field(description="选择的工具名，必须是可用工具列表中的某个 name")
    reason: str = Field(description="选择该工具的简短理由")
    confidence: float = Field(description="置信度 0.0~1.0", ge=0.0, le=1.0)


# ── 模块级单例 ──

_registry = None
_model_manager = None


def _get_registry():
    global _registry
    if _registry is None:
        _registry = create_default_registry()
    return _registry


def _get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# ── 降级路径：关键词匹配 ──

_KEYWORD_TOOL_MAP: list[tuple[str, str, dict[str, str]]] = [
    ("检索", "search_docs", {"query": "user_input"}),
    ("查", "search_docs", {"query": "user_input"}),
    ("找", "search_docs", {"query": "user_input"}),
    ("了解", "search_docs", {"query": "user_input"}),
    ("学习", "search_docs", {"query": "plan_topic"}),
    ("计划", "search_docs", {"query": "plan_topic"}),
    ("总结", "search_docs", {"query": "user_input"}),
    ("测验", "search_docs", {"query": "plan_topic"}),
]


def _match_tool(user_input: str) -> tuple[str, dict[str, str], str]:
    for keyword, tool_name, arg_sources in _KEYWORD_TOOL_MAP:
        if keyword in user_input:
            return tool_name, arg_sources, f"用户输入含「{keyword}」，匹配工具 {tool_name}"
    return "search_docs", {"query": "user_input"}, "无明确关键词，默认检索"


def _build_arguments(
    arg_sources: dict[str, str], user_input: str, plan: dict[str, Any]
) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for param_name, source in arg_sources.items():
        if source == "user_input":
            args[param_name] = user_input
        elif source == "plan_topic":
            topic = plan.get("topic", "") if plan else ""
            args[param_name] = topic or user_input
        elif source.startswith("fixed_"):
            args[param_name] = source[len("fixed_"):]
    return args


# ── 节点主函数 ──

def decide_node(state: dict[str, Any]) -> dict[str, Any]:
    """基于可用工具列表做策略决策"""
    logger = get_logger()
    logger.node_start("DECIDE", state)

    registry = _get_registry()
    user_input = state.get("user_input", "")
    plan = state.get("plan", {})

    # ── 提取对话历史 ──
    from runtime.node_planner import _format_history
    stm = state.get("short_term_memory", {}) or {}
    conversation_history = _format_history(stm.get("buffer", []))

    # ── 环境变量控制：测试时跳过 LLM 加速 ──
    import os
    if os.getenv("DECIDE_USE_KEYWORD", "") == "1":
        result = _decide_via_keyword(registry, user_input, plan)
        logger.node_end("DECIDE", {**state, **result})
        return result

    # ── 优先走 LLM 路径 ──
    try:
        result = _decide_via_llm(registry, user_input, plan, conversation_history)
        logger.node_end("DECIDE", {**state, **result})
        return result
    except Exception as e:
        logger.info(f"LLM 路径失败，降级到关键词匹配: {e}")
        fallback_error = f"DECIDE: {type(e).__name__}"

    # ── 降级路径：关键词匹配 ──
    result = _decide_via_keyword(registry, user_input, plan)
    result["fallback_triggered"] = True
    result["fallback_reason"] = fallback_error
    logger.node_end("DECIDE", {**state, **result})
    return result


def _decide_via_llm(
    registry, user_input: str, plan: dict[str, Any],
    conversation_history: str = "（无历史对话）",
) -> dict[str, Any]:
    """LLM 驱动路径：工具选择 + 参数生成（generate + JSON 解析，兼容任意模型）"""
    model = _get_model_manager()
    tools = registry.list_tools()

    # ── 拼接 user_input 与 plan 上下文 ──
    topic = plan.get("topic", "") if plan else ""
    full_context = f"用户输入: {user_input}" + (f"；当前学习主题: {topic}" if topic else "")

    tools_list = format_tools_list(tools)

    # ── Step A：工具选择（文本输出 → 解析 JSON） ──
    selection_prompt = render(
        "decide_tool_selection",
        user_input=full_context,
        tools_list=tools_list,
        conversation_history=conversation_history,
    )
    selection_text = model.generate(selection_prompt)
    selection = _parse_json(selection_text, DecideToolSelection)

    # ── Step B：校验工具存在 ──
    tool = registry.get(selection.tool_name)
    if tool is None:
        raise ValueError(f"LLM 选了未注册的工具: {selection.tool_name}")

    # ── Step C：参数生成（文本输出 → 用工具 input_schema 校验） ──
    tool_params = next(t for t in tools if t["name"] == selection.tool_name)
    arguments_prompt = render(
        "decide_arguments",
        user_input=full_context,
        name=tool.name,
        description=tool.description,
        parameters_schema=format_parameters_schema(tool_params["parameters"]),
    )
    arguments_text = model.generate(arguments_prompt)
    validated_args = _parse_json(arguments_text, tool.input_schema)

    # ── Step D：组装决策 ──
    return _build_decision(
        tool_name=selection.tool_name,
        arguments=validated_args.model_dump(),
        reason=selection.reason,
        confidence=selection.confidence,
    )


def _parse_json(text: str, model_class):
    """从 LLM 输出中提取 JSON 并用 Pydantic 模型校验"""
    import json as _json
    text = text.strip()
    # 去掉可能的 markdown 代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        # 尝试提取第一个完整 JSON 对象
        import re
        match = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}", text, re.DOTALL)
        if match:
            data = _json.loads(match.group())
        else:
            raise ValueError(f"无法解析 LLM 输出的 JSON: {text[:200]}")
    return model_class(**data)


def _decide_via_keyword(
    registry, user_input: str, plan: dict[str, Any]
) -> dict[str, Any]:
    """降级路径：关键词规则匹配"""
    tool_name, arg_sources, reason = _match_tool(user_input)

    if not registry.exists(tool_name):
        tool_name = "search_docs"
        arg_sources = {"query": "user_input"}
        reason = "匹配的工具不可用，回退到 search_docs"

    arguments = _build_arguments(arg_sources, user_input, plan)
    return _build_decision(
        tool_name=tool_name,
        arguments=arguments,
        reason=reason,
        confidence=0.75,
    )


def _build_decision(
    tool_name: str,
    arguments: dict[str, Any],
    reason: str,
    confidence: float,
) -> dict[str, Any]:
    """组装 decision 返回结构"""
    requires_retrieval = tool_name == "search_docs"
    return {
        "current_step": "DECIDE",
        "decision": {
            "action": tool_name,
            "tool_name": tool_name,
            "arguments": arguments,
            "reason": reason,
            "confidence": confidence,
            "requires_retrieval": requires_retrieval,
        },
        "messages": [
            {"role": "ai", "content": f"决定调用 {tool_name}：{reason}"}
        ],
    }
