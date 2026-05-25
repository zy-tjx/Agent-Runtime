"""
Query Rewrite
基于对话历史将省略主语/代词/上下文引用的查询改写为完整独立查询
LLM 驱动，失败时返回原 query 不阻断检索
"""
from engine.model_manager import ModelManager
from engine.prompt_manager import render

_model: ModelManager | None = None


def _get_model() -> ModelManager:
    global _model
    if _model is None:
        _model = ModelManager()
    return _model


def rewrite(query: str, history: list[dict] | None = None) -> str:
    """
    将可能省略上下文的查询改写为完整查询

    Args:
        query: 原始用户查询（可能含"它""刚才那个"等指代）
        history: 最近 N 轮对话历史 [{"role":"human/ai","content":"..."}]

    Returns:
        改写后的查询，失败时返回原 query
    """
    if not history:
        return query

    # 检查是否有改写必要——没有指代词直接跳过
    if not _needs_rewrite(query):
        return query

    history_text = _format_for_prompt(history)

    try:
        prompt = render(
            "query_rewrite",
            query=query,
            conversation_history=history_text,
        )
        result = _get_model().generate(prompt)
        rewritten = result.strip().strip('"').strip("'")
        if rewritten and len(rewritten) >= 2:
            return rewritten
    except Exception:
        pass

    return query


def _needs_rewrite(query: str) -> bool:
    """检查查询是否含指代词或上下文依赖"""
    indicators = ["它", "他", "她", "这个", "那个", "刚才", "上面", "前面", "之前", "再"]
    return any(w in query for w in indicators)


def _format_for_prompt(history: list[dict]) -> str:
    lines = []
    for msg in history[-6:]:
        role = "用户" if msg.get("role") == "human" else "AI"
        content = msg.get("content", "")
        lines.append(f"{role}: {content[:150]}")
    return "\n".join(lines)
