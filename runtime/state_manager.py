"""
State Schema 定义与状态管理辅助函数
AgentState 是整个系统的全局状态，所有节点通过它交换数据
"""
from typing import TypedDict, Optional, Annotated, Any
from langgraph.graph.message import add_messages


# ============================================================
# 核心 State Schema
# ============================================================

class AgentState(TypedDict, total=False):
    """
    单 Agent Runtime 的全局 State

    采用 total=False 使得所有字段可选，
    每个节点只返回自己关心的 Partial State Dict 即可。
    """

    # ── 全局流转字段 ──
    messages: Annotated[list[dict[str, Any]], add_messages]
    """对话历史，使用 add_messages reducer 追加而非覆盖"""

    current_step: str
    """当前所在节点名称（PLANNER/DECIDE/RETRIEVE/EXECUTE/REFLECT）"""

    user_input: str
    """用户本轮原始输入"""

    mode: str
    """运行模式（PLANNER 产出）：learn / qa。工具选择由 DECIDE 节点负责"""

    final_output: Optional[str]
    """最终回复文本，在 REFLECT 确认后写入"""

    # ── 错误与恢复（runtime/recovery_handler） ──
    error: Optional[str]
    """当前错误信息，无错误时为 None"""

    retry_count: int
    """全局反思进入reflect计数"""

    max_retries: int
    """全局最大重试上限，默认 3"""

    recovery_action: Optional[str]
    """恢复策略标记：retry / fallback / guardrail / escalate"""
    #重试 / 降级 / 护栏 / 升级
    #当前项目只有重试与降级这两种

    # ── PLANNER 节点产出（配合 tools/create_plan） ──
    plan: Optional[dict[str, Any]]
    """学习计划结构：
    {
        "topic": str,           # 学习主题
        "goals": list[str],     # 学习目标
        "modules": [            # 模块列表
            {"title": str, "content": str, "duration_minutes": int}
        ],
        "estimated_total_minutes": int,
        "created_at": str       # ISO 时间戳
    }
    """

    # ── DECIDE 节点产出（配合 runtime/edge_router） ──
    decision: Optional[dict[str, Any]]
    """策略决策结构：
    {
        "tool_name": str,       # 选定工具名称
        "arguments": dict,      # 传递给工具的参数字典
        "reason": str,          # 决策理由
        "confidence": float,    # 决策置信度 0.0~1.0
        "requires_retrieval": bool  # 是否需要先检索知识库
    }
    """

    # ── RETRIEVE 节点产出（配合 rag/） ──
    rewritten_query: Optional[str]
    """Query Rewrite 改写后的查询文本"""

    retrieval_score: Optional[float]
    """检索质量分（RETRIEVE 产出，检索结果的平均相似度分数）"""

    retrieved_context: list[dict[str, Any]]
    """检索到的文档片段：
    [
        {
            "doc_id": str,
            "content": str,
            "source": str,          # PDF/MD/HTML
            "score": float,         # 相似度分数
            "metadata": dict        # 文本的描述信息，如标题、作者、日期等
        }
    ]
    """

    # ── RAG 中间产物（配合 rag/，可观测性与错误归因用） ──以下字段当前项目没有用到
    vector_search_results: list[dict[str, Any]]
    """向量检索原始结果（重排序前）：
    [
        {
            "doc_id": str,
            "content": str,
            "score": float,
            "source": str
        }
    ]
    """

    reranked_results: list[dict[str, Any]]
    """重排序后的结果（向量检索→rerank 后）：
    [
        {
            "doc_id": str,
            "content": str,
            "original_score": float,    # 向量检索原始分数
            "rerank_score": float,      # 重排序后分数
            "rank_delta": int           # 排名变化（正=上升，负=下降）
        }
    ]
    """

    context_assembly_info: Optional[dict[str, Any]]
    """上下文组装信息（预留，Phase 2/3 填充）：
    {
        "total_chars": int,             # 拼入上下文的总字符数
        "truncation_strategy": str,     # 截断策略（head/tail/sliding_window）
        "truncated_count": int,         # 被截断的文档数
        "prompt_template_version": str  # 使用的 Prompt 模板版本
    }
    """

    # ── EXECUTE 节点产出（配合 tools/） ──
    tool_calls: list[dict[str, Any]]
    """工具调用记录：
    [
        {
            "tool_name": str,
            "arguments": dict,      # 调用工具传入的具体参数
            "called_at": str        # ISO 时间戳
        }
    ]
    """

    tool_results: list[dict[str, Any]]
    """工具执行结果：
    [
        {
            "tool_name": str,
            "status": str,          # success / failed / timeout
            "result": Any,
            "error": Optional[str],
            "duration_ms": int      #工具执行耗时
        }
    ]
    """

    # ── REFLECT 节点产出（配合 reflection/） ──
    reflection: Optional[dict[str, Any]]
    """反思结果结构：
    {
        "confidence": float,                # 自我评估置信度 0.0~1.0
        "is_satisfactory": bool,            # 结果是否满意
        "error_root_cause": Optional[str],          # 错误根因分析
        "improvement_suggestion": Optional[str],    # 改进建议
        "next_action": str,                 # end / retry
        "retry_target_node": Optional[str],         # retry 时指定：PLANNER / RETRIEVE / EXECUTE
        "hallucination_flag": bool          # 是否检测到幻觉
    }
    """

    # ── 治理字段（供 Eval / LangSmith / Memory 消费） ──
    groundedness_score: Optional[float]
    """答案基于检索文档的程度 0.0~1.0（REFLECT 评估）"""

    completeness_score: Optional[float]
    """答案对用户问题的覆盖度 0.0~1.0（REFLECT 评估）"""

    answer_source: Optional[str]
    """答案来源标记：rag / llm_fallback """

    retry_reason: Optional[str]
    """重试原因（REFLECT 写入，供 Memory 积累经验）"""

    fallback_triggered: bool
    """本次请求是否触发过 LLM 降级（任意节点降级即为 True）"""

    fallback_reason: Optional[str]
    """降级原因（触发降级的节点名 + 错误摘要，供 Eval 审计）"""

    # ── Memory 短期记忆（配合 memory/short_term_memory） ──
    short_term_memory: dict[str, Any]
    """当前会话的短期记忆缓冲区：
    {
        "session_id": str,
        "buffer": list[dict],       # 最近 N 轮对话历史 [{role, content}, ...]
    }
    """

    # ── Memory 长期记忆（配合 memory/long_term_memory） ──当前字段没用到
    long_term_memory: dict[str, Any]
    """从长期存储加载的用户级记忆：
    {
        "user_id": str,
        "profile": dict,            # 用户画像（偏好/水平/目标）
        "learning_progress": dict,  # 学习进度（已完成模块/掌握程度）
        "historical_topics": list,  # 历史学习主题
        "last_updated": str         # ISO 时间戳
    }
    """
    """长期记忆字段（全都没用，预留）。当前长期记忆直接走 SQLite，不通过 State 中转"""

    # ── RAG 链路元数据（可观测性用） ──
    rag_metadata: Optional[dict[str, Any]]
    """RAG 检索元数据：
    {
        "query_rewrite_latency_ms": int,    查询改写耗时
        "total_docs_retrieved": int,        检索到的文档总数
    }
    """


# ============================================================
# 状态管理辅助函数
# ============================================================

def create_initial_state(
    user_input: str,
    history: list[dict] | None = None,
) -> AgentState:
    """根据用户输入构建初始 State（Start 节点调用）

    Args:
        user_input: 当前轮用户输入
        history: 最近 N 轮对话历史 [{"role": "human/ai", "content": "..."}, ...]
    """
    messages = [{"role": "human", "content": user_input}]
    stm: dict = {"buffer": history or []}
    # 如果提供了历史，拼到 messages 最前面供 LangGraph 的 add_messages reducer 累积
    if history:
        messages = history + messages

    return AgentState(
        messages=messages,
        current_step="START",
        user_input=user_input,
        mode="learn",
        final_output=None,
        error=None,
        retry_count=0,
        max_retries=3,
        recovery_action=None,
        plan=None,
        decision=None,
        rewritten_query=None,
        retrieval_score=None,
        retrieved_context=[],
        vector_search_results=[],
        reranked_results=[],
        context_assembly_info=None,
        tool_calls=[],
        tool_results=[],
        reflection=None,
        groundedness_score=None,
        completeness_score=None,
        answer_source=None,
        retry_reason=None,
        fallback_triggered=False,
        fallback_reason=None,
        short_term_memory=stm,
        long_term_memory={},
        rag_metadata=None,
    )


