"""
知识检索工具（search_docs）
根据用户查询从知识库中检索相关文档，输出格式对齐 AgentState.retrieved_context
"""
from pydantic import BaseModel, Field
from tools.tool_executor import ToolOutput

MOCK_KNOWLEDGE_BASE = [
    {
        "doc_id": "doc_001",
        "content": "LangGraph 是一个基于有向图的状态机框架，用于构建可治理的 AI Agent。"
                   "它将 Agent 的决策流程建模为节点（Node）和边（Edge），"
                   "支持条件路由、中断恢复和流式输出。",
        "source": "langgraph_overview.md",
        "metadata": {"title": "LangGraph 概述", "section": "核心概念", "author": "LangChain Team"},
    },
    {
        "doc_id": "doc_002",
        "content": "RAG（Retrieval-Augmented Generation）检索增强生成将信息检索与文本生成结合。"
                   "核心流程包括：Query Rewrite → Vector Search → Rerank → Context Assemble，"
                   "能有效减少大模型幻觉。",
        "source": "rag_guide.md",
        "metadata": {"title": "RAG 检索增强指南", "section": "流程概览", "author": "AI Research"},
    },
    {
        "doc_id": "doc_003",
        "content": "Agent Engineering 关注如何设计和治理智能代理系统。"
                   "关键模块包括：State 管理、Tool 调度、Memory 生命周期、"
                   "RAG 检索、Reflection 反思和异常恢复。",
        "source": "agent_engineering.md",
        "metadata": {"title": "Agent Engineering 实践", "section": "架构设计", "author": "Agent Lab"},
    },
    {
        "doc_id": "doc_004",
        "content": "Prompt Engineering 是设计和优化提示词以引导大语言模型生成期望输出的技术。"
                   "包括 few-shot prompting、chain-of-thought、structured output 等方法。",
        "source": "prompt_guide.md",
        "metadata": {"title": "Prompt Engineering 指南", "section": "方法概览", "author": "Prompt Lab"},
    },
    {
        "doc_id": "doc_005",
        "content": "Memory 系统在 Agent 中负责信息的持久化和检索。分为三层："
                   "短期记忆（会话上下文）、长期记忆（用户画像/进度）、"
                   "知识记忆（事实/经验/反思）。",
        "source": "memory_design.md",
        "metadata": {"title": "Agent Memory 系统设计", "section": "架构", "author": "Memory Team"},
    },
]


class SearchDocsInput(BaseModel):
    """search_docs 输入参数"""

    query: str = Field(..., description="检索查询文本", min_length=1)
    top_k: int = Field(default=5, description="返回文档数量", ge=1, le=20)


def run(input_data: SearchDocsInput) -> ToolOutput:
    """执行知识检索（Mock：关键词匹配 + 返回 top_k 条）"""
    query_lower = input_data.query.lower()
    results = []

    for doc in MOCK_KNOWLEDGE_BASE:
        # 简单关键词匹配：query 中任意词出现在 doc 中即命中
        if any(word in doc["content"] for word in query_lower.split()):
            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "content": doc["content"],
                    "source": doc["source"],
                    "score": 0.90 + 0.02 * len(results),
                    "metadata": doc["metadata"],
                }
            )

    if not results:
        # 无匹配时返回最相近的一条
        results.append(
            {
                "doc_id": "doc_001",
                "content": MOCK_KNOWLEDGE_BASE[0]["content"],
                "source": MOCK_KNOWLEDGE_BASE[0]["source"],
                "score": 0.60,
                "metadata": MOCK_KNOWLEDGE_BASE[0]["metadata"],
            }
        )

    return ToolOutput(
        tool_name="search_docs",
        status="success",
        result={"documents": results[: input_data.top_k]},
        error=None,
        duration_ms=0,
    )
