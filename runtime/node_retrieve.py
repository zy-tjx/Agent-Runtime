"""
RETRIEVE 节点
负责 RAG 检索增强全流程，产出 retrieved_context 及中间产物
含 Query Rewrite 层：将上下文依赖查询改写为完整独立查询
"""
"""
retrieve_node (入口)
  ├── rewrite (Query Rewrite：基于对话历史改写查询，解决指代不明)
  ├── get_retriever (获取向量检索器单例)
  └── retriever.retrieve (执行向量相似度检索，捞取 Top-K 相关文档)
"""
import time
from typing import Any
from rag.vector_retriever import get_retriever
from rag.query_rewrite import rewrite
from observability.logger import get_logger


def retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
    """执行向量检索（含 Query Rewrite）"""
    logger = get_logger()
    logger.node_start("RETRIEVE", state)

    try:
        decision = state.get("decision", {})
        arguments = decision.get("arguments", {})
        user_input = state.get("user_input", "")

        # ── 提取查询文本 ──
        raw_query = arguments.get("query") or user_input

        # ── Query Rewrite：基于对话历史改写查询 ──
        stm = state.get("short_term_memory", {}) or {}
        history = stm.get("buffer", [])
        rewrite_start = time.time()
        query = rewrite(raw_query, history)
        rewrite_ms = int((time.time() - rewrite_start) * 1000)

        # ── 向量检索 ──
        retriever = get_retriever()
        documents = retriever.retrieve(query, top_k=5)

        retrieval_score = None
        top_score = None
        if documents:
            scores = [d["score"] for d in documents if "score" in d]
            if scores:
                retrieval_score = sum(scores) / len(scores)
                top_score = scores[0]

        # ── 检索质量阈值：top-1 < 0.45 视为无相关文档，避免无效重试 ──
        RETRIEVAL_THRESHOLD = 0.45
        if top_score is not None and top_score < RETRIEVAL_THRESHOLD:
            documents = []
            retrieval_score = None

        result = {
            "current_step": "RETRIEVE",
            "rewritten_query": query,
            "retrieval_score": retrieval_score,
            "vector_search_results": documents,
            "reranked_results": documents,
            "retrieved_context": documents,
            "rag_metadata": {
                "query_rewrite_latency_ms": rewrite_ms,
                "total_docs_retrieved": len(documents),
            },
            "messages": [
                {
                    "role": "ai",
                    "content": (
                        f"检索完成：{len(documents)} 条文档"
                        + (f"，avg score={retrieval_score:.2f}" if retrieval_score else "")
                    ),
                }
            ],
        }
        logger.node_end("RETRIEVE", {**state, **result})
        return result
    except Exception as e:
        logger.node_error("RETRIEVE", str(e))
        raise
