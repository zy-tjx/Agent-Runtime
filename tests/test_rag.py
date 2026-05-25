"""
RAG 检索链路测试
覆盖：document_loader / text_splitter / embedding / vector_store / vector_retriever
"""
import os
import pytest
import numpy as np

from rag.document_loader import load_documents
from rag.text_splitter import split_documents
from rag.embedding import QwenEmbeddingProvider
from rag.vector_store import FAISSVectorStore
from rag.vector_retriever import get_retriever


# ============================================================
# 1. Document Loader 测试
# ============================================================

class TestDocumentLoader:
    def test_loads_all_md_files(self):
        docs = load_documents("data/knowledge")
        assert len(docs) >= 3
        for d in docs:
            assert d["filename"].endswith(".md")
            assert len(d["content"]) > 100

    def test_parses_metadata(self):
        docs = load_documents("data/knowledge")
        langgraph = next(d for d in docs if d["filename"] == "langgraph_overview.md")
        m = langgraph["metadata"]
        assert "LangGraph" in m.get("主题", "")
        assert "中级" in m.get("难度", "")
        assert len(m.get("关键词", [])) >= 3


# ============================================================
# 2. Text Splitter 测试
# ============================================================

class TestTextSplitter:
    @pytest.fixture
    def chunks(self):
        docs = load_documents("data/knowledge")
        return split_documents(docs, chunk_size=500, overlap=50)

    def test_produces_chunks(self, chunks):
        assert len(chunks) >= 20

    def test_no_heading_cut(self, chunks):
        for c in chunks:
            assert not c["content"].startswith("##")

    def test_chunks_have_heading(self, chunks):
        for c in chunks:
            assert "heading" in c
            assert "source" in c
            assert "doc_id" in c

    def test_chunk_sizes(self, chunks):
        for c in chunks:
            assert len(c["content"]) <= 550  # chunk_size + overlap 余量


# ============================================================
# 3. Embedding 测试
# ============================================================

@pytest.mark.integration
class TestEmbedding:
    def test_generate_returns_vectors(self):
        provider = QwenEmbeddingProvider()
        vecs = provider.generate(["测试文本", "另一段文本"])
        assert len(vecs) == 2
        assert all(isinstance(v, list) for v in vecs)
        assert len(vecs[0]) > 0

    def test_vector_dimensions_consistent(self):
        provider = QwenEmbeddingProvider()
        vecs = provider.generate(["A", "B", "C"])
        dim = len(vecs[0])
        for v in vecs:
            assert len(v) == dim


# ============================================================
# 4. VectorStore 测试
# ============================================================

@pytest.mark.integration
class TestVectorStore:
    @pytest.fixture
    def store_with_index(self):
        docs = load_documents("data/knowledge")
        chunks = split_documents(docs)[:10]  # 只取前 10 个加速
        provider = QwenEmbeddingProvider()
        store = FAISSVectorStore()
        store.build_index(chunks, provider)
        return store

    def test_build_index(self, store_with_index):
        assert store_with_index.index is not None
        assert store_with_index.index.ntotal == 10

    def test_search(self, store_with_index):
        provider = QwenEmbeddingProvider()
        query_vec = provider.generate(["LangGraph 状态机"])[0]
        results = store_with_index.search(query_vec, top_k=3)
        assert len(results) >= 1
        assert results[0]["score"] > 0

    def test_save_and_load(self, store_with_index):
        store_with_index.save_index()
        # 加载到新实例
        store2 = FAISSVectorStore()
        store2.load_index()
        assert store2.index.ntotal == 10

        provider = QwenEmbeddingProvider()
        query_vec = provider.generate(["测试"])[0]
        r1 = store_with_index.search(query_vec, top_k=1)
        r2 = store2.search(query_vec, top_k=1)
        assert r1[0]["doc_id"] == r2[0]["doc_id"]

    def test_search_result_structure(self, store_with_index):
        provider = QwenEmbeddingProvider()
        query_vec = provider.generate(["Agent 状态机"])[0]
        results = store_with_index.search(query_vec, top_k=1)
        r = results[0]
        for field in ["doc_id", "content", "source", "score", "heading", "metadata"]:
            assert field in r, f"缺少字段: {field}"


# ============================================================
# 5. VectorRetriever 集成测试
# ============================================================

@pytest.mark.integration
class TestVectorRetriever:
    def test_retrieve_returns_results(self):
        retriever = get_retriever()
        results = retriever.retrieve("LangGraph 状态机是什么", top_k=3)
        assert len(results) >= 1
        assert results[0]["score"] > 0
        # 验证结果中有 LangGraph 相关文档（可能在 top-K 中的任意位置）
        sources = [r["source"] for r in results]
        assert any("langgraph" in s for s in sources)

    def test_retrieve_empty_query(self):
        retriever = get_retriever()
        results = retriever.retrieve("")
        assert isinstance(results, list)

    def test_retriever_singleton(self):
        r1 = get_retriever()
        r2 = get_retriever()
        assert r1 is r2


# ============================================================
# 6. QA 真实检索集成测试
# ============================================================

@pytest.mark.integration
class TestQARetrievalIntegration:
    def test_real_retrieval_in_qa_chain(self):
        """QA 链路基于真实向量检索"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.state_graph import run_graph

        result = run_graph("什么是 RAG")
        docs = result.get("retrieved_context", [])
        assert len(docs) >= 1
        assert result.get("retrieval_score") is not None
        assert result.get("answer_source") is not None

    def test_retrieval_score_positive(self):
        """真实检索分数应 > 0"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.state_graph import run_graph

        result = run_graph("Agent 状态机")
        assert result.get("retrieval_score", 0) > 0
