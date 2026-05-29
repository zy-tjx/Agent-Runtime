"""
向量检索器
封装 embedding + FAISS 搜索，提供统一的检索入口
"""
from rag.embedding import QwenEmbeddingProvider
from rag.vector_store import FAISSVectorStore
from observability.logger import get_logger


class VectorRetriever:
    """
    检索器封装类

    用法:
        retriever = VectorRetriever()
        retriever.ensure_index_ready()  # 确保索引存在（首次自动构建）
        results = retriever.retrieve("什么是 LangGraph", top_k=5)
        # → [{doc_id, content, source, score, heading, metadata}, ...]
    """

    def __init__(self):

        # 初始化向量存储库和嵌入提供者
        self.store = FAISSVectorStore()
        self.provider = QwenEmbeddingProvider()

    def ensure_index_ready(self) -> None:
        """确保索引已加载，不存在则自动构建"""
        try:
            self.store.load_index()
        except FileNotFoundError:
            self._build_and_save()

    def _build_and_save(self) -> None:
        """从知识库构建 FAISS 索引并持久化"""
        # 导入文档加载和文本分割模块
        from rag.document_loader import load_documents
        from rag.text_splitter import split_documents

        logger = get_logger()
        logger.info("[VectorRetriever] 首次构建索引...")
        # 从指定路径加载文档
        docs = load_documents("data/knowledge")
        chunks = split_documents(docs)
        # 使用嵌入模型构建索引并保存
        self.store.build_index(chunks, self.provider)
        self.store.save_index()
        logger.info(f"[VectorRetriever] 索引构建完成: {len(chunks)} 个 chunk")
        #_build_and_save（从零构建）：这个方法把文档加载（load_documents）、
        # 文本切块（split_documents）、向量化建索引（build_index）
        # 以及最后的持久化保存（save_index）一气呵成地全部搞定。

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        检索与查询最相关的 chunk

        Args:
            query: 用户查询文本
            top_k: 返回数量

        Returns:
            [{doc_id, content, source, score, heading, metadata}, ...]
            分别对应来源文档ID、chunk内容、来源文件名、相似度分数、所在标题、原文档元数据
        """
        query_vec = self.provider.generate([query])[0]
        return self.store.search(query_vec, top_k=top_k)


# ── 模块级单例（避免重复建索引） ──

_retriever: VectorRetriever | None = None

# 对外暴露的索引检索接口
def get_retriever() -> VectorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever()
        _retriever.ensure_index_ready()
    return _retriever
