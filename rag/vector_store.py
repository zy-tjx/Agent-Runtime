"""
向量存储与检索（FAISS）
支持建索引、搜索、持久化（save/load）
"""
import os
import json
import numpy as np
import faiss

from rag.embedding import EmbeddingProvider

# ── 持久化路径 ──
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")
#索引文件夹路径
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
#FAISS 库自带的二进制文件（高效存储海量的向量索引数据）
META_PATH = os.path.join(INDEX_DIR, "chunks_meta.json")
# chunk 元数据文件路径


class FAISSVectorStore:
    """
    FAISS 向量存储

    用法:
        store = FAISSVectorStore()
        store.build_index(chunks, provider)  # 首次建索引
        results = store.search(query_vector, top_k=5)
        相似度检索，提取前5个最相关的 chunk
        store.save_index()  # 持久化
        store.load_index()  # 从磁盘恢复
    """

    def __init__(self):
        self.index = None
        self.chunks_meta: list[dict] = []  # 存储 chunk 元数据

    # ── 索引构建 ──

    def build_index(
        self, chunks: list[dict], provider: EmbeddingProvider, batch_size: int = 10
    ):
        """
        向量化所有 chunk 并建立 FAISS 索引

        Args:
            chunks: text_splitter 输出的 chunk 列表
            provider: EmbeddingProvider 实例
            batch_size: 每批向量化的文本数（千问上限 10）
        """
        texts = [c["content"] for c in chunks]
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_vectors.extend(provider.generate(batch))
        embeddings = np.array(all_vectors, dtype=np.float32)

        # 归一化（余弦相似度 = 归一化后的内积）
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = 余弦相似度
        self.index.add(embeddings)
        self.chunks_meta = chunks

    # ── 持久化 ──

    def save_index(self, index_dir: str | None = None):
        """
        将 FAISS 索引和 chunk 元数据保存到磁盘

        Args:
            index_dir: 自定义路径，默认 data/vector_store/
        """
        if self.index is None:
            raise RuntimeError("无索引可保存")

        idx_path = os.path.join(index_dir, "faiss.index") if index_dir else INDEX_PATH
        meta_path = os.path.join(index_dir, "chunks_meta.json") if index_dir else META_PATH

        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        faiss.write_index(self.index, idx_path) # 将索引写入二进制文件

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_meta, f, ensure_ascii=False, indent=2) # 将 chunk 元数据保存到 JSON 文件

    def load_index(self, index_dir: str | None = None):
        """
        从磁盘加载 FAISS 索引和 chunk 元数据

        Args:
            index_dir: 自定义路径，默认 data/vector_store/
        """
        idx_path = os.path.join(index_dir, "faiss.index") if index_dir else INDEX_PATH
        meta_path = os.path.join(index_dir, "chunks_meta.json") if index_dir else META_PATH

        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"索引文件不存在: {idx_path}")

        self.index = faiss.read_index(idx_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks_meta = json.load(f)

    # ── 检索 ──

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """
        检索最相似的 top_k 个 chunk

        Args:
            query_vector: 查询向量（方法内部自动归一化）
            top_k: 返回数量

        Returns:
            [{doc_id, content, source, score, heading, metadata}, ...]
            分别对应来源文档ID、chunk内容、来源文件名、相似度分数、所在标题、原文档元数据
        """
        if self.index is None:
            raise RuntimeError("索引未构建，请先 build_index 或 load_index")

        vec = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(vec)

        scores, indices = self.index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_meta):
                continue
            meta = self.chunks_meta[idx]
            results.append(
                {
                    "doc_id": meta["doc_id"],
                    "content": meta["content"],
                    "source": meta["source"],
                    "score": float(score),
                    "heading": meta.get("heading", ""),
                    "metadata": meta.get("metadata", {}),
                }
            )
        return results
