"""
Embedding Provider
提供文本向量化能力，支持 Provider 抽象 + 千问实现
"""
from abc import ABC, abstractmethod
import requests

from engine.config import get_config


class EmbeddingProvider(ABC):
    """Embedding 服务抽象基类，换模型只需实现子类"""

    @abstractmethod
    def generate(self, texts: list[str]) -> list[list[float]]:
        """
        将文本列表转为向量列表

        Args:
            texts: 待向量化的文本列表

        Returns:
            与 texts 等长的向量列表，每个向量为 float 列表
        """
        ...


class QwenEmbeddingProvider(EmbeddingProvider):
    """千问 text-embedding-v3 实现"""

    MODEL_NAME = "text-embedding-v3"

    def __init__(self):
        config = get_config()
        self.api_key = config["api_key"]
        # base_url 如 https://dashscope.aliyuncs.com/compatible-mode/v1
        # embedding 端点拼接为 {base_url}/embeddings
        base = config["base_url"].rstrip("/")
        self.endpoint = f"{base}/embeddings"

    def generate(self, texts: list[str]) -> list[list[float]]:
        """
        调千问 embedding API（含简单重试）

        Raises:
            RuntimeError: API 调用失败
        """
        import time

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.MODEL_NAME,
            "input": texts,
        }

        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    self.endpoint, json=payload, headers=headers, timeout=30
                )
            except requests.exceptions.ConnectionError as e:
                last_error = e
                time.sleep(2 * (attempt + 1))
                continue

            if response.status_code == 200:
                data = response.json()
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]

            if response.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue

            last_error = RuntimeError(
                f"Embedding API 返回 {response.status_code}: {response.text[:300]}"
            )
            break

        raise last_error or RuntimeError("Embedding API 调用失败（已重试 3 次）")
