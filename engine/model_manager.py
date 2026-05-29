"""
模型管理
统一的 LLM 调用封装，支持文本生成和结构化输出
"""
import time
from typing import Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from engine.config import get_config
from observability.logger import get_logger


class ModelManager:
    """
    LLM 调用管理器

    用法:
        manager = ModelManager()
        text = manager.generate("你好")
        result = manager.generate_structured(prompt, MyPydanticModel)
    """

    def __init__(self):
        config = get_config()
        self.model_name = config["model_name"]
        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=0.3,      # 低温度，保证决策稳定性
            max_tokens=1024,      # DECIDE 输出很短，1024 足够
        )

    def generate(self, prompt: str) -> str:
        """
        文本生成

        Args:
            prompt: 完整 Prompt 字符串

        Returns:
            模型输出的文本

        Raises:
            RuntimeError: 模型调用失败时抛出（含原始错误信息）
        """
        start = time.time()
        try:
            response = self._client.invoke(prompt)
            elapsed_ms = int((time.time() - start) * 1000)
   
            content = response.content.strip()
            logger = get_logger()
            logger.info(f"[ModelManager] generate: {elapsed_ms}ms, {len(content)} chars")
            #添加耗时日志
            return content
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            raise RuntimeError(
                f"模型调用失败（{elapsed_ms}ms）: {type(e).__name__}: {e}"
            ) from e
    #功能：接收一段完整的提示词（prompt），调用大模型，并返回模型生成的纯文本内容。

    def generate_structured(
        self, prompt: str, output_model: type[BaseModel]
    ) -> Any:
        """
        结构化输出生成

        使用 LangChain with_structured_output 约束模型输出，
        返回已校验的 Pydantic 对象，无需手动解析 JSON。

        Args:
            prompt: 完整 Prompt 字符串
            output_model: 目标 Pydantic 模型类

        Returns:
            已校验的 Pydantic 实例

        Raises:
            RuntimeError: 模型调用失败或输出不符合预期时抛出
        """
        start = time.time()
        try:
            structured_client = self._client.with_structured_output(output_model)
            #with_structured_output
            # 自动把你的 Pydantic 类转换成系统提示词(SystemMessage)的一部分，
            # 告诉大模型：“你必须严格按照这个 JSON Schema 来输出”
            result = structured_client.invoke(prompt)
            elapsed_ms = int((time.time() - start) * 1000)
            logger = get_logger()
            logger.info(f"[ModelManager] generate_structured: {elapsed_ms}ms, type={type(result).__name__}")
            return result

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            raise RuntimeError(
                f"结构化输出失败（{elapsed_ms}ms）: {type(e).__name__}: {e}"
            ) from e
#功能：强制要求大模型返回的数据必须符合你指定的格式（也就是代码中的 Pydantic 模型）
