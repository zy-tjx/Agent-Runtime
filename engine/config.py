"""
配置管理
从 .env 文件加载环境变量，提供统一的配置查询入口
"""
import os
from dotenv import load_dotenv

load_dotenv()
#作用是从 .env 文件加载环境变量，将这些环境变量装进内存
# 使得在代码中可以通过 os.getenv() 获取这些变量的值。


def get_config() -> dict[str, str]:
    """
    获取模型相关配置

    返回：{"api_key": ..., "base_url": ..., "model_name": ...}

    缺少必需变量时直接报错（不兜底），确保配置问题尽早暴露。
    """
    required = {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
        "model_name": "MODEL_NAME",
    }

    config = {}
    for key, env_var in required.items():
        value = os.getenv(env_var, "").strip()
        #os.getenv(env_var, "")：尝试获取内存中的环境变量
        if not value:
            raise ValueError(
                f"缺少环境变量 {env_var}，请检查 .env 文件。"
                f"可参考 .env.example 模板。"
            )
        config[key] = value

    return config
