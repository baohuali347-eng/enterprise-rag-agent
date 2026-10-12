"""DeepSeek LLM 客户端封装。

DeepSeek 提供 OpenAI 兼容接口（https://api.deepseek.com），
直接用 langchain-openai 的 ChatOpenAI 对接，base_url 指过去即可。
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from kb_agent import config


def check_api_key() -> None:
    if not config.LLM_API_KEY:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY。请到 https://platform.deepseek.com 创建 API Key，"
            "然后写入项目根目录 .env:  DEEPSEEK_API_KEY=sk-xxx"
        )


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """获取 LLM 实例。检索/生成类任务用低温度保证稳定性。"""
    check_api_key()
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )


def get_structured_llm(model_cls: type, temperature: float = 0.0):
    """获取带结构化输出的 LLM。

    DeepSeek 不支持 OpenAI 的 response_format 模式（报 "unavailable"），
    但支持 function calling，所以显式用 method="function_calling"。
    """
    return get_llm(temperature).with_structured_output(model_cls, method="function_calling")
