# src/utils/llm_factory.py
import os
from typing import Callable, AnyStr

from langchain_openai import ChatOpenAI
from src.config import Config




class LLMFactory:
    """大模型工厂类"""


    @staticmethod
    def get_deepseek_model(temperature: float = 0.7):
        return ChatOpenAI(
            model=Config.DEEPSEEK_MODEL_NAME,
            openai_api_key=Config.DASHSCOPE_API_KEY,
            openai_api_base=Config.DEEPSEEK_BASE_URL,
            temperature=temperature
        )



    @staticmethod
    def get_minimax_model(temperature: float = 0.7):
        return ChatOpenAI(
            model=Config.MINIMAX_MODEL_NAME,
            openai_api_key=Config.MINIMAX_API_KEY,
            openai_api_base=Config.MINIMAX_BASE_URL,
            temperature=temperature
        )


    @staticmethod
    def get_qwen_model(temperature: float = 0.7):
        return ChatOpenAI(
            model=Config.QWEN_MODEL_NAME,
            openai_api_key=Config.DASHSCOPE_API_KEY,
            openai_api_base=Config.QWEN_BASE_URL,
            temperature=temperature
        )


    @staticmethod
    def get_model(temperature: float = 0.7):
        """正式代码（如 nodes.py）使用这个，解耦配置"""
        provider = getattr(Config, "LLM_PROVIDER", "deepseek").lower()

        if provider == "minimax":
            return LLMFactory.get_minimax_model(temperature)
        elif provider == "deepseek":
            return LLMFactory.get_deepseek_model(temperature)
        else:
            return LLMFactory.get_qwen_model(temperature)


# 为了方便，导出一个全局通用的获取函数
def get_llm(temperature: float = 0.7):
    return LLMFactory.get_model(temperature)
