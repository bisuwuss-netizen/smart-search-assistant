import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """配置类"""
    # API Keys
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # 可选

    # 模型配置
    MODEL_NAME = "qwen-plus"
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 对话配置
    MAX_HISTORY_MESSAGES = 10  # 保留最近5轮（5问+5答）

    # 持久化配置
    CHECKPOINT_DIR = "./checkpoints"