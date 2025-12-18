import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """配置类"""
    # API Keys
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # 可选

    MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")

    # 模型配置
    #deepseek-r1模型
    DEEPSEEK_MODEL_NAME = "deepseek-r1"
    DEEPSEEK_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # minimax模型
    MINIMAX_MODEL_NAME = "MiniMax-M2"
    MINIMAX_BASE_URL = "https://api.minimax.io/v1"

    #qwen-plus模型
    QWEN_MODEL_NAME = "qwen-plus"
    QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


    #默认提供的大模型
    LLM_PROVIDER = "qwen"

    # 对话配置
    MAX_HISTORY_MESSAGES = 10  # 保留最近5轮（5问+5答）

    # 持久化配置
    CHECKPOINT_DIR = "./checkpoints"

    
    # === RAG 配置 ===
    # Embedding 模型
    EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
    
    # Rerank 模型
    RERANK_MODEL = "BAAI/bge-reranker-base"
    
    # 知识库目录
    KNOWLEDGE_DIR = "./data/knowledge"
    
    # 向量数据库目录
    VECTOR_DB_DIR = "./data/vector_db"
    
    # 切分配置
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    
    # 检索配置
    VECTOR_SEARCH_TOP_K = 20
    RERANK_TOP_N = 5
    VECTOR_WEIGHT = 0.6
    USE_RERANK = True
