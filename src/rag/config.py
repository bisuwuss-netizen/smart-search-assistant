# """配置文件"""

# # Embedding 模型
# EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# # Rerank 模型
# RERANK_MODEL = "BAAI/bge-reranker-base"

# # 大模型配置
# LLM_API_KEY = "your-api-key"
# LLM_BASE_URL = "base_url"
# LLM_MODEL = "deepseek-r1"

# # 切分配置
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 100

# # 检索配置
# VECTOR_SEARCH_TOP_K = 20  # 向量检索召回数量
# RERANK_TOP_N = 5          # Rerank 后保留数量
# VECTOR_WEIGHT = 0.6       # 混合检索中向量权重

"""RAG 模块配置"""
from src.config import Config

class RAGConfig:
    EMBEDDING_MODEL = Config.EMBEDDING_MODEL        # Embedding 模型
    RERANK_MODEL = Config.RERANK_MODEL          # Rerank 模型
    CHUNK_SIZE = Config.CHUNK_SIZE          # 文档切分大小
    CHUNK_OVERLAP = Config.CHUNK_OVERLAP         # 文档切分重叠部分

    LLM_API_KEY = Config.DASHSCOPE_API_KEY      # 大模型 API Key
    LLM_BASE_URL = Config.DEEPSEEK_BASE_URL          # 大模型基础 URL
    LLM_MODEL = Config.QWEN_MODEL_NAME           # 大模型名称

    VECTOR_SEARCH_TOP_K = Config.VECTOR_SEARCH_TOP_K  # 向量检索召回数量
    RERANK_TOP_N = Config.RERANK_TOP_N          # Rerank 后保留数量
    VECTOR_WEIGHT = Config.VECTOR_WEIGHT       # 混合检索中向量权重

    # 向量库持久化目录
    VECTOR_DB_DIR = Config.VECTOR_DB_DIR #持久化向量库路径