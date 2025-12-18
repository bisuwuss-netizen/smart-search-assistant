from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """智能搜索助手的状态定义"""
    # 对话历史（最近5轮）
    messages: Annotated[List[BaseMessage], add_messages]

    # 当前用户问题
    current_query: str

    # 是否需要搜索
    need_search: bool

    # 搜索结果
    search_results: str

    # 最终答案
    final_answer: str

    # 当前步骤（用于 Streaming 显示）
    current_step: str


    # === 新增 RAG 相关字段 ===
    # 搜索类型：local / web / hybrid / none
    search_type: str

    # 本地知识库检索结果
    local_contexts: str

    # 知识来源追溯
    sources: List[dict]  # [{"type": "local/web", "source": "...", "score": 0.9}]

    # === Human-in-the-loop 相关 ===
    # 用户是否确认执行搜索
    human_approved: bool
    # 待确认的操作描述（展示给用户看）
    pending_action: str

    # === Reflector 反思机制相关 ===
    # 反思结果：sufficient / insufficient / irrelevant
    reflection_result: str
    # 反思原因（用于日志和调试）
    reflection_reason: str
    # 当前循环次数
    loop_count: int
    # 最大循环次数
    max_loops: int
    # 改进后的查询（用于重新搜索）
    refined_query: str

    # === Multi-Query 查询扩展相关 ===
    # 扩展后的多个查询
    expanded_queries: List[str]
    # 是否启用 Multi-Query
    use_multi_query: bool
