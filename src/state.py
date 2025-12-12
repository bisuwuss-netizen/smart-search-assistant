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