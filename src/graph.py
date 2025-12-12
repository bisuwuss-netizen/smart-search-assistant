import sqlite3

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import AgentState
from src.nodes import decide_search, search_web, generate_answer, skip_search
from src.config import Config
import os


def should_search(state: AgentState) -> str:
    """路由函数：决定是否搜索"""
    return "search" if state["need_search"] else "skip_search"


def create_graph():
    """创建搜索助手 Graph"""

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("decide", decide_search) #通过该节点，先更新一遍 state
    workflow.add_node("search", search_web)
    workflow.add_node("skip_search", skip_search)
    workflow.add_node("answer", generate_answer)

    # 设置入口
    workflow.set_entry_point("decide")

    # 添加条件边
    workflow.add_conditional_edges(
        "decide",  # ←从 decide 边开始
        should_search,
        {       # ← 路由映射表
            "search": "search",
            "skip_search": "skip_search"
        }
    )

    # 添加普通边
    workflow.add_edge("search", "answer")
    workflow.add_edge("skip_search", "answer")
    workflow.add_edge("answer", END)

    # 添加持久化
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    # memory = SqliteSaver.from_conn_string(
    #     f"{Config.CHECKPOINT_DIR}/checkpoints.db"
    # )
    conn = sqlite3.connect(
        f"{Config.CHECKPOINT_DIR}/checkpoints.db",
        check_same_thread=False #允许多线程访问
    )
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


# 创建全局图实例
graph = create_graph()