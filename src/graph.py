import sqlite3

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import AgentState
from src.nodes import (
    decide_search, search_web, generate_answer, skip_search, 
    local_rag_search, hybrid_search, expand_query, reflect_on_results, refine_search
)
from src.config import Config
import os


def should_search(state: AgentState) -> str:
    """路由函数：决定是否搜索"""
    return "search" if state["need_search"] else "skip_search"

def route_search(state: AgentState) -> str:
    """路由到不同的搜索节点"""
    search_type = state.get("search_type", "none")
    
    routing = {
        "local": "local_rag",
        "web": "web_search",
        "hybrid": "hybrid_search",
        "none": "skip_search"
    }
    return routing.get(search_type, "skip_search")

def route_after_reflection(state: AgentState) -> str:
    """反思后的路由决策"""
    reflection_result = state.get("reflection_result", "sufficient")
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)

    if loop_count >= max_loops:
        print(f"  ⚠️ 达到最大循环次数 ({max_loops})，强制生成答案")
        return "answer"

    if reflection_result == "sufficient":
        return "answer"
    elif reflection_result == "insufficient":
        return "refine"
    else:
        # IRRELEVANT 或其他暂且按不足处理一次
        if loop_count < 2:
            return "refine"
        return "answer"


def create_graph():
    """创建搜索助手 Graph"""

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("decide", decide_search)
    workflow.add_node("expand", expand_query)  # Multi-Query 扩展
    workflow.add_node("local_rag", local_rag_search)
    workflow.add_node("hybrid_search", hybrid_search)
    workflow.add_node("web_search", search_web)
    workflow.add_node("skip_search", skip_search)
    workflow.add_node("reflector", reflect_on_results) # 反思评估
    workflow.add_node("refine", refine_search)         # 改进搜索
    workflow.add_node("answer", generate_answer)

    # 设置入口
    workflow.set_entry_point("decide")

    # decide -> expand (总是先尝试扩展查询)
    workflow.add_edge("decide", "expand")

    # 添加条件边：从 expand 根据类型路由
    workflow.add_conditional_edges(
        "expand",
        route_search,
        {
            "local_rag": "local_rag",
            "hybrid_search": "hybrid_search",
            "web_search": "web_search",
            "skip_search": "skip_search"
        }
    )

    # 搜索节点指向 reflector
    workflow.add_edge("local_rag", "reflector")
    workflow.add_edge("hybrid_search", "reflector")
    workflow.add_edge("web_search", "reflector")
    
    # reflector -> 条件路由
    workflow.add_conditional_edges(
        "reflector",
        route_after_reflection,
        {
            "answer": "answer",
            "refine": "refine"
        }
    )

    # refine -> reflector (形成循环)
    workflow.add_edge("refine", "reflector")

    # skip_search 直接到 answer
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

    return workflow.compile(
        checkpointer=memory,
        # 在执行实际搜索前暂停，等待人工审批
        interrupt_before=["local_rag", "web_search", "hybrid_search"]
    )


# 创建全局图实例
graph = create_graph()