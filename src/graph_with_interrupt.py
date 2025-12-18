"""
带 Human-in-the-loop 的 Graph

核心概念：
- interrupt_before: 在指定节点执行前暂停
- 用户可以查看即将执行的操作，选择确认或修改
- 调用 graph.invoke() 会在 interrupt 点返回，需要再次调用才能继续

流程：
    用户输入 → decide → [INTERRUPT] → 搜索节点 → answer → 输出

使用方式：
    # 第一次调用，会在搜索前暂停
    result = graph.invoke(state, config)
    print(f"即将执行: {result['pending_action']}")

    # 用户确认后，继续执行
    result = graph.invoke(None, config)  # 传 None 表示继续
"""
import sqlite3
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import AgentState
from src.nodes import (
    decide_search, search_web, generate_answer, skip_search,
    local_rag_search, hybrid_search
)
from src.config import Config


def prepare_search(state: AgentState) -> AgentState:
    """
    准备搜索节点：设置待确认的操作描述

    这个节点在 interrupt 前执行，告诉用户即将做什么
    """
    search_type = state.get("search_type", "none")
    query = state["current_query"]

    # 根据搜索类型生成操作描述
    action_descriptions = {
        "local": f"📚 即将在本地知识库中搜索: '{query}'",
        "web": f"🌐 即将进行网络搜索: '{query}'",
        "hybrid": f"🔄 即将进行混合搜索（本地+网络）: '{query}'",
        "none": f"💭 无需搜索，将直接回答: '{query}'"
    }

    state["pending_action"] = action_descriptions.get(
        search_type,
        f"❓ 未知操作类型: {search_type}"
    )
    state["current_step"] = "⏸️ 等待用户确认..."

    return state


def route_after_confirm(state: AgentState) -> str:
    """
    确认后的路由函数

    根据 search_type 决定走哪个搜索节点
    """
    search_type = state.get("search_type", "none")

    routing = {
        "local": "local_rag",
        "web": "web_search",
        "hybrid": "hybrid_search",
        "none": "skip_search"
    }
    return routing.get(search_type, "skip_search")


def create_graph_with_interrupt():
    """
    创建带 Human-in-the-loop 的 Graph

    关键点：
    1. 添加 prepare_search 节点，设置待确认信息
    2. 在搜索节点前设置 interrupt_before
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("decide", decide_search)
    workflow.add_node("prepare", prepare_search)  # 新增：准备确认信息
    workflow.add_node("local_rag", local_rag_search)
    workflow.add_node("web_search", search_web)
    workflow.add_node("hybrid_search", hybrid_search)
    workflow.add_node("skip_search", skip_search)
    workflow.add_node("answer", generate_answer)

    # 设置入口
    workflow.set_entry_point("decide")

    # decide → prepare（先准备确认信息）
    workflow.add_edge("decide", "prepare")

    # prepare → 根据类型路由到不同搜索节点
    workflow.add_conditional_edges(
        "prepare",
        route_after_confirm,
        {
            "local_rag": "local_rag",
            "web_search": "web_search",
            "hybrid_search": "hybrid_search",
            "skip_search": "skip_search"
        }
    )

    # 所有搜索节点 → answer
    workflow.add_edge("local_rag", "answer")
    workflow.add_edge("web_search", "answer")
    workflow.add_edge("hybrid_search", "answer")
    workflow.add_edge("skip_search", "answer")
    workflow.add_edge("answer", END)

    # 持久化
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    conn = sqlite3.connect(
        f"{Config.CHECKPOINT_DIR}/checkpoints_interrupt.db",
        check_same_thread=False
    )
    memory = SqliteSaver(conn)

    # 关键：设置 interrupt_before
    # 在这些节点执行前会暂停，等待用户确认
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["local_rag", "web_search", "hybrid_search"]
        # 注意：skip_search 不需要确认
    )


# 创建全局实例
graph_with_interrupt = create_graph_with_interrupt()


# ============ 使用示例 ============
if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Human-in-the-loop 演示")
    print("=" * 60)

    config = {"configurable": {"thread_id": "interrupt-demo"}}

    # 初始状态
    state = {
        "current_query": "什么是 LangGraph？",
        "messages": [],
        "need_search": False,
        "search_results": "",
        "final_answer": "",
        "current_step": "",
        "search_type": "",
        "local_contexts": "",
        "sources": [],
        "human_approved": False,
        "pending_action": ""
    }

    print(f"\n❓ 用户问题: {state['current_query']}")
    print("-" * 40)

    # 第一次调用：会在搜索前暂停
    print("\n📤 第一次调用 graph.invoke()...")
    result = graph_with_interrupt.invoke(state, config)

    print(f"\n⏸️ Graph 暂停!")
    print(f"   搜索类型: {result.get('search_type', 'unknown')}")
    print(f"   待确认操作: {result.get('pending_action', 'N/A')}")

    # 模拟用户确认
    user_input = input("\n是否继续执行? (y/n): ").strip().lower()

    if user_input == 'y':
        print("\n📤 用户确认，继续执行...")
        # 第二次调用：传 None 表示继续执行
        result = graph_with_interrupt.invoke(None, config)
        print(f"\n✅ 执行完成!")
        print(f"   最终答案: {result.get('final_answer', 'N/A')[:200]}...")
    else:
        print("\n❌ 用户取消操作")
