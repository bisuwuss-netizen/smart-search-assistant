"""
高级 Agentic RAG Graph - 完整版

整合所有高级功能：
1. Multi-Query 查询扩展
2. Reflector 反思评估 + 循环机制
3. 混合搜索（本地 + 网络）
4. 来源追溯

流程图：
    用户输入 → decide → expand_query → search → reflector → [判断]
                                          ↑                      │
                                          │ insufficient         │ sufficient
                                          └──── refine ←─────────┘
                                                                 ↓
                                                              answer → 输出

这是项目的核心入口，展示了完整的 Agentic RAG 能力。
"""
import sqlite3
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import AgentState
from src.nodes import (
    decide_search, search_web, generate_answer, skip_search,
    local_rag_search, hybrid_search, reflect_on_results, refine_search,
    expand_query
)
from src.config import Config


def route_after_decide(state: AgentState) -> str:
    """决定搜索后的路由：
    1. 不需要搜索 -> skip_search
    2. 需要搜索且复杂 -> expand (Multi-Query)
    3. 需要搜索但简单 -> web/local/hybrid (直接搜索)
    """
    search_type = state.get("search_type", "none")
    use_multi_query = state.get("use_multi_query", False)

    if search_type == "none":
        return "skip_search"
    
    if use_multi_query:
        return "expand"
    
    return search_type


def route_search(state: AgentState) -> str:
    """路由到具体的搜索执行节点"""
    search_type = state.get("search_type", "web")
    routing = {
        "local": "local_rag",
        "web": "web_search",
        "hybrid": "hybrid_search"
    }
    return routing.get(search_type, "web_search")


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
        if loop_count < 2:
            return "refine"
        return "answer"


def create_advanced_graph():
    """
    创建高级 Agentic RAG Graph

    特点：
    1. Multi-Query 查询扩展提高召回率
    2. Reflector 反思机制保证答案质量
    3. 循环机制自动优化搜索
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("decide", decide_search)
    workflow.add_node("expand", expand_query)  # Multi-Query 扩展
    workflow.add_node("local_rag", local_rag_search)
    workflow.add_node("web_search", search_web)
    workflow.add_node("hybrid_search", hybrid_search)
    workflow.add_node("skip_search", skip_search)
    workflow.add_node("reflector", reflect_on_results)
    workflow.add_node("refine", refine_search)
    workflow.add_node("answer", generate_answer)

    # 设置入口
    workflow.set_entry_point("decide")

    # 添加条件边：从 decide 判断进入哪个分支
    workflow.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "expand": "expand",
            "skip_search": "skip_search",
            "web": "web_search",
            "local": "local_rag",
            "hybrid": "hybrid_search"
        }
    )

    # 从 expand 根据类型路由到具体的搜索节点
    workflow.add_conditional_edges(
        "expand",
        route_search,
        {
            "local_rag": "local_rag",
            "web_search": "web_search",
            "hybrid_search": "hybrid_search"
        }
    )

    # 所有搜索节点 → reflector
    workflow.add_edge("local_rag", "reflector")
    workflow.add_edge("web_search", "reflector")
    workflow.add_edge("hybrid_search", "reflector")
    workflow.add_edge("skip_search", "answer")  # 跳过搜索直接回答

    # reflector → 条件路由
    workflow.add_conditional_edges(
        "reflector",
        route_after_reflection,
        {
            "answer": "answer",
            "refine": "refine"
        }
    )

    # refine → reflector（循环）
    workflow.add_edge("refine", "reflector")

    # answer → END
    workflow.add_edge("answer", END)

    # 持久化
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    conn = sqlite3.connect(
        f"{Config.CHECKPOINT_DIR}/checkpoints_advanced.db",
        check_same_thread=False
    )
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


# 创建全局实例
graph_advanced = create_advanced_graph()


def create_initial_state(query: str, use_multi_query: bool = True, max_loops: int = 3) -> dict:
    """
    创建初始状态的辅助函数

    Args:
        query: 用户问题
        use_multi_query: 是否启用 Multi-Query 扩展
        max_loops: 最大循环次数
    """
    return {
        "current_query": query,
        "messages": [],
        "search_results": "",
        "final_answer": "",
        "current_step": "",
        "search_type": "",
        "local_contexts": "",
        "sources": [],
        "human_approved": False,
        "pending_action": "",
        # Reflector 相关
        "reflection_result": "",
        "reflection_reason": "",
        "loop_count": 0,
        "max_loops": max_loops,
        "refined_query": "",
        # Multi-Query 相关
        "expanded_queries": [],
        "use_multi_query": use_multi_query
    }


def ask(query: str, thread_id: str = "default", use_multi_query: bool = True) -> dict:
    """
    简化的问答接口

    Args:
        query: 用户问题
        thread_id: 会话 ID（用于多轮对话）
        use_multi_query: 是否启用 Multi-Query

    Returns:
        包含答案和元信息的字典
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = create_initial_state(query, use_multi_query=use_multi_query)

    result = graph_advanced.invoke(state, config)

    return {
        "answer": result.get("final_answer", ""),
        "sources": result.get("sources", []),
        "search_type": result.get("search_type", ""),
        "loop_count": result.get("loop_count", 0),
        "reflection_result": result.get("reflection_result", ""),
        "expanded_queries": result.get("expanded_queries", []),
        "local_contexts": result.get("local_contexts", ""),  # ← 添加
        "search_results": result.get("search_results", ""),  # ← 添加
    }


# ============ CLI 入口 ============
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Smart Search Assistant - Advanced Mode")
    print("=" * 60)
    print("""
功能特点：
- Multi-Query 查询扩展
- Reflector 反思评估
- 循环优化机制
- 本地知识库 + 网络搜索

输入 'quit' 退出
""")

    thread_id = "cli-session"

    while True:
        try:
            query = input("\n❓ 请输入问题: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break

            if not query:
                continue

            print("\n" + "-" * 40)
            result = ask(query, thread_id=thread_id)

            print(f"\n🎯 搜索类型: {result['search_type']}")
            print(f"🔄 循环次数: {result['loop_count']}")
            print(f"🤔 反思结果: {result['reflection_result']}")

            if result['expanded_queries']:
                print(f"📝 扩展查询: {len(result['expanded_queries'])} 个")

            print(f"\n💡 答案:\n{result['answer']}")

            if result['sources']:
                print(f"\n📚 来源 ({len(result['sources'])} 条):")
                for i, src in enumerate(result['sources'][:5], 1):
                    print(f"   [{i}] {src['type']}: {src['source'][:60]}...")

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
