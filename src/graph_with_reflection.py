"""
带反思循环的 Agentic RAG Graph

核心特点：
1. Reflector 节点：LLM 评估检索结果是否足够
2. 循环机制：如果不足，自动改进查询并重新搜索
3. 最大循环限制：防止无限循环（默认 3 次）

流程图：
    用户输入 → decide → search → reflector → [判断]
                          ↑                      │
                          │ insufficient         │ sufficient
                          └──── refine ←─────────┘
                                                 ↓
                                              answer → 输出

这是 Agentic RAG 的核心进阶点，体现了 Agent 的"自主决策"能力。
"""
import sqlite3
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import AgentState
from src.nodes import (
    decide_search, search_web, generate_answer, skip_search,
    local_rag_search, hybrid_search, reflect_on_results, refine_search
)
from src.config import Config


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
    """
    反思后的路由决策

    返回值：
    - "answer": 结果充分，生成答案
    - "refine": 结果不足，需要改进搜索
    - "answer": 达到最大循环次数，强制生成答案
    """
    reflection_result = state.get("reflection_result", "sufficient")
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)

    # 达到最大循环次数，强制结束
    if loop_count >= max_loops:
        print(f"  ⚠️ 达到最大循环次数 ({max_loops})，强制生成答案")
        return "answer"

    # 根据反思结果决定
    if reflection_result == "sufficient":
        return "answer"
    elif reflection_result == "insufficient":
        return "refine"
    else:  # irrelevant
        # 不相关的结果，尝试改进一次
        if loop_count < 2:
            return "refine"
        return "answer"


def create_graph_with_reflection():
    """
    创建带反思循环的 Graph

    这是项目的核心亮点，体现了：
    1. Agentic RAG 的自主决策能力
    2. LangGraph 的循环（Loop）机制
    3. 质量保证的自动化
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("decide", decide_search)
    workflow.add_node("local_rag", local_rag_search)
    workflow.add_node("web_search", search_web)
    workflow.add_node("hybrid_search", hybrid_search)
    workflow.add_node("skip_search", skip_search)
    workflow.add_node("reflector", reflect_on_results)  # 反思节点
    workflow.add_node("refine", refine_search)          # 改进搜索节点
    workflow.add_node("answer", generate_answer)

    # 设置入口
    workflow.set_entry_point("decide")

    # decide → 根据类型路由到搜索节点
    workflow.add_conditional_edges(
        "decide",
        route_search,
        {
            "local_rag": "local_rag",
            "web_search": "web_search",
            "hybrid_search": "hybrid_search",
            "skip_search": "skip_search"
        }
    )

    # 所有搜索节点 → reflector（反思评估）
    workflow.add_edge("local_rag", "reflector")
    workflow.add_edge("web_search", "reflector")
    workflow.add_edge("hybrid_search", "reflector")

    # skip_search 不需要反思，直接生成答案
    workflow.add_edge("skip_search", "answer")

    # reflector → 条件路由（循环的关键）
    workflow.add_conditional_edges(
        "reflector",
        route_after_reflection,
        {
            "answer": "answer",
            "refine": "refine"
        }
    )

    # refine → reflector（形成循环）
    workflow.add_edge("refine", "reflector")

    # answer → END
    workflow.add_edge("answer", END)

    # 持久化
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    conn = sqlite3.connect(
        f"{Config.CHECKPOINT_DIR}/checkpoints_reflection.db",
        check_same_thread=False
    )
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


# 创建全局实例
graph_with_reflection = create_graph_with_reflection()


# ============ 使用示例 ============
if __name__ == "__main__":
    print("=" * 60)
    print("🔄 Agentic RAG with Reflection Loop 演示")
    print("=" * 60)
    print("""
这个演示展示了 Reflector + Loop 机制：
- Agent 会评估检索结果是否足够
- 如果不足，会自动改进查询并重新搜索
- 最多循环 3 次，确保不会无限循环
""")

    config = {"configurable": {"thread_id": "reflection-demo"}}

    # 测试问题
    questions = [
        "LangGraph 的 checkpointer 有哪些实现？",
        "2024年诺贝尔物理学奖的具体贡献是什么？",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"❓ 问题: {question}")
        print("=" * 60)

        # 初始状态
        state = {
            "current_query": question,
            "messages": [],
            "need_search": False,
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
            "max_loops": 3,
            "refined_query": ""
        }

        result = graph_with_reflection.invoke(state, config)

        print(f"\n🎉 最终结果:")
        print(f"   循环次数: {result.get('loop_count', 0)}")
        print(f"   反思结果: {result.get('reflection_result', 'N/A')}")
        print(f"   答案预览: {result.get('final_answer', 'N/A')[:300]}...")

        # 显示来源
        sources = result.get('sources', [])
        if sources:
            print(f"\n   📚 信息来源 ({len(sources)} 条):")
            for i, src in enumerate(sources[:5], 1):
                print(f"      [{i}] {src.get('type', '?')}: {src.get('source', 'N/A')[:50]}")

        # 更换 thread_id
        config = {"configurable": {"thread_id": f"reflection-{hash(question)}"}}
