from src.graph import graph
from src.state import AgentState


def run_with_streaming(query: str, thread_id: str = "demo"):
    """运行图并流式显示进度"""

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = AgentState(
        messages=[],
        current_query=query,
        need_search=False,
        search_results="",
        final_answer="",
        current_step=""
    )

    print(f"\n{'=' * 60}")
    print(f"问题: {query}")
    print(f"{'=' * 60}\n")

    # 流式执行
    for event in graph.stream(initial_state, config, stream_mode="values"):
        step = event.get("current_step", "")
        if step:
            print(f"{step}")

    # 打印最终答案
    print(f"\n{'=' * 60}")
    print("答案:")
    print(event["final_answer"])
    print(f"{'=' * 60}\n")

    return event


if __name__ == "__main__":
    # 测试1: 需要搜索的问题
    run_with_streaming("2024年诺贝尔物理学奖得主是谁？")

    # 测试2: 不需要搜索的问题
    run_with_streaming("什么是 Python 的列表推导式？")