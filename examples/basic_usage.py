from rich.pretty import pprint

from src.graph import graph
from src.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from src.config import Config


class ConversationManager:
    """对话管理器"""

    def __init__(self, thread_id: str = "default"):
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}

    def ask(self, query: str) -> str:
        """提问并获取答案"""

        # 获取当前状态（包含历史）
        current_state = graph.get_state(self.config)

        if current_state.values:
            messages = current_state.values.get("messages", [])
        else:
            messages = []

        # 限制历史长度
        if len(messages) > Config.MAX_HISTORY_MESSAGES:
            messages = messages[-Config.MAX_HISTORY_MESSAGES:]

        # 创建新状态
        initial_state = AgentState(
            messages=messages,
            current_query=query,
            need_search=False,
            search_results="",
            final_answer="",
            current_step=""
        )

        # 执行
        result = None
        for event in graph.stream(initial_state, self.config, stream_mode="values"):
            step = event.get("current_step", "")
            if step:
                print(f"  {step}")
            result = event

        return result["final_answer"]

    def get_history(self):
        """获取对话历史"""
        state = graph.get_state(self.config)
        return state.values.get("messages", []) if state.values else []


# 使用示例
if __name__ == "__main__":
    manager = ConversationManager(thread_id="user123")

    print("\n=== 多轮对话测试 ===\n")

    # 第一轮
    print("Q1: 介绍一下 Antropic 最近发布的 skills")
    answer1 = manager.ask("介绍一下 Antropic 最近发布的 skills")
    print(f"A1: {answer1[:100]}...\n")

    # 第二轮（测试上下文理解）
    print("Q2: 它的主要优势是什么？")
    answer2 = manager.ask("它的主要优势是什么？")
    print(f"A2: {answer2[:100]}...\n")

    # 查看历史
    history = manager.get_history()
    print(f"对话历史：")
    # pprint(history)
    print(f"对话历史: {len(history)} 条消息")