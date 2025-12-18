"""
Human-in-the-loop 演示脚本

这个脚本展示了 LangGraph 的 interrupt 功能：
1. 用户提问后，Agent 判断需要什么类型的搜索
2. 在执行搜索前暂停，展示即将执行的操作
3. 用户确认后才真正执行搜索
4. 这种模式适合需要人工审批的敏感操作

运行方式：
    python -m src.examples.interrupt_demo
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.graph_with_interrupt import graph_with_interrupt


def demo_interrupt():
    """演示 interrupt 功能"""
    print("=" * 60)
    print("🔧 Human-in-the-loop (Interrupt) 演示")
    print("=" * 60)
    print("""
这个演示展示了 LangGraph 的 interrupt 功能：
- Agent 在执行搜索前会暂停
- 用户可以查看即将执行的操作
- 确认后才会真正执行

这种模式的应用场景：
- 敏感操作前的人工审批
- 外部 API 调用前的确认
- 成本控制（避免不必要的 API 调用）
""")

    config = {"configurable": {"thread_id": "interrupt-demo-1"}}

    questions = [
        "2024年诺贝尔物理学奖得主是谁？",  # 需要网络搜索
        "什么是 RAG？",  # 可能用本地知识库
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"❓ 用户问题: {question}")
        print("=" * 60)

        # 构造初始状态
        state = {
            "current_query": question,
            "messages": [],
                "search_results": "",
            "final_answer": "",
            "current_step": "",
            "search_type": "",
            "local_contexts": "",
            "sources": [],
            "human_approved": False,
            "pending_action": ""
        }

        # 第一次调用：判断搜索类型，然后暂停
        print("\n🔄 正在分析问题...")
        result = graph_with_interrupt.invoke(state, config)

        # 检查是否暂停
        print(f"\n⏸️  Agent 暂停，等待确认")
        print(f"   📋 搜索类型: {result.get('search_type', 'unknown')}")
        print(f"   📝 待执行操作: {result.get('pending_action', 'N/A')}")

        # 等待用户输入
        user_input = input("\n👉 是否执行此操作? (y=确认 / n=取消 / m=修改): ").strip().lower()

        if user_input == 'y':
            print("\n✅ 用户确认，继续执行...")
            # 传 None 表示继续当前状态
            final_result = graph_with_interrupt.invoke(None, config)

            print(f"\n🎉 执行完成!")
            print(f"   答案预览: {final_result.get('final_answer', 'N/A')[:300]}...")

            # 显示来源
            sources = final_result.get('sources', [])
            if sources:
                print(f"\n   📚 信息来源:")
                for i, src in enumerate(sources[:3], 1):
                    print(f"      [{i}] {src.get('type', '?')}: {src.get('source', 'N/A')[:50]}")

        elif user_input == 'm':
            # 修改搜索类型
            print("\n可选的搜索类型: local / web / hybrid / none")
            new_type = input("请输入新的搜索类型: ").strip().lower()
            if new_type in ['local', 'web', 'hybrid', 'none']:
                # 更新状态并继续
                result['search_type'] = new_type
                print(f"\n已修改为: {new_type}，继续执行...")
                final_result = graph_with_interrupt.invoke(None, config)
                print(f"\n🎉 执行完成!")
                print(f"   答案: {final_result.get('final_answer', 'N/A')[:300]}...")
            else:
                print("❌ 无效的类型，操作取消")

        else:
            print("\n❌ 用户取消操作")

        # 更换 thread_id，开始新对话
        config = {"configurable": {"thread_id": f"interrupt-demo-{hash(question)}"}}

    print("\n" + "=" * 60)
    print("✅ 演示结束")
    print("=" * 60)


if __name__ == "__main__":
    demo_interrupt()
