"""RAG 功能演示

运行方式：
    python -m src.examples.rag_demo

测试内容：
    1. 文档导入
    2. 本地知识库检索
    3. 完整 Graph 流程
"""
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config


def test_rag_only():
    """仅测试 RAG 模块（不依赖 Graph）"""
    print("=" * 50)
    print("📚 RAG 模块独立测试")
    print("=" * 50)

    # 延迟导入，避免加载 Graph 时的依赖问题
    from src.rag.rag_manager import RAGManager

    # 1. 初始化 RAG 管理器
    print("\n[1/4] 初始化 RAG 管理器...")
    rag = RAGManager.get_instance()
    print("✅ RAG 管理器初始化成功")

    # 2. 检查知识库目录
    knowledge_dir = Config.KNOWLEDGE_DIR
    print(f"\n[2/4] 检查知识库目录: {knowledge_dir}")

    if not os.path.exists(knowledge_dir):
        print(f"❌ 目录不存在，正在创建...")
        os.makedirs(knowledge_dir, exist_ok=True)
        print(f"⚠️  请将文档放入 {knowledge_dir} 目录后重新运行")
        return

    files = [f for f in os.listdir(knowledge_dir) if f.endswith(('.pdf', '.txt', '.md'))]
    if not files:
        print(f"⚠️  目录为空，请将文档放入 {knowledge_dir}")
        return

    print(f"✅ 找到 {len(files)} 个文档: {files}")

    # 3. 导入文档
    print("\n[3/4] 导入文档到知识库...")
    count = rag.add_documents_from_dir(knowledge_dir)
    print(f"✅ 共导入 {count} 个文档块")

    # 4. 测试检索
    print("\n[4/4] 测试检索...")
    test_queries = [
        "什么是 RAG？",
        "LangGraph 的核心概念有哪些？",
        "如何实现多轮对话？"
    ]

    for query in test_queries:
        print(f"\n❓ 问题: {query}")
        result = rag.query(query, top_n=2)
        print(result["formatted"][:500])
        print("-" * 30)

    print("\n✅ RAG 模块测试完成！")


def test_full_graph():
    """测试完整 Graph 流程"""
    print("\n" + "=" * 50)
    print("🤖 完整 Graph 流程测试")
    print("=" * 50)

    from src.graph import graph

    config = {"configurable": {"thread_id": "rag-demo-test"}}

    questions = [
        "根据知识库，介绍一下什么是 RAG",
        "它有哪些核心流程？",  # 测试代词解析
        "LangGraph 和 LangChain Agent 有什么区别？"
    ]

    for q in questions:
        print(f"\n❓ 问题: {q}")
        state = {
            "current_query": q,
            "messages": [],
                "search_results": "",
            "final_answer": "",
            "current_step": "",
            "search_type": "",
            "local_contexts": "",
            "sources": []
        }

        try:
            result = graph.invoke(state, config)
            print(f"📍 搜索类型: {result.get('search_type', 'unknown')}")
            print(f"💬 回答:\n{result['final_answer'][:500]}...")
        except Exception as e:
            print(f"❌ 错误: {e}")

        print("-" * 50)

    print("\n✅ Graph 流程测试完成！")


def main():
    print("🚀 Smart Search Assistant - RAG 功能演示\n")

    # 先测试 RAG 模块
    try:
        test_rag_only()
    except Exception as e:
        print(f"❌ RAG 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 询问是否继续测试 Graph
    print("\n" + "=" * 50)
    user_input = input("是否继续测试完整 Graph 流程？(y/n): ").strip().lower()
    if user_input == 'y':
        try:
            test_full_graph()
        except Exception as e:
            print(f"❌ Graph 测试失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
