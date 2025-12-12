import pytest
from src.graph import graph
from src.state import AgentState


def test_search_decision():
    """测试搜索判断"""
    test_cases = [
        ("2024年奥运会在哪举办？", True),  # 应该搜索
        ("Python 如何定义函数？", False),  # 不应该搜索
    ]

    for query, expected_search in test_cases:
        state = AgentState(
            messages=[],
            current_query=query,
            need_search=False,
            search_results="",
            final_answer="",
            current_step=""
        )

        config = {"configurable": {"thread_id": "test"}}
        result = graph.invoke(state, config)

        assert result["need_search"] == expected_search, \
            f"查询 '{query}' 的搜索判断错误"


def test_answer_generation():
    """测试答案生成"""
    state = AgentState(
        messages=[],
        current_query="什么是 Python？",
        need_search=False,
        search_results="",
        final_answer="",
        current_step=""
    )

    config = {"configurable": {"thread_id": "test2"}}
    result = graph.invoke(state, config)

    assert len(result["final_answer"]) > 0, "答案不能为空"
    assert "Python" in result["final_answer"], "答案应包含关键词"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])