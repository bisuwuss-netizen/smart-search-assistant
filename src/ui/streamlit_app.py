"""
Streamlit Web UI

启动方式：
    cd smart-search-assistant
    streamlit run src/ui/streamlit_app.py

功能：
    - 对话式问答界面
    - 文档上传和管理
    - 实时显示搜索状态
    - 来源追溯展示
"""
import sys
import os

# 添加项目根目录到 Python 路径（解决 src.xxx 导入问题）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import time
from typing import Optional

# 设置页面配置（必须在其他 st 调用之前）
st.set_page_config(
    page_title="Smart Search Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入项目模块
from src.graph_advanced import graph_advanced, create_initial_state
from src.rag.rag_manager import RAGManager


def init_session_state():
    """初始化 session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{int(time.time())}"
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = RAGManager.get_instance()


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("⚙️ 设置")

    # 功能开关
    st.sidebar.subheader("功能选项")
    use_multi_query = st.sidebar.checkbox("Multi-Query 扩展", value=True, help="将问题扩展为多个查询")
    max_loops = st.sidebar.slider("最大循环次数", 1, 5, 3, help="反思循环的最大次数")

    st.sidebar.divider()

    # 知识库管理
    st.sidebar.subheader("📚 知识库管理")

    # 文档上传
    uploaded_file = st.sidebar.file_uploader(
        "上传文档",
        type=["pdf", "txt", "md"],
        help="支持 PDF、TXT、Markdown 格式"
    )

    if uploaded_file:
        if st.sidebar.button("📥 导入文档"):
            with st.spinner("正在导入文档..."):
                # 保存临时文件
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    rag = st.session_state.rag_manager
                    chunks = rag.add_document(tmp_path)
                    st.sidebar.success(f"✅ 已导入 {chunks} 个文档块")
                except Exception as e:
                    st.sidebar.error(f"❌ 导入失败: {e}")
                finally:
                    os.unlink(tmp_path)

    # 显示已导入文档
    rag = st.session_state.rag_manager
    doc_count = rag.count()
    documents = rag.list_documents()

    st.sidebar.metric("文档块数量", doc_count)

    if documents:
        with st.sidebar.expander(f"📄 已导入 {len(documents)} 个文档"):
            for doc in documents:
                st.write(f"• {doc}")

    # 清空按钮
    if st.sidebar.button("🗑️ 清空知识库", type="secondary"):
        rag.clear()
        st.sidebar.success("知识库已清空")
        st.rerun()

    st.sidebar.divider()

    # 会话管理
    st.sidebar.subheader("💬 会话管理")
    if st.sidebar.button("🔄 新建对话"):
        st.session_state.messages = []
        st.session_state.thread_id = f"streamlit-{int(time.time())}"
        st.rerun()

    return use_multi_query, max_loops


def render_chat_history():
    """渲染聊天历史"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 显示来源（如果是助手消息）
            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    with st.expander(f"📚 来源 ({len(sources)} 条)"):
                        for i, src in enumerate(sources, 1):
                            st.write(f"{i}. [{src['type']}] {src['source']}")


def process_query(query: str, use_multi_query: bool, max_loops: int) -> dict:
    """处理用户查询"""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = create_initial_state(
        query=query,
        use_multi_query=use_multi_query,
        max_loops=max_loops
    )

    result = graph_advanced.invoke(state, config)

    return {
        "answer": result.get("final_answer", ""),
        "sources": result.get("sources", []),
        "search_type": result.get("search_type", ""),
        "loop_count": result.get("loop_count", 0),
        "reflection_result": result.get("reflection_result", ""),
        "expanded_queries": result.get("expanded_queries", [])
    }


def main():
    """主函数"""
    init_session_state()

    # 标题
    st.title("🔍 Smart Search Assistant")
    st.caption("基于 LangGraph 的智能搜索助手 | Multi-Query | Reflector | RAG")

    # 侧边栏
    use_multi_query, max_loops = render_sidebar()

    # 聊天历史
    render_chat_history()

    # 输入框
    if query := st.chat_input("输入你的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 处理查询
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 显示处理状态
                status_placeholder = st.empty()

                # 执行查询
                result = process_query(query, use_multi_query, max_loops)

                # 显示元信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("搜索类型", result["search_type"])
                with col2:
                    st.metric("循环次数", result["loop_count"])
                with col3:
                    st.metric("反思结果", result["reflection_result"])

                # 显示扩展查询
                if result["expanded_queries"]:
                    with st.expander("🔄 扩展查询"):
                        for i, q in enumerate(result["expanded_queries"], 1):
                            st.write(f"{i}. {q}")

                # 显示答案
                st.markdown(result["answer"])

                # 显示来源
                if result["sources"]:
                    with st.expander(f"📚 信息来源 ({len(result['sources'])} 条)"):
                        for i, src in enumerate(result["sources"], 1):
                            source_text = src.get("source", "N/A")
                            score = src.get("score", 0)
                            st.write(f"{i}. [{src['type']}] {source_text[:80]}... (相关度: {score:.2f})")

        # 添加助手消息到历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })


if __name__ == "__main__":
    main()
