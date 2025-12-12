from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import AgentState
from src.config import Config
from src.tools import create_search_tool

# 初始化 LLM
llm = ChatOpenAI(
    model=Config.MODEL_NAME,
    openai_api_key=Config.DASHSCOPE_API_KEY,
    openai_api_base=Config.BASE_URL,
    temperature=0.7
)

# 搜索工具
search_tool = create_search_tool()


def decide_search(state: AgentState) -> AgentState:
    """节点1: 判断是否需要搜索"""
    state["current_step"] = "🤔 正在分析问题..."

    query = state["current_query"]

    # 使用 LLM 判断
    prompt = f"""判断以下问题是否需要网络搜索来回答：

问题：{query}

规则：
- 需要最新信息、新闻、数据 → 回答"YES"
- 常识性问题、数学计算、编程问题 → 回答"NO"

只回答 YES 或 NO。"""

    response = llm.invoke([HumanMessage(content=prompt)])

    need_search = "YES" in response.content.upper() #判断是否需要网络搜索
    state["need_search"] = need_search #更新状态

    return state


def search_web(state: AgentState) -> AgentState:
    """节点2: 执行搜索"""
    state["current_step"] = "🔍 正在搜索网络..."

    query = state["current_query"]
    # 这里需要重写query，防止后续问到“它”等代词，不知道指代的是什么
    messages = state["messages"]
    if messages:  # 有历史对话
        # 让 LLM 基于历史重写查询
        rewrite_prompt = f"""基于以下对话历史，将用户的新问题改写为一个独立的、完整的搜索查询。
        对话历史：
        {chr(10).join([f"{msg.type}: {msg.content[:100]}" for msg in messages[-4:]])}
        用户新问题：{query}
        要求：
        1. 如果问题包含"它"、"这个"等代词，替换为具体事物
        2. 如果问题是追问，补充必要的上下文
        3. 只输出改写后的搜索查询，不要解释
        改写后的查询："""

        #调用 llm 重写提示词
        rewritten = llm.invoke([HumanMessage(content=rewrite_prompt)])
        search_query = rewritten.content.strip()
        print(f"  原始查询: {query}")
        print(f"  改写查询: {search_query}")
    else:
        search_query = query
    results = search_tool.invoke(search_query)

    # 格式化结果
    if isinstance(results, list):
        formatted = "\n\n".join([
            f"来源 {i + 1}: {r.get('url', 'N/A')}\n搜索内容：{r.get('content', '')}"
            for i, r in enumerate(results)
        ])
    else:
        formatted = str(results)

    state["search_results"] = formatted
    return state


def generate_answer(state: AgentState) -> AgentState:
    """节点3: 生成最终答案"""
    state["current_step"] = "✍️ 正在生成答案..."

    query = state["current_query"]
    search_results = state.get("search_results", "")

    if state["need_search"]:
        prompt = f"""基于以下搜索结果回答问题：

问题：{query}

搜索结果：
{search_results}

请提供清晰、结构化的答案。如果搜索结果不够充分，请说明。"""
    else:
        prompt = f"""回答以下问题：

问题：{query}

请基于你的知识提供准确答案。"""

    # 包含历史对话
    messages = state["messages"] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    state["final_answer"] = response.content
    state["current_step"] = "✅ 完成"

    # 更新对话历史
    state["messages"].append(HumanMessage(content=query))
    state["messages"].append(AIMessage(content=response.content))

    return state


def skip_search(state: AgentState) -> AgentState:
    """跳过搜索的节点（直接生成答案）"""
    state["current_step"] = "💭 无需搜索，直接回答..."
    return state