from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import AgentState
from src.config import Config
from src.tools import create_search_tool
from src.rag.rag_manager import RAGManager
from src.utils.llm_factory import LLMFactory

# 初始化 LLM
llm = LLMFactory.get_model()

# 搜索工具
search_tool = create_search_tool()

# 初始化 RAG 管理器（延迟初始化，避免启动时加载模型）
_rag_manager = None

def get_rag_manager():
    """延迟获取 RAG 管理器实例"""
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = RAGManager.get_instance()
    return _rag_manager


def decide_search(state: AgentState) -> AgentState:
    """判断搜索类型：local / web / hybrid / none"""
    query = state["current_query"]

    prompt = f"""判断以下问题应该使用哪种搜索方式：

问题：{query}

选项：
- LOCAL: 问题涉及已上传的文档/知识库内容
- WEB: 需要最新信息、新闻、实时数据
- HYBRID: 需要结合本地知识和网络信息
- NONE: 常识问题，无需搜索

只回答 LOCAL / WEB / HYBRID / NONE 之一。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    search_type = response.content.strip().upper()

    # 验证返回值
    if search_type not in ["LOCAL", "WEB", "HYBRID", "NONE"]:
        search_type = "WEB"  # 默认网络搜索

    state["search_type"] = search_type.lower()
    return state


def expand_query(state: AgentState) -> AgentState:
    """
    Multi-Query 查询扩展节点

    将单个问题扩展为多个相关查询，提高检索召回率。
    这是 RAG 优化的重要技术，可以：
    1. 捕捉问题的不同表述方式
    2. 覆盖相关的子问题
    3. 使用不同的关键词组合
    """
    state["current_step"] = "🔄 正在扩展查询..."

    # 如果禁用了 Multi-Query，直接返回原查询
    if not state.get("use_multi_query", True):
        state["expanded_queries"] = [state["current_query"]]
        return state

    query = state["current_query"]

    expand_prompt = f"""你是一个查询扩展专家。请将用户的问题扩展为 3-4 个相关但不同角度的搜索查询。

## 用户原始问题
{query}

## 扩展要求
1. 保留原始问题的核心意图
2. 使用不同的关键词和表述方式
3. 可以包含相关的子问题
4. 每个查询都应该是独立的、可搜索的

## 输出格式（每行一个查询，不要编号）
查询1
查询2
查询3
查询4

请扩展："""

    response = llm.invoke([HumanMessage(content=expand_prompt)])
    result_text = response.content.strip()

    # 解析扩展的查询
    expanded = []
    for line in result_text.split("\n"):
        line = line.strip()
        # 跳过空行和编号
        if line and not line.startswith("#") and len(line) > 5:
            # 去除可能的编号前缀 (1. 2. 等)
            if line[0].isdigit() and (line[1] == '.' or line[1] == '、'):
                line = line[2:].strip()
            expanded.append(line)

    # 确保至少有原始查询
    if not expanded:
        expanded = [query]
    else:
        # 限制最多 4 个查询
        expanded = expanded[:4]
        # 确保原始查询在列表中
        if query not in expanded:
            expanded.insert(0, query)

    state["expanded_queries"] = expanded

    print(f"  📝 原始查询: {query}")
    print(f"  🔄 扩展查询 ({len(expanded)} 个):")
    for i, q in enumerate(expanded, 1):
        print(f"     {i}. {q}")

    return state


def local_rag_search(state: AgentState) -> AgentState:
    """
    本地知识库检索

    支持 Multi-Query：如果有扩展查询，会对每个查询执行检索并合并结果
    """
    state["current_step"] = "📚 正在检索本地知识库..."

    # 获取查询列表（支持 Multi-Query）
    queries = state.get("expanded_queries", [state["current_query"]])
    if not queries:
        queries = [state["current_query"]]

    all_contexts = []
    all_sources = []
    seen_contents = set()  # 用于去重

    for query in queries:
        result = get_rag_manager().query(query, top_n=3)  # 每个查询取 top 3

        for ctx in result["contexts"]:
            content = ctx.get("content", "")
            # 简单去重：跳过重复内容
            content_hash = hash(content[:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_contexts.append(ctx)
                all_sources.append({
                    "type": "local",
                    "source": ctx.get("metadata", {}).get("source", ""),
                    "score": float(ctx.get("score", 0))
                })

    # 按分数排序，取 top 5
    all_contexts.sort(key=lambda x: x.get("score", 0), reverse=True)
    all_contexts = all_contexts[:5]
    all_sources = all_sources[:5]

    # 格式化结果
    state["local_contexts"] = _format_local_contexts(all_contexts)
    state["sources"] = all_sources

    print(f"  📚 检索到 {len(all_contexts)} 条本地结果")
    return state


def _format_local_contexts(contexts: list) -> str:
    """格式化本地检索结果"""
    if not contexts:
        return "## 本地知识库检索结果\n\n未找到相关内容。"

    import os
    result = "## 本地知识库检索结果\n\n"
    for i, ctx in enumerate(contexts, 1):
        source = ctx.get('metadata', {}).get('source', '未知来源')
        if source and source != '未知来源':
            source = os.path.basename(source)
        score = ctx.get('score', 0)
        result += f"[{i}] 来源: {source} (相关度: {score:.2f})\n"
        result += f"内容: {ctx.get('content', '')}\n\n"
    return result


def hybrid_search(state: AgentState) -> AgentState:
    """混合搜索：本地 + 网络"""
    state["current_step"] = "🔄 正在进行混合搜索..."
    
    # 1. 本地检索
    local_result = get_rag_manager().query(state["current_query"], top_n=3)
    state["local_contexts"] = local_result["formatted"]
    
    # 2. 网络搜索
    search_query = state["current_query"]
    web_results = search_tool.invoke(search_query)
    
    # 格式化网络结果
    if isinstance(web_results, list):
        formatted_web = "\n\n".join([
            f"[网络{i+1}] 来源: {r.get('url', 'N/A')}\n内容: {r.get('content', '')}"
            for i, r in enumerate(web_results)
        ])
    else:
        formatted_web = str(web_results)
    
    state["search_results"] = formatted_web
    
    # 合并来源（将 numpy.float32 转换为 Python float）
    state["sources"] = [
        {"type": "local", "source": ctx.get("metadata", {}).get("source", ""), "score": float(ctx.get("score", 0))}
        for ctx in local_result["contexts"]
    ] + [
        {"type": "web", "source": r.get("url", ""), "score": 1.0}
        for r in (web_results if isinstance(web_results, list) else [])
    ]
    
    return state


def generate_answer(state: AgentState) -> AgentState:
    """生成带引用的答案"""
    state["current_step"] = "✍️ 正在生成答案..."
    
    query = state["current_query"]
    search_type = state.get("search_type", "none")
    
    # 构建上下文
    context_parts = []
    if state.get("local_contexts"):
        context_parts.append(state["local_contexts"])
    if state.get("search_results"):
        context_parts.append("## 网络搜索结果\n" + state["search_results"])
    
    context = "\n\n".join(context_parts) if context_parts else ""
    
    if context:
        prompt = f"""基于以下检索结果回答问题：

{context}

## 用户问题
{query}

## 要求
1. 综合本地知识库和网络信息回答
2. 使用 [来源N] 格式标注引用
3. 如果信息不足，请明确说明"""
    else:
        prompt = f"回答以下问题：{query}"
    
    messages = state["messages"] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    state["final_answer"] = response.content
    state["current_step"] = "✅ 完成"
    
    # 更新对话历史
    state["messages"].append(HumanMessage(content=query))
    state["messages"].append(AIMessage(content=response.content))
    
    return state


def search_web(state: AgentState) -> AgentState:
    """网络搜索节点"""
    state["current_step"] = "🔍 正在搜索网络..."

    query = state["current_query"]
    # 这里需要重写query，防止后续问到"它"等代词，不知道指代的是什么
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


def skip_search(state: AgentState) -> AgentState:
    """跳过搜索的节点（直接生成答案）"""
    state["current_step"] = "💭 无需搜索，直接回答..."
    return state


def reflect_on_results(state: AgentState) -> AgentState:
    """
    反思节点：评估检索结果是否足够回答问题

    这是 Agentic RAG 的关键进阶点：
    - LLM 评估检索到的信息是否足够、是否相关
    - 如果不足，生成改进的查询并触发重新搜索
    - 最多循环 max_loops 次，防止无限循环
    """
    state["current_step"] = "🤔 正在评估检索结果..."

    query = state["current_query"]
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)

    # 收集所有检索结果
    context_parts = []
    if state.get("local_contexts"):
        context_parts.append(f"本地知识库结果:\n{state['local_contexts']}")
    if state.get("search_results"):
        context_parts.append(f"网络搜索结果:\n{state['search_results']}")

    all_contexts = "\n\n".join(context_parts) if context_parts else "无检索结果"

    # 如果没有任何结果，直接标记为不足
    if not context_parts:
        state["reflection_result"] = "insufficient"
        state["reflection_reason"] = "没有检索到任何结果"
        state["loop_count"] = loop_count + 1
        return state

    # 让 LLM 评估结果质量
    reflect_prompt = f"""你是一个信息质量评估专家。请评估以下检索结果是否足以回答用户问题。

## 用户问题
{query}

## 检索结果
{all_contexts[:3000]}  # 限制长度避免 token 过多

## 评估标准
1. SUFFICIENT（充分）：检索结果直接回答了问题，信息完整、相关
2. INSUFFICIENT（不足）：检索结果相关但不完整，需要更多信息
3. IRRELEVANT（不相关）：检索结果与问题无关

## 输出格式（严格按此格式）
RESULT: [SUFFICIENT/INSUFFICIENT/IRRELEVANT]
REASON: [一句话说明原因]
REFINED_QUERY: [如果是 INSUFFICIENT和IRRELEVANT，给出改进的搜索查询；否则留空]

请评估："""

    response = llm.invoke([HumanMessage(content=reflect_prompt)])
    result_text = response.content.strip()

    # 解析 LLM 输出
    reflection_result = "sufficient"  # 默认充分
    reflection_reason = ""
    refined_query = ""

    for line in result_text.split("\n"):
        line = line.strip()
        if line.startswith("RESULT:"):
            result_value = line.replace("RESULT:", "").strip().upper()
            if result_value in ["SUFFICIENT", "INSUFFICIENT", "IRRELEVANT"]:
                reflection_result = result_value.lower()
        elif line.startswith("REASON:"):
            reflection_reason = line.replace("REASON:", "").strip()
        elif line.startswith("REFINED_QUERY:"):
            refined_query = line.replace("REFINED_QUERY:", "").strip()

    # 更新状态
    state["reflection_result"] = reflection_result
    state["reflection_reason"] = reflection_reason
    state["refined_query"] = refined_query if refined_query else query
    state["loop_count"] = loop_count + 1

    # 打印反思结果（调试用）
    print(f"  🔍 反思结果: {reflection_result}")
    print(f"  📝 原因: {reflection_reason}")
    if reflection_result == "insufficient" and refined_query:
        print(f"  🔄 改进查询: {refined_query}")
    print(f"  📊 当前循环: {state['loop_count']}/{max_loops}")

    return state


def refine_search(state: AgentState) -> AgentState:
    """
    改进搜索节点：使用改进后的查询重新搜索

    当 Reflector 判断结果不足时，用改进的查询重新检索
    """
    state["current_step"] = "🔄 正在使用改进的查询重新搜索..."

    refined_query = state.get("refined_query", state["current_query"])
    search_type = state.get("search_type", "web")

    print(f"  🔄 改进搜索: {refined_query}")

    # 根据搜索类型执行搜索
    if search_type == "local":
        result = get_rag_manager().query(refined_query, top_n=5)
        # 追加到现有结果
        existing = state.get("local_contexts", "")
        state["local_contexts"] = existing + "\n\n--- 改进搜索结果 ---\n" + result["formatted"]
        # 追加来源
        new_sources = [
            {"type": "local", "source": ctx.get("metadata", {}).get("source", ""), "score": float(ctx.get("score", 0))}
            for ctx in result["contexts"]
        ]
        state["sources"] = state.get("sources", []) + new_sources

    elif search_type == "web":
        results = search_tool.invoke(refined_query)
        if isinstance(results, list):
            formatted = "\n\n".join([
                f"来源: {r.get('url', 'N/A')}\n内容：{r.get('content', '')}"
                for r in results
            ])
        else:
            formatted = str(results)
        existing = state.get("search_results", "")
        state["search_results"] = existing + "\n\n--- 改进搜索结果 ---\n" + formatted

    elif search_type == "hybrid":
        # 混合搜索
        local_result = get_rag_manager().query(refined_query, top_n=3)
        web_results = search_tool.invoke(refined_query)

        existing_local = state.get("local_contexts", "")
        state["local_contexts"] = existing_local + "\n\n--- 改进搜索结果 ---\n" + local_result["formatted"]

        if isinstance(web_results, list):
            formatted_web = "\n\n".join([
                f"来源: {r.get('url', 'N/A')}\n内容: {r.get('content', '')}"
                for r in web_results
            ])
        else:
            formatted_web = str(web_results)
        existing_web = state.get("search_results", "")
        state["search_results"] = existing_web + "\n\n--- 改进搜索结果 ---\n" + formatted_web

    return state