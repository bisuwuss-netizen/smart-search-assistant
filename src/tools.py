from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
# 方案1：使用 Tavily（推荐）
def create_search_tool():
    return TavilySearch(
        max_results=3,  # 限制返回结果数量，节省 Token
        search_depth="advanced",  # 使用更深层次的搜索，提高结果质量
        include_answer=True  # **关键：** 要求 Tavily 返回一个即时答案
    )

# 方案2：使用 DuckDuckGo（免费备选）
# from langchain_community.tools import DuckDuckGoSearchRun
# def create_search_tool():
#     return DuckDuckGoSearchRun()