# 🔍 Smart Search Assistant

基于 LangGraph 的智能搜索助手，支持多轮对话、自动判断搜索、查询改写、流式输出。

## ✨ 核心功能

- 🤖 **智能决策**：自动判断问题是否需要网络搜索
- 🔄 **查询改写**：基于对话历史优化搜索查询
- 💬 **多轮对话**：支持上下文理解和追问
- 🔄 **流式输出**：实时显示执行进度
- 💾 **持久化存储**：自动保存对话历史
- 🛠️ **可扩展架构**：基于 LangGraph 图结构

## 🎯 技术亮点

### 1. 查询改写
当用户使用代词（"它"、"这个"）追问时，系统会基于对话历史重写查询：
```
用户：介绍一下 LangGraph
助手：LangGraph 是...

用户：它的主要优势是什么？
系统改写：LangGraph 的主要优势是什么？ ✅
```

### 2. 状态管理
使用 LangGraph 的 State 机制实现数据共享：
- 所有节点共享同一个 State
- 自动持久化到 SQLite
- 支持多用户隔离

### 3. 流程控制
```
用户输入 
  ↓
[判断节点] - 决定是否搜索
  ↓
[搜索节点] - 改写查询并搜索（可选）
  ↓
[生成节点] - 结合历史和搜索结果生成答案
  ↓
返回结果
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Keys:
# DASHSCOPE_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here
```

### 3. 运行示例
```bash
# 流式输出演示
python examples/streaming_demo.py

# 多轮对话演示
python examples/basic_usage.py
```

## 📖 使用方法

### 基础用法
```python
from src.graph import graph
from src.state import AgentState

# 创建初始状态
state = AgentState(
    messages=[],
    current_query="你的问题",
    need_search=False,
    search_results="",
    final_answer="",
    current_step=""
)

# 执行
config = {"configurable": {"thread_id": "user1"}}
result = graph.invoke(state, config)
print(result["final_answer"])
```

### 多轮对话
```python
from examples.basic_usage import ConversationManager

manager = ConversationManager(thread_id="user123")

# 第一轮
answer1 = manager.ask("什么是 Python？")

# 第二轮（自动加载历史）
answer2 = manager.ask("它适合初学者吗？")
```

## 🏗️ 项目结构
```
smart-search-assistant/
├── src/
│   ├── config.py          # 配置管理
│   ├── state.py           # State 定义
│   ├── tools.py           # 搜索工具
│   ├── nodes.py           # 节点函数
│   └── graph.py           # Graph 定义
├── examples/
│   ├── streaming_demo.py  # 流式输出示例
│   └── basic_usage.py     # 多轮对话示例
├── tests/
│   ├── test_tools.py      # 工具测试
│   ├── test_nodes.py      # 节点测试
│   └── test_graph.py      # 集成测试
├── checkpoints/           # 持久化存储
├── README.md
├── requirements.txt
└── .env.example
```

## 🔧 配置说明

在 `src/config.py` 中可配置：
- `MODEL_NAME`: 使用的 LLM 模型
- `MAX_HISTORY_MESSAGES`: 历史消息保留数量
- `CHECKPOINT_DIR`: Checkpoint 存储路径

## 📊 技术栈

- **LangGraph**: 工作流编排
- **LangChain**: LLM 集成
- **DeepSeek**: 大语言模型（通过阿里云百炼）
- **Tavily**: 网络搜索 API
- **SQLite**: 持久化存储

## 🎓 核心概念

### State（状态）
所有数据都存储在 State 中，节点之间通过 State 共享数据：
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]  # 对话历史
    current_query: str            # 当前问题
    need_search: bool             # 是否需要搜索
    search_results: str           # 搜索结果
    final_answer: str             # 最终答案
```

### 节点（Node）
每个节点是一个纯函数：接收 State，返回更新后的 State：
```python
def decide_search(state: AgentState) -> AgentState:
    # 读取 state
    query = state["current_query"]
    
    # 执行逻辑
    need_search = judge_need_search(query)
    
    # 更新 state
    state["need_search"] = need_search
    return state
```

### 持久化（Checkpointer）
使用 `thread_id` 区分不同用户，自动保存和恢复对话：
```python
config = {"configurable": {"thread_id": "user123"}}
result = graph.invoke(state, config)  # 自动保存
```

## 📈 实际运行效果
```
Q1: 介绍一下 LangGraph
  🤔 正在分析问题...
  🔍 正在搜索网络...
  ✅ 完成
A1: LangGraph 是由 LangChain 团队开发的...

Q2: 它的主要优势是什么？
  🤔 正在分析问题...
  原始查询: 它的主要优势是什么？
  改写查询: LangGraph 的主要优势是什么？ ✅
  🔍 正在搜索网络...
  ✅ 完成
A2: LangGraph 的主要优势体现在...
```

## 🧪 测试
```bash
# 运行所有测试
pytest tests/ -v

# 测试特定模块
python tests/test_tools.py
python tests/test_nodes.py
python tests/test_graph.py
```

## 📝 开发计划

### MVP（已完成）✅
- [x] 基础搜索功能
- [x] 多轮对话
- [x] 流式输出
- [x] 查询改写
- [x] 持久化

### 计划中（方案 B）
- [ ] Interrupt 人工审批
- [ ] 多源搜索聚合
- [ ] Web UI（Streamlit）
- [ ] 引用溯源功能
- [ ] 子图（Subgraph）集成

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT

## 👤 作者

李林 - AI 研究生 @ 北京信息科技大学

## 🔗 相关链接

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [DeepSeek API](https://platform.deepseek.com/)
- [Tavily Search](https://tavily.com/)