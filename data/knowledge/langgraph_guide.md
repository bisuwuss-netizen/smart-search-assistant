# LangGraph 完全指南

## 什么是 LangGraph？

LangGraph 是由 LangChain 团队开发的一个用于构建有状态、多角色应用程序的框架。它基于图（Graph）的概念，让开发者可以定义复杂的工作流程。

## 核心概念

### 1. State（状态）

State 是 LangGraph 的核心，用于在节点之间传递数据。通常使用 TypedDict 定义：

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_query: str
    search_results: str
```

### 2. Node（节点）

节点是执行具体任务的函数，接收 State 并返回更新后的 State：

```python
def search_node(state: AgentState) -> AgentState:
    query = state["current_query"]
    results = search_tool.invoke(query)
    state["search_results"] = results
    return state
```

### 3. Edge（边）

边定义了节点之间的连接关系：
- 普通边：无条件跳转
- 条件边：根据条件选择下一个节点

### 4. Checkpointer（检查点）

用于持久化状态，支持多轮对话：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("checkpoints.db")
graph = workflow.compile(checkpointer=memory)
```

## LangGraph vs LangChain Agent

| 特性 | LangGraph | LangChain Agent |
|------|-----------|-----------------|
| 流程控制 | 显式定义 | 隐式（LLM决定） |
| 可预测性 | 高 | 低 |
| 调试难度 | 低 | 高 |
| 适用场景 | 复杂工作流 | 简单任务 |

## 最佳实践

1. **状态设计**：保持 State 简洁，只包含必要字段
2. **节点职责**：每个节点只做一件事
3. **错误处理**：在节点内部处理异常
4. **测试**：为每个节点编写单元测试

## 常见问题

### Q: 如何实现多轮对话？
A: 使用 Checkpointer 持久化 State，通过 thread_id 区分不同会话。

### Q: 如何处理并行执行？
A: 使用 `add_node` 添加多个独立节点，LangGraph 会自动并行执行。

### Q: State 的更新策略是什么？
A: 默认是覆盖，使用 `Annotated` 和 reducer 函数可以实现追加等策略。
