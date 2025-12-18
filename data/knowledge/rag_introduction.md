# RAG（检索增强生成）技术详解

## 什么是 RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的技术，通过从外部知识库检索相关信息来增强大语言模型的回答能力。

## 为什么需要 RAG？

大语言模型存在以下局限：
1. **知识截止**：训练数据有时间限制
2. **幻觉问题**：可能生成虚假信息
3. **私有数据**：无法访问企业内部知识

RAG 通过检索外部知识库解决这些问题。

## RAG 核心流程

```
用户问题 → 检索相关文档 → 构造 Prompt → LLM 生成答案
```

### 1. 文档处理

将文档切分成小块（Chunk），便于检索：

```python
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

### 2. 向量化

使用 Embedding 模型将文本转换为向量：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("shibing624/text2vec-base-chinese")
embeddings = model.encode(chunks)
```

### 3. 向量存储

将向量存入向量数据库（如 Chroma、FAISS、Milvus）：

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("knowledge_base")
collection.add(documents=chunks, embeddings=embeddings, ids=ids)
```

### 4. 检索策略

#### 向量检索
基于语义相似度检索，适合理解同义表达。

#### 关键词检索（BM25）
基于词频统计，适合精确匹配。

#### 混合检索（Hybrid Search）
结合向量和关键词检索，效果最佳：

```python
final_score = vector_score * 0.7 + keyword_score * 0.3
```

### 5. 重排序（Rerank）

使用 Cross-Encoder 对候选文档精排：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")
scores = reranker.predict([[query, doc] for doc in candidates])
```

## RAG 评估指标

1. **召回率**：相关文档是否被检索到
2. **准确率**：检索结果中相关文档的比例
3. **答案质量**：生成答案的正确性和完整性

## 常见优化技巧

1. **Chunk 策略**：按语义切分而非固定长度
2. **Query 改写**：扩展或重写用户问题
3. **多路召回**：使用多种检索策略
4. **上下文压缩**：去除冗余信息

## 实际应用场景

- 企业知识库问答
- 客服机器人
- 文档助手
- 代码搜索
