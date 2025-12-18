"""检索器：Hybrid Search + Rerank"""
import jieba
import numpy as np
from collections import Counter

from numpy import argsort
from sentence_transformers import CrossEncoder
from typing import List, Dict


# 假设 VectorStore 已实现
from src.rag.vector_store import VectorStore


class HybridRetriever:
    """
    混合检索器

    流程：
    1. 向量检索（召回）
    2. 关键词检索（补充）
    3. 融合两路结果
    4. Rerank 精排
    """

    def __init__(self, vector_store: VectorStore, rerank_model: str = None):
        """
        参数：
            vector_store: 向量数据库实例
            rerank_model: Rerank 模型名称，None 则不使用 Rerank
        """
        self.vector_store = vector_store
        self.reranker = CrossEncoder(rerank_model) if rerank_model else None

        # 关键词检索需要的数据（从 vector_store 同步）
        self.documents = []  # 原始文档
        self.tokenized_docs = []  # 分词后的文档

    def _sync_documents(self):
        """
        从向量库同步文档用于关键词检索

        提示：Chroma 可以用 collection.get() 获取所有文档
        """
        all_data = self.vector_store.get_all_documents()
        self.documents = [dic['content'] for dic in all_data]
        self.tokenized_docs = []
        for doc_text in self.documents:
            tokens = list(jieba.cut(doc_text))
            self.tokenized_docs.append(tokens)

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """
        关键词检索（简化版 BM25）

        返回：[{"content": "...", "score": 0.8}, ...]
        """
        # 1. 对查询进行分词，并去重
        query_tokens = set(jieba.cut(query))
        scores = []  # 记录query 在每一个文档的得分情况

        # 2. 遍历所有文档计算得分
        for i, doc_tokens in enumerate(self.tokenized_docs):
            doc_counter = Counter(doc_tokens)  # 将每一个文档转换成一个词频字典
            frequency = sum(doc_counter.get(t, 0) for t in query_tokens)
            score = frequency / (len(doc_tokens) + 1)
            scores.append(score)

        # 3. 排序并取出 top_k
        scores = np.array(scores)
        if not self.documents:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]

        # 4. 转换成目标格式 List[Dict]
        results = []
        for i in top_indices:
            results.append({
                "content": self.documents[i],
                "score": scores[i]
            })

        return results

    def _merge_results(self, vector_results: List[Dict], keyword_results: List[Dict],
                       vector_weight: float = 0.7) -> List[Dict]:
        #融合两路检索结果

        # max-min归一化
        def normalize(results):
            if not results:
                return {}
            #拿到分数
            scores = [r['score'] for r in results]
            min_score = min(scores)
            max_score = max(scores)

            if min_score == max_score:
                return {r['content'] : 1.0 for r in results}
            return {r['content']: (r['score'] - min_score) / (max_score - min_score) for r in results}

        vector_scores = normalize(vector_results)
        keyword_scores = normalize(keyword_results)

        #融合分数
        all_content = set(vector_scores.keys()) | set(keyword_scores.keys())
        final_score = {}

        for content in all_content:
            v_score = vector_scores.get(content,0)
            k_score = keyword_scores.get(content,0)
            final_score[content] = v_score * vector_weight + k_score * (1-vector_weight)

        #排序
        sorted_content = sorted(final_score.keys(),key = lambda x: final_score[x],reverse=True)

        return [
            {
                "content": content,
                "score": final_score[content]
            }
            for content in sorted_content
        ]

    def _rerank(self, query: str, candidates: List[Dict], top_n: int) -> List[Dict]:
        if not self.reranker:
            # 如果没有 Rerank 模型，直接返回前 top_n 个候选
            return candidates[:top_n]

        # 1. 构造模型输入：[[query, doc1], [query, doc2], ...]
        sentences_pairs = [[query, candidate['content']] for candidate in candidates]

        # 2. 调用模型获取分数
        # scores 是一个 NumPy 数组
        scores = self.reranker.predict(sentences_pairs)
        # 3. 排序
        sorted_indices = np.argsort(scores)[::-1][:top_n]

        return [
            {
                "content": candidates[i]['content'],
                "score": scores[i]
            }
            for i in  sorted_indices
        ]


    def retrieve(self, query: str, top_k: int = 20, top_n: int = 5,
                 vector_weight: float = 0.7, use_rerank: bool = True) -> List[Dict]:
        """
        完整检索流程

        参数：
            query: 查询文本
            top_k: 粗排召回数量
            top_n: 精排后保留数量
            vector_weight: 向量检索权重
            use_rerank: 是否使用 Rerank
        """
        # 0. 同步文档
        self._sync_documents()
        # 1. 向量检索
        vector_results = self.vector_store.search(query, top_k)
        # 2. 关键词检索
        keyword_results = self._keyword_search(query,top_k)
        candidates = self._merge_results(vector_results,keyword_results,vector_weight)
        # 3. Rerank（如果启用）
        #TODO: 返回值适配 LangChain Document 格式
        if use_rerank:
           return self._rerank(query, candidates, top_n=top_n)
        else:
            return candidates[:top_n]


