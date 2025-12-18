"""向量数据库封装"""
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class VectorStore:
    """
    Chroma 基本用法：
        client = chromadb.Client() #获取客户端，类似获取数据库
        collection = client.create_collection("my_collection") #获取数据库的表
        collection.add(documents=[...], ids=[...], embeddings=[...]) #添加一行数据
        results = collection.query(query_embeddings=[...], n_results=10) #查询
    """

    def __init__(self, embedding_model: str, collection_name: str = "knowledge_base",
                 persist_dir: str = None):
        """
        初始化向量存储

        Args:
            embedding_model: Embedding 模型名称
            collection_name: 集合名称（类似数据库的表名）
            persist_dir: 持久化目录，None 则使用内存模式
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # 关键区别：内存模式 vs 持久化模式
        if persist_dir:
            # 持久化模式：数据保存到磁盘，重启后数据还在
            import os
            os.makedirs(persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            # 内存模式：数据在内存中，重启后丢失
            self.client = chromadb.Client()

        # 获取或创建集合（get_or_create 避免重复创建报错）
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦距离
        )

        # 加载 Embedding 模型
        self.embedder = SentenceTransformer(embedding_model)


    def add_documents(self, documents: List[str], metadatas: List[Dict] = None)->int:
        """添加文档到向量库"""

        if not documents:
            return 0

        doc_embedding = self.embedder.encode(documents) #多个embedding向量 [[..], [..], [..]]
        ids = [str(uuid.uuid4()) for _ in documents] #每个文档有一个 id

        #对传入进来的metadatas做判断
        if metadatas is None:
            metadatas = [{} for _ in documents]
        elif len(metadatas) != len(documents):
            raise ValueError("metadata的长度必须匹配documents的长度")

        self.collection.add( #把向量化文档存入数据库的表中
            documents=documents,
            embeddings = doc_embedding.tolist(),
            ids = ids,
            metadatas= metadatas
        )
        return len(documents)

    def get_all_documents(self)->List[Dict]: #获取所有文档及元信息
        result = self.collection.get(
            include=['documents','metadatas']
        )
        return [
            {
                'content':result["documents"][i],
                'metadata':result["metadatas"][i]
            }
            for i in range(len(result["documents"]))
        ]

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """向量检索"""

        #先将query向量化
        query_vec = self.embedder.encode(query).tolist()

        #调用向量数据库（里面的表）进行查询
        results = self.collection.query(
            query_embeddings=[query_vec], #支持多条检索
            n_results=top_k
        )
        docs = results['documents'][0] #支持“同时查询多个 query”，所以它的返回结构是二维的：[[doc1],[doc2]...]
        metas = results['metadatas'][0]
        distances = results['distances'][0]
        ret = []
        for i in range(len(docs)):
            ret.append({
                "content":docs[i],
                "metadata":metas[i],
                "score":1 - distances[i] / 2 #距离转换成分数
            })
        return ret

    def clear(self):
        """清空向量库"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦距离
        )  # 表

    def count(self)->int:
        """获取文档数量"""
        return self.collection.count()


if __name__ == "__main__":
    store = VectorStore("shibing624/text2vec-base-chinese")

    # 测试添加
    docs = [
        "LangChain 是一个大模型应用开发框架",
        "RAG 是检索增强生成的缩写",
        "向量数据库用于存储向量",
    ]
    store.add_documents(docs, metadatas=[{"source": "test"} for _ in docs])
    print(f"文档数量: {store.count()}")  # 应该输出 3

    # 测试检索
    results = store.search("什么是 RAG", top_k=2)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")

    # 测试清空
    store.clear()
    print(f"清空后数量: {store.count()}")  # 应该输出 0