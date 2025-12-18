"""RAG 管理器 - 统一对外接口"""
import os
import hashlib
from typing import List, Dict, Set
from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import VectorStore
from src.rag.retriever import HybridRetriever
from src.rag.config import RAGConfig


class RAGManager:
    _instance = None  # 单例模式

    def __init__(self):
        self.loader = DocumentLoader()
        # 使用持久化模式，文档导入后重启不会丢失
        self.vector_store = VectorStore(
            embedding_model=RAGConfig.EMBEDDING_MODEL,
            persist_dir=RAGConfig.VECTOR_DB_DIR  # 启用持久化
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            rerank_model=RAGConfig.RERANK_MODEL
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _compute_file_hash(self, file_path: str) -> str:
        """
        计算文件的 MD5 哈希值

        用于判断文件内容是否已经导入过（即使文件名相同，内容变了也会重新导入）
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # 分块读取，避免大文件内存溢出
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_indexed_sources(self) -> Set[str]:
        """
        获取已经索引的文档来源集合

        从向量库的 metadata 中提取所有 source 字段
        返回格式: {"filename1.md:hash1", "filename2.pdf:hash2", ...}
        """
        indexed = set()
        all_docs = self.vector_store.get_all_documents()
        for doc in all_docs:
            source = doc.get("metadata", {}).get("source", "")
            file_hash = doc.get("metadata", {}).get("file_hash", "")
            if source:
                # 用 "文件名:哈希" 作为唯一标识
                filename = os.path.basename(source)
                indexed.add(f"{filename}:{file_hash}")
        return indexed

    def is_document_indexed(self, file_path: str) -> bool:
        """
        检查文档是否已经被索引

        判断逻辑：文件名 + 文件内容哈希 都匹配才算已索引
        这样即使文件名相同但内容变了，也会重新导入
        """
        filename = os.path.basename(file_path)
        file_hash = self._compute_file_hash(file_path)
        identifier = f"{filename}:{file_hash}"

        indexed_sources = self._get_indexed_sources()
        return identifier in indexed_sources

    def add_document(self, file_path: str, force: bool = False) -> int:
        """
        添加单个文档到知识库

        Args:
            file_path: 文档路径
            force: 是否强制重新导入（即使已存在）

        Returns:
            导入的 chunk 数量，如果已存在则返回 0
        """
        filename = os.path.basename(file_path)

        # 去重检查
        if not force and self.is_document_indexed(file_path):
            print(f"⏭️  跳过（已存在）: {filename}")
            return 0

        # 计算文件哈希，用于后续去重判断
        file_hash = self._compute_file_hash(file_path)

        # 加载并切分文档
        chunks = self.loader.load_and_split(
            file_path,
            chunk_size=RAGConfig.CHUNK_SIZE,
            overlap=RAGConfig.CHUNK_OVERLAP
        )

        documents = [chunk["content"] for chunk in chunks]
        # 在 metadata 中添加 file_hash，用于去重
        metadatas = []
        for chunk in chunks:
            meta = chunk["metadata"].copy()
            meta["file_hash"] = file_hash  # 添加哈希值
            metadatas.append(meta)

        return self.vector_store.add_documents(documents, metadatas)

    def add_documents_from_dir(self, dir_path: str, force: bool = False) -> int:
        """
        批量添加目录下的文档

        Args:
            dir_path: 文档目录路径
            force: 是否强制重新导入所有文档

        Returns:
            新导入的 chunk 总数
        """
        total_added = 0
        skipped = 0
        supported_extensions = ('.pdf', '.txt', '.md')

        files = [f for f in os.listdir(dir_path) if f.endswith(supported_extensions)]

        if not files:
            print(f"⚠️  目录为空或无支持的文件: {dir_path}")
            return 0

        print(f"📂 扫描到 {len(files)} 个文档")

        for filename in files:
            file_path = os.path.join(dir_path, filename)
            try:
                count = self.add_document(file_path, force=force)
                if count > 0:
                    total_added += count
                    print(f"✅ 已添加: {filename} ({count} chunks)")
                else:
                    skipped += 1
            except Exception as e:
                print(f"❌ 添加失败: {filename} - {e}")

        # 打印统计信息
        print(f"\n📊 导入统计: 新增 {total_added} chunks, 跳过 {skipped} 个已存在文档")
        return total_added

    def query(self, question: str, top_n: int = 5) -> Dict:
        """检索并返回相关文档"""
        # 检查知识库是否为空
        if self.vector_store.count() == 0:
            return {
                "contexts": [],
                "formatted": "## 本地知识库检索结果\n\n暂无文档，请先添加文档到知识库。"
            }

        contexts = self.retriever.retrieve(
            question,
            top_k=RAGConfig.VECTOR_SEARCH_TOP_K,
            top_n=top_n,
            vector_weight=RAGConfig.VECTOR_WEIGHT
        )
        return {
            "contexts": contexts,
            "formatted": self._format_contexts(contexts)
        }

    def _format_contexts(self, contexts: List[Dict]) -> str:
        """格式化检索结果，用于 Prompt"""
        if not contexts:
            return "## 本地知识库检索结果\n\n未找到相关内容。"

        result = "## 本地知识库检索结果\n\n"
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get('metadata', {}).get('source', '未知来源')
            # 只显示文件名，不显示完整路径
            if source and source != '未知来源':
                source = os.path.basename(source)
            score = ctx.get('score', 0)
            result += f"[{i}] 来源: {source} (相关度: {score:.2f})\n"
            result += f"内容: {ctx['content']}\n\n"
        return result

    def clear(self):
        """清空知识库"""
        self.vector_store.clear()

    def count(self) -> int:
        """获取知识库文档数量"""
        return self.vector_store.count()

    def list_documents(self) -> List[str]:
        """
        列出已索引的文档

        Returns:
            文档名列表（去重后）
        """
        all_docs = self.vector_store.get_all_documents()
        sources = set()
        for doc in all_docs:
            source = doc.get("metadata", {}).get("source", "")
            if source:
                sources.add(os.path.basename(source))
        return sorted(list(sources))
