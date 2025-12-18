"""文档加载与切分"""
import fitz  # PyMuPDF

"""
fitz.open() 是 PyMuPDF 提供的接口，
专门用来读取 PDF 这种二进制的排版文件格式，
它知道如何解析 PDF 里的页面、文字层、图片等。
PDF 不是普通的纯文本文件，with open(..., 'r', encoding=...) 只适合打开 .txt 这种文本文件，
读出来的就是字符流。
PDF 则是结构化的，
需要库帮你解析出每一页的文字，
所以需要用 fitz.open() 来创建文档对象，再逐页提取文字。

常用的 PyMuPDF（fitz）API，可以先记住这几个：
 1.fitz.open(path_or_stream)
打开 PDF 文件，返回一个 Document 对象，用它来访问页数、页面、元数据。
例如：doc = fitz.open("report.pdf")
2.doc.page_count 或 doc.pageCount
获取总页数，常用在循环提取每一页内容时做范围判断。
3.doc.load_page(page_index) 或 doc[page_index]
加载某一页（从 0 开始计数），返回 Page 对象。
如：page = doc.load_page(0)
4.page.get_text("text") 或 page.get_text()
提取这一页的纯文本。参数可以指定不同的提取模式，"text" 是默认纯文本。
例如：text = page.get_text("text")
5.page.get_text("blocks")
如果想更细粒度，也可以用 blocks/pdfminer风格提取；不过入门先用 "text" 更简单。
6.doc.close()
处理完记得关闭文档（或者用 with fitz.open(...) as doc: 自动关闭）。
基本流程就是：open → 遍历页 → get_text → close。这些 API 足够你实现 load_pdf 了。
"""
from typing import List


class DocumentLoader:
    """
    1. load_pdf(file_path) - 加载 PDF 文件
    2. load_txt(file_path) - 加载 TXT 文件
    3. split_text(text, chunk_size, overlap) - 切分文本
    """

    def load_pdf(self, file_path: str) -> str:
        """
        加载 PDF 文件，返回纯文本
        提示：使用 fitz.open() 打开，遍历每页提取文字
        """
        text = []
        with fitz.open(file_path) as doc:
            for page_index in range(doc.page_count):
                # 拿到 page 对象
                page = doc.load_page(page_index)
                # 获取当页文字
                page_text = page.get_text("text")
                text.append(page_text)
        return "\n".join(text)

    def load_txt(self, file_path: str) -> str:
        """加载 TXT/MD 文件"""
        text = []
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load(self, file_path: str) -> str:
        """自动识别文件类型并加载"""
        if file_path.endswith('.pdf'):
            return self.load_pdf(file_path)
        elif file_path.endswith(('.txt', '.md')):  # 注意这里是元组
            return self.load_txt(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")

    def split_text(self, text: str, chunk_size: int = 500,
                   overlap: int = 100) -> List[str]:

        chunk_size = int(chunk_size)
        overlap = int(overlap)
        """
        切分文本

        要求：
        1. 每个 chunk 大小约为 chunk_size
        2. 相邻 chunk 有 overlap 个字符重叠
        3. 尽量在句号、换行处切分，保持语义完整 （检查每个 chunk 末50字符）
        """
        if not text:
            return []

        start = 0
        chunks = []
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)

            #如果不是最后一个块
            if end < text_len:
                search_start = max(end - 50, start)
                chunk_tail = text[search_start:end]
                # 判断切分的 chunk 末尾 50个字符里，是否有换行符
                best_split = -1
                # 先按照段落分割，再按照句子分割
                for sep in ["\n\n", "\n"]:
                    pos = chunk_tail.rfind(sep)
                    if pos != -1:  # 找到换行符
                        best_split = search_start + pos + len(sep) #best_split处于换行符的后一个位置
                if best_split == -1:
                    #没找到换行符，那就找是否有完整的句子
                    for sep in ['.','。','！','？','?',"!"]:
                        pos = chunk_tail.rfind(sep)
                        if pos !=-1:
                            best_split = search_start + pos + len(sep)

                if best_split > start:
                    end = best_split

            #提取 chunk
            chunk = text[start:end]
            if chunk:
                chunks.append(chunk)
            #计算下一个位置
            start = end - overlap if end < text_len else text_len

        return chunks

    def load_and_split(self, file_path: str, chunk_size: int = 500,overlap: int = 100) -> List[dict]:
        """
        加载文件并切分，返回带元数据的 chunks
        面试加分点：保留来源信息，方便后续引用溯源
        """
        text = self.load(file_path)
        chunks = self.split_text(text,chunk_size,overlap)
        return [
            {
            "content":chunk,
            "metadata":{
                "source":file_path,
                "chunk_index":i
            }
        }
            for i,chunk in enumerate(chunks)]
