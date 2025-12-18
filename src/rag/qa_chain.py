"""问答链"""
from openai import OpenAI
from typing import List, Dict

import config


class QAChain:
    """
    RAG 问答链

    功能：
    1. 构造带引用的 Prompt
    2. 调用大模型生成答案
    3. 返回答案 + 引用来源
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        # 1. 设定系统角色和指令
        system_instruction = "你是一个知识库问答助手。\n\n"

        # 2. 构造参考文档部分
        context_section = "## 参考文档\n"
        for i, doc in enumerate(contexts):
            # 格式：[1] 文档内容...
            context_section += f"[{i + 1}] {doc['content']}\n"

        # 3. 构造规则部分
        rules_section = """
                ## 规则
                1. 只能基于参考文档回答，不要编造
                2. 回答时请标注引用来源，如"根据[1]，..."
                3. 如果文档中没有相关信息，请说"根据现有资料无法回答"
                """

        # 4. 构造用户问题部分
        query_section = f"\n## 用户问题\n{query}\n"

        # 5. 组合所有部分
        final_prompt = context_section + rules_section + query_section

        # 返回 final_prompt
        return final_prompt

    def answer(self, query: str, contexts: List[Dict]) -> Dict:
        """
        生成答案
        """
        # 1. 构造完整的 Prompt
        full_prompt = self._build_prompt(query, contexts)

        # 2. 准备 API 调用的消息格式
        messages = [
            # 这里的角色通常是 'user'，将整个 Prompt 作为用户输入
            # 也可以考虑将 system_instruction 拆分到 role='system' 中，但一个 Prompt 也可以
            {"role":"system","content":"你是一个知识库问答助手,一切答案根据参考文档回答，不要捏造，不知道的直接说不知道"},
            {"role": "user", "content": full_prompt}
        ]

        # 3. 调用大模型 API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0  # RAG 倾向于低温度，以保证事实准确性
        )

        # 4. 提取答案文本
        answer_text = response.choices[0].message.content

        # 5. 返回结果，包含答案和原始的 contexts (作为 sources)
        return {
            "answer": answer_text,
            "sources": contexts
        }


# ============ 测试 ============
if __name__ == "__main__":
    # 替换成你的 API 信息
    chain = QAChain(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL
    )

    # 模拟检索结果
    contexts = [
        {"content": "RAG 是检索增强生成的缩写，它结合了检索和生成两种能力", "score": 0.95},
        {"content": "RAG 的核心流程：1.检索相关文档 2.构造Prompt 3.大模型生成答案", "score": 0.88},
        {"content": "LangChain 是一个大模型应用开发框架", "score": 0.75},
    ]

    result = chain.answer("什么是 RAG？", contexts)

    print("=" * 50)
    print(f"答案：{result['answer']}")
    print("=" * 50)
    print("引用来源：")
    for i, src in enumerate(result['sources'], 1):
        print(f"  [{i}] {src['content'][:50]}...")

