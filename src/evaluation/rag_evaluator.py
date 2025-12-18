"""
RAG 评估模块

提供对检索和生成质量的评估指标：
1. 检索评估：Precision, Recall, MRR, NDCG
2. 生成评估：Faithfulness, Answer Relevancy
3. 端到端评估：Answer Correctness

使用方式：
    python -m src.evaluation.rag_evaluator

参考框架：RAGAS (https://github.com/explodinggradients/ragas)
"""
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.config import Config
from src.utils.llm_factory import LLMFactory


@dataclass
class EvaluationResult:
    """单个评估结果"""
    question: str
    expected_answer: Optional[str]
    generated_answer: str
    contexts: List[str]
    # 评估指标
    faithfulness: float  # 答案是否基于检索内容
    answer_relevancy: float  # 答案与问题的相关性
    context_precision: float  # 检索精确度
    context_recall: float  # 检索召回率（需要ground truth）


@dataclass
class EvaluationReport:
    """评估报告"""
    total_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    results: List[EvaluationResult]


class RAGEvaluator:
    """
    RAG 系统评估器

    使用 LLM 作为评判者（LLM-as-a-Judge）来评估：
    - Faithfulness: 答案是否忠实于检索到的上下文
    - Answer Relevancy: 答案是否回答了用户的问题
    - Context Precision: 检索到的内容是否精确相关
    - Context Recall: 是否检索到了所有必要的信息
    """

    def __init__(self):
        self.llm = LLMFactory.get_qwen_model()

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        评估答案忠实度

        检查答案中的陈述是否都能在检索到的上下文中找到依据
        """
        if not contexts:
            return 0.0

        context_text = "\n\n".join(contexts[:5])  # 限制长度

        prompt = f"""你是一个答案质量评估专家。请评估以下答案的"忠实度"（Faithfulness）。

## 检索到的上下文
{context_text}

## 生成的答案
{answer}

## 评估标准
忠实度衡量答案中的信息是否都能在上下文中找到依据。
- 1.0: 答案完全基于上下文，没有编造信息
- 0.7-0.9: 大部分基于上下文，少量合理推断
- 0.4-0.6: 部分基于上下文，部分可能是编造
- 0.0-0.3: 大量信息无法从上下文中找到依据

请只输出一个 0 到 1 之间的数字（保留两位小数）："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        评估答案相关性

        检查答案是否真正回答了用户的问题
        """
        prompt = f"""你是一个答案质量评估专家。请评估以下答案对问题的"相关性"（Answer Relevancy）。

## 用户问题
{question}

## 生成的答案
{answer}

## 评估标准
相关性衡量答案是否直接回答了用户的问题。
- 1.0: 完美回答了问题的所有方面
- 0.7-0.9: 回答了主要问题，可能遗漏细节
- 0.4-0.6: 部分回答了问题，但有偏离
- 0.0-0.3: 基本没有回答问题或完全跑题

请只输出一个 0 到 1 之间的数字（保留两位小数）："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def evaluate_context_precision(self, question: str, contexts: List[str]) -> float:
        """
        评估检索精确度

        检查检索到的内容是否都与问题相关
        """
        if not contexts:
            return 0.0

        context_text = "\n\n---\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts[:5])])

        prompt = f"""你是一个检索质量评估专家。请评估以下检索结果的"精确度"（Context Precision）。

## 用户问题
{question}

## 检索到的内容
{context_text}

## 评估标准
精确度衡量检索到的内容是否都与问题相关。
- 1.0: 所有检索内容都高度相关
- 0.7-0.9: 大部分内容相关，少量不太相关
- 0.4-0.6: 部分内容相关，部分不相关
- 0.0-0.3: 大部分内容与问题无关

请只输出一个 0 到 1 之间的数字（保留两位小数）："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def evaluate_context_recall(
        self,
        question: str,
        contexts: List[str],
        expected_answer: Optional[str] = None
    ) -> float:
        """
        评估检索召回率

        检查是否检索到了回答问题所需的所有信息
        需要 ground truth answer 来判断
        """
        if not expected_answer:
            return 0.5  # 无法评估，返回中性分数

        context_text = "\n\n".join(contexts[:5])

        prompt = f"""你是一个检索质量评估专家。请评估以下检索结果的"召回率"（Context Recall）。

## 用户问题
{question}

## 标准答案
{expected_answer}

## 检索到的内容
{context_text}

## 评估标准
召回率衡量检索到的内容是否包含了回答问题所需的所有信息。
对比标准答案，看检索内容是否覆盖了回答所需的关键信息。
- 1.0: 检索内容完全覆盖了标准答案所需的信息
- 0.7-0.9: 覆盖了大部分关键信息
- 0.4-0.6: 覆盖了部分关键信息
- 0.0-0.3: 几乎没有覆盖关键信息

请只输出一个 0 到 1 之间的数字（保留两位小数）："""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        expected_answer: Optional[str] = None
    ) -> EvaluationResult:
        """评估单个样本"""
        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            generated_answer=answer,
            contexts=contexts,
            faithfulness=self.evaluate_faithfulness(answer, contexts),
            answer_relevancy=self.evaluate_answer_relevancy(question, answer),
            context_precision=self.evaluate_context_precision(question, contexts),
            context_recall=self.evaluate_context_recall(question, contexts, expected_answer)
        )

    def evaluate_batch(
        self,
        samples: List[Dict]
    ) -> EvaluationReport:
        """
        批量评估

        Args:
            samples: 样本列表，每个样本包含:
                - question: 问题
                - answer: 生成的答案
                - contexts: 检索到的上下文列表
                - expected_answer: (可选) 标准答案
        """
        results = []
        for sample in samples:
            result = self.evaluate_single(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample.get("contexts", []),
                expected_answer=sample.get("expected_answer")
            )
            results.append(result)
            print(f"  ✓ 评估完成: {sample['question'][:30]}...")

        # 计算平均分
        n = len(results)
        return EvaluationReport(
            total_samples=n,
            avg_faithfulness=sum(r.faithfulness for r in results) / n if n else 0,
            avg_answer_relevancy=sum(r.answer_relevancy for r in results) / n if n else 0,
            avg_context_precision=sum(r.context_precision for r in results) / n if n else 0,
            avg_context_recall=sum(r.context_recall for r in results) / n if n else 0,
            results=results
        )


def print_report(report: EvaluationReport):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("📊 RAG 评估报告")
    print("=" * 60)

    print(f"\n📈 总体指标 ({report.total_samples} 个样本)")
    print("-" * 40)
    print(f"  忠实度 (Faithfulness):     {report.avg_faithfulness:.2%}")
    print(f"  答案相关性 (Relevancy):    {report.avg_answer_relevancy:.2%}")
    print(f"  检索精确度 (Precision):    {report.avg_context_precision:.2%}")
    print(f"  检索召回率 (Recall):       {report.avg_context_recall:.2%}")

    print(f"\n📝 详细结果")
    print("-" * 40)
    for i, r in enumerate(report.results, 1):
        print(f"\n[{i}] {r.question[:50]}...")
        print(f"    忠实度: {r.faithfulness:.2f} | 相关性: {r.answer_relevancy:.2f}")
        print(f"    精确度: {r.context_precision:.2f} | 召回率: {r.context_recall:.2f}")


# ============ 测试用例 ============
if __name__ == "__main__":
    from src.graph_advanced import ask

    print("🧪 RAG 评估演示")
    print("=" * 60)

    # 测试问题
    test_questions = [
        {
            "question": "什么是 LangGraph？",
            "expected_answer": "LangGraph 是一个用于构建有状态、多角色的 LLM 应用的框架。"
        },
        {
            "question": "RAG 的核心步骤是什么？",
            "expected_answer": "RAG 的核心步骤包括：检索（Retrieval）、增强（Augmentation）、生成（Generation）。"
        }
    ]

    # 执行问答并收集结果
    samples = []
    for q in test_questions:
        print(f"\n❓ 测试问题: {q['question']}")
        result = ask(q['question'], thread_id=f"eval-{hash(q['question'])}")

        contexts = []
        if result.get('local_contexts'):
            contexts.append(result['local_contexts'])
        if result.get('search_results'):
            contexts.append(result['search_results'])

        samples.append({
            "question": q['question'],
            "answer": result['answer'],
            "contexts": contexts, #检索内容
            "expected_answer": q.get('expected_answer')
        })
        print(f"   ✓ 获得答案: {result['answer'][:100]}...")

    # 评估
    print("\n" + "=" * 60)
    print("🔍 开始评估...")
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_batch(samples)

    # 打印报告
    print_report(report)

    # 保存报告
    report_dict = {
        "total_samples": report.total_samples,
        "avg_faithfulness": report.avg_faithfulness,
        "avg_answer_relevancy": report.avg_answer_relevancy,
        "avg_context_precision": report.avg_context_precision,
        "avg_context_recall": report.avg_context_recall,
        "results": [asdict(r) for r in report.results]
    }

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    print(f"\n💾 报告已保存到 evaluation_report.json")
