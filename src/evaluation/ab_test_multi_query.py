"""
Multi-Query A/B 对照实验 (优化版)

通过对比实验验证查询扩展（Multi-Query）对检索质量和执行效率的影响。
"""
import asyncio
from src.graph import graph
from src.evaluation.rag_evaluator import RAGEvaluator

# 🧪 复杂的测试用例（需要多维度召回）
TEST_CASES = [
    {
        "question": "LangGraph 的状态管理是如何实现的？请详细说明 checkpoint, thread_id 和 State 的关系。",
        "expected_answer": "LangGraph 通过 StateGraph 定义状态结构(State)，使用检查点(checkpoint)持久化状态. thread_id 用于标识不同的会话，使得系统可以恢复和继续特定 thread 的执行过程。"
    },
    {
        "question": "分析智能搜索助手（Smart Search Assistant）中反思机制（Reflector）的作用及其流程。",
        "expected_answer": "反思机制通过 LLM 评估检索结果的充分性。流程包括：评估信息、判断是否足够（SUFFICIENT/INSUFFICIENT）、若不足则生成改进查询并重试搜索，直到达到最大循环次数或信息满足要求。"
    },
    {
        "question": "印度与东盟在数字经济合作中面临哪些挑战？分别从政策和基础设施两个角度说明。",
        "expected_answer": "面临挑战包括政策层面的监管差异、数据隐私标准不一；以及基础设施层面的数字鸿沟、网络连接不均衡和跨境支付系统的兼容性问题。"
    }
]

async def run_experimental_run(use_multi_query: bool):
    """
    运行一组实验
    """
    print(f"\n🚀 开始实验: {'Multi-Query (开启)' if use_multi_query else 'Baseline (关闭)'}")
    print("=" * 60)
    
    samples = []
    loop_counts = []
    evaluator = RAGEvaluator()
    
    for i, case in enumerate(TEST_CASES, 1):
        q = case["question"]
        print(f"[{i}/{len(TEST_CASES)}] 处理问题: {q[:40]}...")
        
        # 构造初始状态
        state = {
            "messages": [],
            "current_query": q,
            "use_multi_query": use_multi_query,
            "max_loops": 3,
            "loop_count": 0
        }
        
        # 运行图
        config = {"configurable": {"thread_id": f"ab-test-{'mq' if use_multi_query else 'base'}-{i}"}}
        result = graph.invoke(state, config)
        
        # 记录循环次数
        final_loops = result.get("loop_count", 0)
        loop_counts.append(final_loops)
        
        # 收集上下文
        contexts = []
        if result.get("local_contexts"):
            contexts.append(result["local_contexts"])
        if result.get("search_results"):
            contexts.append(result["search_results"])
            
        samples.append({
            "question": q,
            "answer": result["final_answer"],
            "contexts": contexts,
            "expected_answer": case["expected_answer"]
        })
        
    print("\n🔍 正在评估质量指标...")
    report = evaluator.evaluate_batch(samples)
    avg_loops = sum(loop_counts) / len(loop_counts)
    return report, avg_loops

async def main():
    # 1. 运行 Baseline
    baseline_report, base_avg_loops = await run_experimental_run(use_multi_query=False)
    
    # 2. 运行 Multi-Query 版
    mq_report, mq_avg_loops = await run_experimental_run(use_multi_query=True)
    
    # 3. 输出对比表格
    print("\n" + "🏆" * 10 + " A/B 实验最终结果对比 " + "🏆" * 10)
    print("-" * 85)
    print(f"{'指标 (Metric)':<30} | {'Baseline':<15} | {'Multi-Query':<15} | {'提升 (Lift)':<10}")
    print("-" * 85)
    
    metrics = [
        ("忠实度 (Faithfulness)", baseline_report.avg_faithfulness, mq_report.avg_faithfulness),
        ("答案相关性 (Relevancy)", baseline_report.avg_answer_relevancy, mq_report.avg_answer_relevancy),
        ("检索精确度 (Precision)", baseline_report.avg_context_precision, mq_report.avg_context_precision),
        ("检索召回率 (Recall)", baseline_report.avg_context_recall, mq_report.avg_context_recall),
    ]
    
    for name, base, mq in metrics:
        lift = (mq - base) / base if base > 0 else 0
        print(f"{name:<24} | {base:>14.2%} | {mq:>14.2%} | {lift:>+10.1%}")
    
    # 效率指标
    loop_reduction = (base_avg_loops - mq_avg_loops) / base_avg_loops if base_avg_loops > 0 else 0
    print("-" * 85)
    print(f"{'平均检索循环数 (Avg Loops)':<24} | {base_avg_loops:>14.2f} | {mq_avg_loops:>14.2f} | {loop_reduction:>+10.1%}")
    print("-" * 85)

if __name__ == "__main__":
    asyncio.run(main())
