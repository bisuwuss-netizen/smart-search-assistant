"""
SSE（Server-Sent Events）最简Demo

运行方式：
    python sse_demo.py

然后打开另一个终端测试：
    curl http://localhost:8000/stream

你会看到数据一条一条推送出来！
"""
import time
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()


def event_generator():
    """
    这是一个生成器函数
    每次 yield 会向客户端推送一条数据
    """
    
    # 模拟 AI 处理的5个步骤
    steps = [
        "🤔 正在理解你的问题...",
        "🔍 正在搜索相关资料...",
        "📚 正在阅读文档...",
        "🧠 正在思考答案...",
        "✍️ 正在组织语言...",
    ]
    
    # 逐个推送进度
    for i, step in enumerate(steps):
        # SSE 格式固定：event: 事件名\ndata: 数据\n\n
        data = {"step": step, "progress": (i + 1) * 20}
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        time.sleep(1)  # 每秒推送一条
    
    # 推送最终答案
    answer = "RAG 是 Retrieval-Augmented Generation 的缩写，即检索增强生成。它结合了搜索和生成，让 AI 能够基于真实资料回答问题。"
    yield f"data: {json.dumps({'answer': answer}, ensure_ascii=False)}\n\n"
    
    # 推送完成信号
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.get("/stream")
def stream():
    """
    SSE 接口
    
    关键点：
    1. 返回 StreamingResponse
    2. media_type 必须是 "text/event-stream"
    """
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/normal")
def normal():
    """
    普通接口（对比用）
    必须等 5 秒才能返回
    """
    time.sleep(5)
    return {"answer": "这是答案，但你等了5秒才看到"}


if __name__ == "__main__":
    print("=" * 60)
    print("SSE Demo 启动!")
    print("=" * 60)
    print("\n测试方法：")
    print("  1. 普通接口: curl http://localhost:8000/normal")
    print("     → 等待5秒，一次性返回")
    print("\n  2. SSE接口:  curl http://localhost:8000/stream")
    print("     → 每秒推送一条数据")
    print("\n" + "=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
