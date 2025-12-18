"""
SSE 最简Demo
运行: python sse_demo.py
测试: curl http://localhost:8000/stream
"""
import time
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()


def event_generator():
    """生成器：每次 yield 向客户端推送一条数据"""

    steps = [
        "🤔 正在理解问题...",
        "🔍 正在搜索资料...",
        "📚 正在阅读文档...",
        "🧠 正在思考答案...",
        "✍️ 正在组织语言...",
    ]

    for i, step in enumerate(steps):
        data = {"step": step, "progress": (i + 1) * 20}
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        time.sleep(0.38)  # 每秒推送一条

    yield f"data: {json.dumps({'answer': '这是最终答案！'},ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.get("/")
def stream():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# @app.get("/")
# def index_page():
#     return "/stream"

if __name__ == "__main__":
    print("启动服务，测试命令: curl http://localhost:8000/stream")
    uvicorn.run(app, host="0.0.0.0", port=8000)