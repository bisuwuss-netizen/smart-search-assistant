"""
FastAPI 服务器 - 提供 RESTful API 和 SSE 流式输出

启动方式：
    cd smart-search-assistant
    uvicorn src.api.server:app --reload --port 8000

API 端点：
    POST /ask         - 普通问答（返回完整结果）
    POST /ask/stream  - 流式问答（SSE 实时输出）
    POST /documents   - 上传文档到知识库
    GET  /documents   - 列出已索引文档
    DELETE /documents - 清空知识库
    GET  /health      - 健康检查
"""
import sys
import os

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
import json
import uuid
import tempfile
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.graph_advanced import graph_advanced, create_initial_state
from src.rag.rag_manager import RAGManager
from src.config import Config


# ============ Pydantic 模型 ============
class AskRequest(BaseModel):
    """问答请求"""
    query: str
    thread_id: Optional[str] = None
    use_multi_query: bool = True
    max_loops: int = 3


class AskResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: List[dict]
    search_type: str
    loop_count: int
    reflection_result: str
    expanded_queries: List[str]
    thread_id: str


class DocumentInfo(BaseModel):
    """文档信息"""
    filename: str
    chunks: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    document_count: int
    version: str


# ============ 应用初始化 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("🚀 Smart Search Assistant API 启动中...")
    # 预热 RAG 管理器
    RAGManager.get_instance()
    print("✅ 服务就绪")
    yield
    print("👋 服务关闭")


app = FastAPI(
    title="Smart Search Assistant API",
    description="基于 LangGraph 的智能搜索助手 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ API 端点 ============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    rag = RAGManager.get_instance()
    return HealthResponse(
        status="healthy",
        document_count=rag.count(),
        version="1.0.0"
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    普通问答接口

    返回完整的回答结果，适合对延迟不敏感的场景
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state = create_initial_state(
        query=request.query,
        use_multi_query=request.use_multi_query,
        max_loops=request.max_loops
    )

    try:
        result = await asyncio.to_thread(
            graph_advanced.invoke, state, config
        )

        return AskResponse(
            answer=result.get("final_answer", ""),
            sources=result.get("sources", []),
            search_type=result.get("search_type", ""),
            loop_count=result.get("loop_count", 0),
            reflection_result=result.get("reflection_result", ""),
            expanded_queries=result.get("expanded_queries", []),
            thread_id=thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """
    流式问答接口（SSE）

    实时返回处理状态和最终答案，适合需要展示进度的场景

    SSE 事件格式：
        event: step
        data: {"step": "🔄 正在扩展查询...", "progress": 20}

        event: answer
        data: {"answer": "...", "sources": [...]}

        event: done
        data: {"status": "completed"}
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_generator():
        config = {"configurable": {"thread_id": thread_id}}
        state = create_initial_state(
            query=request.query,
            use_multi_query=request.use_multi_query,
            max_loops=request.max_loops
        )

        steps = [
            ("decide", "🤔 判断搜索类型...", 10),
            ("expand", "🔄 扩展查询...", 20),
            ("search", "🔍 执行搜索...", 40),
            ("reflect", "🧐 评估结果...", 60),
            ("answer", "✍️ 生成答案...", 80),
        ]

        try:
            # 发送开始事件
            yield f"event: start\ndata: {json.dumps({'thread_id': thread_id, 'query': request.query})}\n\n"

            # 模拟步骤进度（实际可以通过回调获取）
            for step_name, step_desc, progress in steps:
                yield f"event: step\ndata: {json.dumps({'step': step_desc, 'progress': progress})}\n\n"
                await asyncio.sleep(0.1)

            # 执行实际查询
            result = await asyncio.to_thread(
                graph_advanced.invoke, state, config
            )

            # 发送答案
            answer_data = {
                "answer": result.get("final_answer", ""),
                "sources": result.get("sources", []),
                "search_type": result.get("search_type", ""),
                "loop_count": result.get("loop_count", 0),
                "reflection_result": result.get("reflection_result", ""),
                "expanded_queries": result.get("expanded_queries", [])
            }
            yield f"event: answer\ndata: {json.dumps(answer_data, ensure_ascii=False)}\n\n"

            # 发送完成事件
            yield f"event: done\ndata: {json.dumps({'status': 'completed', 'thread_id': thread_id})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/documents", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    上传文档到知识库

    支持格式：.pdf, .txt, .md
    """
    # 验证文件类型
    allowed_extensions = ('.pdf', '.txt', '.md')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型，仅支持: {', '.join(allowed_extensions)}"
        )

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        rag = RAGManager.get_instance()
        chunks = rag.add_document(tmp_path)

        return DocumentInfo(
            filename=file.filename,
            chunks=chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        os.unlink(tmp_path)


@app.get("/documents")
async def list_documents():
    """列出已索引的文档"""
    rag = RAGManager.get_instance()
    documents = rag.list_documents()
    return {
        "count": len(documents),
        "documents": documents
    }


@app.delete("/documents")
async def clear_documents():
    """清空知识库"""
    rag = RAGManager.get_instance()
    rag.clear()
    return {"message": "知识库已清空"}


# ============ 主入口 ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
