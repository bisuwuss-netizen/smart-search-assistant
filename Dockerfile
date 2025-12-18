# Smart Search Assistant Dockerfile
# 基于 Python 3.11 的多阶段构建

# ============ 构建阶段 ============
FROM python:3.11-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 创建虚拟环境并安装依赖
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir Cython && \
    pip install --no-cache-dir -r requirements.txt


# ============ 运行阶段 ============
FROM python:3.11-slim as runtime

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制项目代码
COPY src/ ./src/
COPY data/ ./data/

# 创建必要的目录
RUN mkdir -p /app/checkpoints /app/data/vector_db /app/data/knowledge

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000 8501

# 默认启动 FastAPI 服务
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
