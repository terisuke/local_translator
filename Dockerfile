# マルチステージビルド用Dockerfile

# ===== ビルドステージ =====
FROM python:3.11-slim AS builder

WORKDIR /app

# ビルドツールとシステム依存関係のインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係のインストール
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# ===== 実行ステージ =====
FROM python:3.11-slim

WORKDIR /app

# 実行に必要なパッケージのみをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ビルドステージから必要なものをコピー
COPY --from=builder /app/wheels /wheels

# Python依存関係をインストール
RUN pip install --no-cache-dir /wheels/*

# アプリケーションコードをコピー
COPY app.py .
COPY static/ ./static/

# モデルディレクトリをコピー
COPY models/ ./models/

# 環境変数ファイルをコピー
COPY .env .

# 非rootユーザーで実行
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# ポートの公開
EXPOSE 8000

# アプリケーションの実行
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
