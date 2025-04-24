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

# Python依存関係とwhisper.cppのクローンとビルド
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# whisper.cppをビルド
RUN git clone https://github.com/ggml-org/whisper.cpp.git /tmp/whisper.cpp \
    && cd /tmp/whisper.cpp \
    && make \
    && mkdir -p /app/bin \
    && cp /tmp/whisper.cpp/main /app/bin/whisper-cli

# ===== モデルステージ =====
FROM python:3.11-slim AS models

WORKDIR /app/models

# モデルをダウンロード
RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

# ggml-small-q8_0.binモデルダウンロード
RUN curl -L -o ggml-small-q8_0.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin

# NLLB-200モデルのダウンロードと準備
RUN curl -L -o nllb-ja-en-q8_0.zip https://huggingface.co/models/nllb-200-distilled-600M-ja-en-q8_0.zip \
    && curl -L -o nllb-en-ja-q8_0.zip https://huggingface.co/models/nllb-200-distilled-600M-en-ja-q8_0.zip \
    && unzip nllb-ja-en-q8_0.zip -d nllb-ja-en-q8_0 \
    && unzip nllb-en-ja-q8_0.zip -d nllb-en-ja-q8_0 \
    && rm *.zip

# ===== 実行ステージ =====
FROM python:3.11-slim

WORKDIR /app

# 実行に必要なパッケージのみをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# ビルドステージから必要なものをコピー
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/bin /app/bin
COPY --from=models /app/models /app/models

# 環境変数の設定
ENV PATH="/app/bin:${PATH}"

# Python依存関係をインストール
RUN pip install --no-cache-dir /wheels/*

# アプリケーションコードをコピー
COPY app.py .
COPY static/ ./static/

# 非rootユーザーで実行
RUN useradd -m appuser
USER appuser

# ポートの公開
EXPOSE 8000

# アプリケーションの実行
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
