# リアルタイム日英翻訳アプリ

ローカル環境で動作する日英双方向リアルタイム音声翻訳アプリケーション

## 機能

- リアルタイム音声認識（ASR）
- 日英双方向翻訳
- 音声活動検出（VAD）
- GUI/CLIインターフェース
- オフライン動作

## システム要件

### 最低要件
- CPU: 4コア/8スレッド
- RAM: 8GB
- ストレージ: 5GB

### 推奨要件（CPU）
- CPU: 8コア/16スレッド
- RAM: 16GB
- ストレージ: 15GB

### 推奨要件（GPU）
- CPU: 8コア/16スレッド
- RAM: 32GB
- GPU: RTX 4070 8GB / Apple M3 Max 38C
- ストレージ: 30GB

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/<org>/jp-en-rt-translator.git
cd jp-en-rt-translator

# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

## 使用方法

```bash
# GUIモードで起動
python src/app.py

# CLIモードで起動
python src/app.py --cli
```

## ライセンス

MIT License 