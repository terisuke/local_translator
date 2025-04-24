import os
from dotenv import load_dotenv
import requests

# 環境変数を読み込む
load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not token:
    print("トークンが設定されていません")
    exit(1)

# トークンをテスト
headers = {
    "Authorization": f"Bearer {token}"
}

try:
    response = requests.get(
        "https://huggingface.co/api/models/facebook/nllb-200-distilled-600M",
        headers=headers
    )
    if response.status_code == 200:
        print("トークンは有効です")
        print("レスポンス:", response.json())
    else:
        print(f"トークンが無効です。ステータスコード: {response.status_code}")
        print("エラーメッセージ:", response.text)
except Exception as e:
    print("エラーが発生しました:", e) 