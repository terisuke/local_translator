"""
モデルダウンロードスクリプト
必要なモデルファイルをダウンロードします
"""

import os
import sys
import logging
import requests
import zipfile
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = "models"

def download_file(url: str, save_path: str) -> None:
    """ファイルをダウンロードして保存する"""
    logger.info(f"ダウンロード中: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"ダウンロード完了: {save_path}")

def download_and_extract_zip(url: str, extract_dir: str) -> None:
    """ZIPファイルをダウンロードして展開する"""
    logger.info(f"ZIPファイルをダウンロード中: {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info(f"ZIPファイルの展開完了: {extract_dir}")

def download_models() -> None:
    """必要なモデルをダウンロードする"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    try:
        if not os.path.exists(f"{MODELS_DIR}/ja.model"):
            download_file("https://huggingface.co/models/ja.model", f"{MODELS_DIR}/ja.model")
        if not os.path.exists(f"{MODELS_DIR}/en.model"):
            download_file("https://huggingface.co/models/en.model", f"{MODELS_DIR}/en.model")
        
        if not os.path.exists(f"{MODELS_DIR}/nllb-ja-en-q8_0"):
            os.makedirs(f"{MODELS_DIR}/nllb-ja-en-q8_0", exist_ok=True)
            download_and_extract_zip(
                "https://huggingface.co/models/nllb-200-distilled-600M-ja-en-q8_0.zip", 
                f"{MODELS_DIR}/nllb-ja-en-q8_0"
            )
        
        if not os.path.exists(f"{MODELS_DIR}/nllb-en-ja-q8_0"):
            os.makedirs(f"{MODELS_DIR}/nllb-en-ja-q8_0", exist_ok=True)
            download_and_extract_zip(
                "https://huggingface.co/models/nllb-200-distilled-600M-en-ja-q8_0.zip", 
                f"{MODELS_DIR}/nllb-en-ja-q8_0"
            )
        
        logger.info("すべてのモデルのダウンロードが完了しました")
        return True
    except Exception as e:
        logger.error(f"モデルのダウンロード中にエラーが発生しました: {e}")
        return False

if __name__ == "__main__":
    logger.info("モデルのダウンロードを開始します...")
    success = download_models()
    if success:
        logger.info("モデルのダウンロードが正常に完了しました")
        sys.exit(0)
    else:
        logger.error("モデルのダウンロードに失敗しました")
        sys.exit(1)
