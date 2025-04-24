import asyncio
import logging
import numpy as np
import os
import torch
import sys
import requests
import zipfile
import io
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from faster_whisper import WhisperModel
import silero_vad
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import langdetect
from dotenv import load_dotenv

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数を読み込む
load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
logger.info(f"Hugging Face Token: {'設定されています' if token else '設定されていません'}")

app = FastAPI()

os.makedirs("models", exist_ok=True)

def download_file(url: str, save_path: str, headers: dict = None) -> None:
    """ファイルをダウンロードして保存する"""
    logger.info(f"ダウンロード中: {url}")
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"ダウンロード完了: {save_path}")

def download_and_extract_zip(url: str, extract_dir: str, headers: dict = None) -> None:
    """ZIPファイルをダウンロードして展開する"""
    logger.info(f"ZIPファイルをダウンロード中: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info(f"ZIPファイルの展開完了: {extract_dir}")

def download_models() -> None:
    """必要なモデルをダウンロードする"""
    try:
        # Hugging Faceのトークンを取得
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise ValueError("HUGGING_FACE_HUB_TOKENが設定されていません。")

        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
        }

        if not os.path.exists("models/ja.model"):
            download_file(
                "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece.bpe.model",
                "models/ja.model",
                headers=headers
            )
        if not os.path.exists("models/en.model"):
            download_file(
                "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece.bpe.model",
                "models/en.model",
                headers=headers
            )
        
        if not os.path.exists("models/nllb-ja-en-q8_0"):
            os.makedirs("models/nllb-ja-en-q8_0", exist_ok=True)
            download_and_extract_zip(
                "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/pytorch_model.bin",
                "models/nllb-ja-en-q8_0",
                headers=headers
            )
        
        if not os.path.exists("models/nllb-en-ja-q8_0"):
            os.makedirs("models/nllb-en-ja-q8_0", exist_ok=True)
            download_and_extract_zip(
                "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/pytorch_model.bin",
                "models/nllb-en-ja-q8_0",
                headers=headers
            )
        
        return True
    except Exception as e:
        logger.error(f"モデルのダウンロード中にエラーが発生しました: {e}")
        return False

class ASRTranslationService:
    def __init__(self):
        logger.info("ASRTranslationServiceを初期化中...")
        
        # Whisperモデルの初期化
        self.asr = WhisperModel("small", device="cpu", compute_type="int8")
        
        # VADモデルの初期化
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False)
        self.vad_get_speech_timestamps = utils[0]
        self.vad_threshold = 0.5
        self.sample_rate = 16000
        
        # NLLBモデルの初期化
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.model.eval()
        logger.info("すべてのモデルを読み込みました")

    async def detect_speech(self, audio_data: np.ndarray) -> bool:
        tensor = torch.from_numpy(audio_data).float()
        speech_timestamps = self.vad_get_speech_timestamps(
            tensor, 
            self.vad_model, 
            threshold=self.vad_threshold,
            sampling_rate=self.sample_rate
        )
        return len(speech_timestamps) > 0

    async def process_audio(self, audio_data: np.ndarray):
        segments, info = self.asr.transcribe(audio_data, beam_size=5)
        segments_list = list(segments)  # イテレータをリストに変換
        
        if not segments_list:
            return {
                "original": {"text": "", "lang": "unknown"},
                "translated": {"text": "", "lang": "unknown"}
            }
        
        text = " ".join([segment.text for segment in segments_list])
        detected_lang = info.language
        
        try:
            if text.strip():
                lang_detect_result = langdetect.detect(text)
                if lang_detect_result in ['ja', 'jp']:
                    lang = 'ja'
                else:
                    lang = 'en'
            else:
                lang = detected_lang if detected_lang in ['ja', 'en'] else 'en'
        except:
            lang = detected_lang if detected_lang in ['ja', 'en'] else 'en'
        
        logger.info(f"認識結果: '{text}' (言語: {lang})")
        
        if lang == 'ja':
            # 日本語から英語への翻訳
            inputs = self.tokenizer(text, return_tensors="pt")
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["eng_Latn"],
                max_length=128
            )
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            target_lang = 'en'
        else:
            # 英語から日本語への翻訳
            inputs = self.tokenizer(text, return_tensors="pt")
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["jpn_Jpan"],
                max_length=128
            )
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            target_lang = 'ja'
            
        logger.info(f"翻訳結果: '{translated_text}' (言語: {target_lang})")
        return {
            "original": {"text": text, "lang": lang},
            "translated": {"text": translated_text, "lang": target_lang}
        }

service = ASRTranslationService()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = np.array([], dtype=np.float32)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            audio_buffer = np.concatenate([audio_buffer, audio_chunk])
            
            if len(audio_buffer) >= 16000 * 0.5:  # 0.5秒分のデータ
                is_speech = await service.detect_speech(audio_buffer)
                
                if is_speech:
                    result = await service.process_audio(audio_buffer)
                    await websocket.send_json(result)
                
                audio_buffer = np.array([], dtype=np.float32)
                
    except WebSocketDisconnect:
        logger.info("クライアントが切断しました")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get():
    with open("static/index.html") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
