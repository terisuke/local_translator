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
import ctranslate2
import sentencepiece as spm
import langdetect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

os.makedirs("models", exist_ok=True)

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
    try:
        if not os.path.exists("models/ja.model"):
            download_file("https://huggingface.co/models/ja.model", "models/ja.model")
        if not os.path.exists("models/en.model"):
            download_file("https://huggingface.co/models/en.model", "models/en.model")
        
        if not os.path.exists("models/nllb-ja-en-q8_0"):
            os.makedirs("models/nllb-ja-en-q8_0", exist_ok=True)
            download_and_extract_zip(
                "https://huggingface.co/models/nllb-200-distilled-600M-ja-en-q8_0.zip", 
                "models/nllb-ja-en-q8_0"
            )
        
        if not os.path.exists("models/nllb-en-ja-q8_0"):
            os.makedirs("models/nllb-en-ja-q8_0", exist_ok=True)
            download_and_extract_zip(
                "https://huggingface.co/models/nllb-200-distilled-600M-en-ja-q8_0.zip", 
                "models/nllb-en-ja-q8_0"
            )
        
        return True
    except Exception as e:
        logger.error(f"モデルのダウンロード中にエラーが発生しました: {e}")
        return False

class ASRTranslationService:
    def __init__(self):
        logger.info("ASRTranslationServiceを初期化中...")
        
        if not os.path.exists("models/nllb-ja-en-q8_0/model.bin") or \
           not os.path.exists("models/nllb-en-ja-q8_0/model.bin") or \
           not os.path.exists("models/ja.model") or \
           not os.path.exists("models/en.model"):
            logger.warning("必要なモデルファイルが見つかりません。自動ダウンロードを試みます...")
            if not download_models():
                logger.error("モデルのダウンロードに失敗しました。")
                raise RuntimeError("必要なモデルファイルが見つかりません。")
        
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False)
        self.vad_get_speech_timestamps = utils[0]
        self.vad_threshold = 0.5
        self.sample_rate = 16000
        
        self.asr = WhisperModel("small", device="cpu", compute_type="int8")
        
        try:
            self.ja_en_translator = ctranslate2.Translator("models/nllb-ja-en-q8_0", device="cpu")
            self.en_ja_translator = ctranslate2.Translator("models/nllb-en-ja-q8_0", device="cpu")
            self.ja_tokenizer = spm.SentencePieceProcessor()
            self.en_tokenizer = spm.SentencePieceProcessor()
            self.ja_tokenizer.Load("models/ja.model")
            self.en_tokenizer.Load("models/en.model")
            logger.info("すべてのモデルを読み込みました")
        except Exception as e:
            logger.error(f"モデルの読み込み中にエラーが発生しました: {e}")
            raise

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
            tokens = self.ja_tokenizer.encode(text, out_type=str)
            translated_tokens = self.ja_en_translator.translate_batch([tokens])[0][0]
            translated_text = self.en_tokenizer.decode(translated_tokens)
            target_lang = 'en'
        else:
            tokens = self.en_tokenizer.encode(text, out_type=str)
            translated_tokens = self.en_ja_translator.translate_batch([tokens])[0][0]
            translated_text = self.ja_tokenizer.decode(translated_tokens)
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
