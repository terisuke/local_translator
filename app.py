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
import time
import re

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
        
        # バッファ設定
        self.min_text_length = 100  # 最小テキスト長を増加
        self.silence_threshold = 2.0  # 無音検出の閾値（秒）
        self.min_buffer_size = 16000 * 2  # 最小バッファサイズ（2秒）
        self.text_buffer = ""
        self.last_speech_time = 0
        
        # 文脈管理
        self.conversation_history = []
        self.max_history_chars = 2000  # 履歴保持文字数を増加
        self.pending_translation = None  # 翻訳待ちのテキスト
        self.last_translation_time = 0
        self.translation_delay = 3.0  # 翻訳を待機する時間（秒）
        
        # 文章結合用の正規表現パターン
        self.sentence_end_pattern = re.compile(r'[.。!！?？]\s*')
        
        # NLLBモデルの初期化
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.model.eval()
        logger.info("すべてのモデルを読み込みました")

    def should_merge_with_previous(self, current_text: str, previous_text: str) -> bool:
        """前のテキストと結合すべきかを判断"""
        if not previous_text or not current_text:
            return False
            
        # 文末で終わっていない場合は結合
        if not self.sentence_end_pattern.search(previous_text):
            return True
            
        # 小文字で始まる場合は前の文の続きの可能性が高い
        if current_text[0].islower():
            return True
            
        # 接続詞で始まる場合は結合
        connecting_words = {'and', 'or', 'but', 'so', 'because', 'however', 'therefore', 'then', 'also'}
        first_word = current_text.split()[0].lower() if current_text.split() else ''
        if first_word in connecting_words:
            return True
            
        return False

    def merge_text_segments(self, new_text: str) -> str:
        """新しいテキストを既存の文脈と結合"""
        if not self.conversation_history:
            return new_text
            
        previous_text = self.conversation_history[-1]["text"]
        if self.should_merge_with_previous(new_text, previous_text):
            # 前のテキストと結合
            merged_text = previous_text.rstrip() + " " + new_text.lstrip()
            self.conversation_history.pop()  # 前のテキストを削除
            return merged_text
            
        return new_text

    def update_conversation_history(self, text: str, lang: str):
        """会話履歴を更新"""
        self.conversation_history.append({
            "text": text,
            "lang": lang,
            "timestamp": time.time()
        })
        
        # 古い履歴を削除
        total_chars = sum(len(msg["text"]) for msg in self.conversation_history)
        while total_chars > self.max_history_chars and self.conversation_history:
            removed = self.conversation_history.pop(0)
            total_chars -= len(removed["text"])

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
        current_time = time.time()
        segments, info = self.asr.transcribe(audio_data, beam_size=5)
        segments_list = list(segments)
        
        if not segments_list:
            # 無音期間が十分長く、保留中の翻訳がある場合
            if (current_time - self.last_speech_time > self.silence_threshold and 
                self.pending_translation):
                result = await self.process_buffered_text(force_translate=True)
                self.pending_translation = None
                return result
            return None
        
        text = " ".join([segment.text for segment in segments_list])
        
        # 言語検出
        has_japanese = any(ord(char) > 0x3040 for char in text)
        lang = 'ja' if has_japanese else 'en'
        
        # テキストの結合
        text = self.merge_text_segments(text)
        
        # 保留中の翻訳がある場合は結合を試みる
        if self.pending_translation:
            if self.should_merge_with_previous(text, self.pending_translation):
                text = self.pending_translation + " " + text
            else:
                # 保留中の翻訳を処理
                result = await self.process_buffered_text(force_translate=True)
                self.pending_translation = text
                return result
        else:
            self.pending_translation = text
        
        self.last_speech_time = current_time
        
        # 以下の条件で翻訳を実行:
        # 1. テキストが十分な長さになった
        # 2. 文末記号で終わっている
        # 3. 最後の翻訳から十分な時間が経過
        if (len(self.pending_translation) >= self.min_text_length or
            self.sentence_end_pattern.search(self.pending_translation) or
            current_time - self.last_translation_time >= self.translation_delay):
            
            result = await self.process_buffered_text(force_translate=True)
            self.pending_translation = None
            self.last_translation_time = current_time
            return result
        
        return None

    async def process_buffered_text(self, force_translate: bool = False):
        text = self.pending_translation
        if not text or not text.strip():
            return None
            
        # 言語検出
        has_japanese = any(ord(char) > 0x3040 for char in text)
        lang = 'ja' if has_japanese else 'en'
        
        # 文脈を考慮した翻訳のために履歴を更新
        self.update_conversation_history(text, lang)
        
        logger.info(f"認識結果: '{text}' (言語: {lang})")
        
        if lang == 'ja':
            # 日本語から英語への翻訳
            inputs = self.tokenizer(text, return_tensors="pt")
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=512,  # より長い文章に対応
                num_beams=5,
                length_penalty=0.6,
                repetition_penalty=1.2  # 繰り返しを防ぐ
            )
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            target_lang = 'en'
        else:
            # 英語から日本語への翻訳
            inputs = self.tokenizer(text, return_tensors="pt")
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("jpn_Jpan"),
                max_length=512,
                num_beams=5,
                length_penalty=0.6,
                repetition_penalty=1.2
            )
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            target_lang = 'ja'
            
        logger.info(f"翻訳結果: '{translated_text}' (言語: {target_lang})")
        return {
            "original": {"text": text.strip(), "lang": lang},
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
            
            # 音声バッファに追加
            audio_buffer = np.concatenate([audio_buffer, audio_chunk])
            
            # バッファが2秒分以上たまったら処理
            if len(audio_buffer) >= service.min_buffer_size:
                is_speech = await service.detect_speech(audio_buffer)
                
                if is_speech:
                    result = await service.process_audio(audio_buffer)
                    if result:  # Noneでない場合のみ送信
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
