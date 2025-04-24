import asyncio
import logging
import numpy as np
import os
import torch
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
import silero_vad
import ctranslate2
import sentencepiece as spm
import langdetect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

os.makedirs("models", exist_ok=True)

class ASRTranslationService:
    def __init__(self):
        logger.info("ASRTranslationServiceを初期化中...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False)
        self.vad_get_speech_timestamps = utils[0]
        self.vad_threshold = 0.5
        self.sample_rate = 16000
        
        self.asr = WhisperModel("small", device="cpu", compute_type="int8")
        
        self.ja_en_translator = ctranslate2.Translator("models/nllb-ja-en-q8_0", device="cpu")
        self.en_ja_translator = ctranslate2.Translator("models/nllb-en-ja-q8_0", device="cpu")
        self.ja_tokenizer = spm.SentencePieceProcessor()
        self.en_tokenizer = spm.SentencePieceProcessor()
        self.ja_tokenizer.Load("models/ja.model")
        self.en_tokenizer.Load("models/en.model")
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
