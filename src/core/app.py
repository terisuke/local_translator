import numpy as np
from typing import Optional, Callable
import threading
import queue
import time
from datetime import datetime
import os

from .audio_processor import AudioProcessor
from .vad import VAD
from .asr import ASR
from .translator import Translator

class TranslatorApp:
    def __init__(
        self,
        device: str = "cpu",
        model_size: str = "small",
        compute_type: str = "int8",
        log_dir: str = "logs"
    ):
        """
        翻訳アプリケーションの初期化
        
        Args:
            device: 実行デバイス ("cpu" or "cuda")
            model_size: Whisperモデルサイズ
            compute_type: 計算精度
            log_dir: ログ保存ディレクトリ
        """
        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type
        self.log_dir = log_dir
        
        # コンポーネントの初期化
        self.vad = VAD()
        self.asr = ASR(
            model_size=model_size,
            device=device,
            compute_type=compute_type
        )
        self.translator = Translator(device=device)
        
        # 音声処理の初期化
        self.audio_processor = AudioProcessor(callback=self.process_audio)
        
        # 状態管理
        self.is_running = False
        self.text_queue = queue.Queue()
        self.translation_queue = queue.Queue()
        
        # ログディレクトリの作成
        os.makedirs(log_dir, exist_ok=True)
        
        # コールバック
        self.on_text: Optional[Callable[[str, str], None]] = None
        self.on_translation: Optional[Callable[[str, str], None]] = None

    def process_audio(self, audio_data: np.ndarray):
        """
        音声データの処理
        
        Args:
            audio_data: 音声データ
        """
        # VADによる音声検出
        processed_audio, is_speech = self.vad.process_audio(audio_data)
        
        if is_speech:
            # ASRによる音声認識
            text, info = self.asr.transcribe_stream(processed_audio)
            
            if text.strip():
                # テキストキューに追加
                self.text_queue.put((text, info["language"]))
                
                # コールバック呼び出し
                if self.on_text:
                    self.on_text(text, info["language"])

    def process_text(self):
        """
        テキスト処理ループ
        """
        while self.is_running:
            try:
                text, source_lang = self.text_queue.get(timeout=1.0)
                
                # 翻訳の実行
                translated, src, tgt = self.translator.translate(
                    text,
                    source_lang=source_lang
                )
                
                # 翻訳キューに追加
                self.translation_queue.put((translated, tgt))
                
                # コールバック呼び出し
                if self.on_translation:
                    self.on_translation(translated, tgt)
                    
                # ログの保存
                self.save_log(text, translated, src, tgt)
                
            except queue.Empty:
                continue

    def save_log(self, text: str, translated: str, src: str, tgt: str):
        """
        翻訳ログの保存
        
        Args:
            text: 原文
            translated: 訳文
            src: ソース言語
            tgt: ターゲット言語
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"translation_{timestamp}.txt")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {src}→{tgt}\n")
            f.write(f"原文: {text}\n")
            f.write(f"訳文: {translated}\n")
            f.write("-" * 50 + "\n")

    def start(self):
        """アプリケーションの開始"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # 音声処理の開始
        self.audio_processor.start()
        
        # テキスト処理スレッドの開始
        self.text_thread = threading.Thread(target=self.process_text)
        self.text_thread.start()

    def stop(self):
        """アプリケーションの停止"""
        self.is_running = False
        
        # 音声処理の停止
        self.audio_processor.stop()
        
        # テキスト処理スレッドの停止
        if hasattr(self, "text_thread"):
            self.text_thread.join()
            
        # キューのクリア
        self.text_queue.queue.clear()
        self.translation_queue.queue.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 