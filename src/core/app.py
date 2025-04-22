import numpy as np
from typing import Optional, Callable
import threading
import queue
import time
from datetime import datetime
import os
import logging

try:
    import tkinter as tk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    tk = None

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
        # ロギングの設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
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
        
        # GUIのルートウィンドウ
        self.root: Optional[tk.Tk] = None

    def set_root(self, root: tk.Tk):
        """GUIのルートウィンドウを設定"""
        self.root = root

    def call_callback(self, callback: Callable, *args, **kwargs):
        """コールバック関数を安全に呼び出す"""
        try:
            self.logger.debug(f"コールバック呼び出し: {callback.__name__}, 引数: {args}, {kwargs}")
            if HAS_TKINTER and self.root and callback:
                # GUIのルートウィンドウを使ってメインスレッドからコールバックを実行
                self.logger.debug(f"ルートウィンドウを使ってコールバックを実行")
                self.root.after(0, lambda: callback(*args, **kwargs))
            elif callback:
                self.logger.debug(f"直接コールバックを実行")
                callback(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"コールバックの呼び出しに失敗: {e}")

    def process_audio(self, audio_data: np.ndarray):
        """
        音声データの処理
        
        Args:
            audio_data: 音声データ
        """
        try:
            # バッファサイズを調整（0.5秒分のデータ）
            buffer_size = int(16000 * 0.5)  # サンプリングレート * 秒数
            if len(audio_data) < buffer_size:
                return
            
            # 音声データの正規化
            audio_data = audio_data.astype(np.float32)
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # VADによる音声検出
            self.logger.debug(f"VAD処理開始: audio_data shape={audio_data.shape}")
            processed_audio, is_speech = self.vad.process_audio(audio_data)
            self.logger.debug(f"VAD処理結果: is_speech={is_speech}")
            
            if is_speech:
                self.logger.info("音声を検出しました")
                # ASRによる音声認識
                self.logger.debug("音声認識開始")
                text, info = self.asr.transcribe_stream(processed_audio)
                self.logger.debug(f"音声認識結果: text='{text}', info={info}")
                
                if text.strip():
                    self.logger.info(f"認識結果: {text}")
                    # テキストキューに追加
                    self.text_queue.put((text, info["language"]))
                    
                    # コールバック呼び出し
                    if self.on_text:
                        self.logger.debug(f"on_textコールバックを呼び出し: text='{text}', lang={info['language']}")
                        self.call_callback(self.on_text, text, info["language"])
                    else:
                        self.logger.warning("on_textコールバックが設定されていません")
                else:
                    self.logger.debug("認識テキストが空のため処理をスキップ")
            else:
                self.logger.debug("音声が検出されませんでした")
        except Exception as e:
            self.logger.error(f"音声処理中にエラーが発生しました: {e}")

    def process_text(self):
        """
        テキスト処理ループ
        """
        self.logger.info("テキスト処理スレッドを開始します")
        while self.is_running:
            try:
                text, source_lang = self.text_queue.get(timeout=1.0)
                self.logger.info(f"翻訳を開始します: {text}")
                
                # 翻訳の実行
                translated, src, tgt = self.translator.translate(
                    text,
                    source_lang=source_lang
                )
                
                self.logger.info(f"翻訳結果: {translated}")
                
                # 翻訳キューに追加
                self.translation_queue.put((translated, tgt))
                
                # コールバック呼び出し
                if self.on_translation:
                    self.call_callback(self.on_translation, translated, tgt)
                    
                # ログの保存
                self.save_log(text, translated, src, tgt)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"テキスト処理中にエラーが発生しました: {e}")

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
        self.logger.info("アプリケーションを開始します")
        
        # 音声処理の開始
        self.audio_processor.start()
        
        # テキスト処理スレッドの開始
        self.text_thread = threading.Thread(target=self.process_text, daemon=True)
        self.text_thread.start()

    def stop(self):
        """アプリケーションの停止"""
        if not self.is_running:
            return
            
        self.logger.info("アプリケーションを停止します")
        self.is_running = False
        
        # 音声処理の停止
        self.audio_processor.stop()
        
        # テキスト処理スレッドの停止
        if hasattr(self, "text_thread"):
            self.text_thread.join(timeout=5.0)
            
        # キューのクリア
        self.text_queue.queue.clear()
        self.translation_queue.queue.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()      