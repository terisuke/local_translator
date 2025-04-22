import sounddevice as sd
import numpy as np
from typing import Callable, Optional
import queue
import threading
import time

class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        callback: Optional[Callable] = None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.callback = callback
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.stream = None
        self.thread = None
        self.buffer = np.array([], dtype=np.float32)

    def audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """音声ストリームのコールバック関数"""
        if status:
            print(f"音声入力エラー: {status}")
        
        # 音声データをバッファに追加
        self.buffer = np.concatenate([self.buffer, indata.flatten()])
        
        # バッファサイズが一定以上になったら処理
        if len(self.buffer) >= self.chunk_size * 4:
            self.audio_queue.put(self.buffer.copy())
            self.buffer = np.array([], dtype=np.float32)

    def process_audio(self):
        """音声データの処理ループ"""
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=1.0)
                if self.callback:
                    self.callback(audio_data)
            except queue.Empty:
                continue

    def start(self):
        """音声処理の開始"""
        if self.is_running:
            return

        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()
        self.thread = threading.Thread(target=self.process_audio)
        self.thread.start()

    def stop(self):
        """音声処理の停止"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.thread:
            self.thread.join()
        self.audio_queue.queue.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 