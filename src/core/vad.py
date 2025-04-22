import numpy as np
import torch
from typing import Optional, Tuple
import os

class VAD:
    def __init__(self):
        """Silero VADモデルの初期化"""
        self.sample_rate = 16000
        self.window_size_samples = 512  # 32ms at 16kHz
        self.speech_pad_ms = 100
        self.speech_pad_samples = int(self.speech_pad_ms / 1000 * self.sample_rate)
        
        # モデルの初期化
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                         model='silero_vad',
                                         force_reload=True,
                                         trust_repo=True)
        self.model.eval()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        音声チャンクが音声を含むかどうかを判定
        
        Args:
            audio_chunk: 音声データ（numpy配列）
            
        Returns:
            bool: 音声を含む場合はTrue
        """
        if len(audio_chunk) < self.window_size_samples:
            return False

        # 音声データの正規化
        audio_chunk = audio_chunk.astype(np.float32)
        if audio_chunk.max() > 1.0 or audio_chunk.min() < -1.0:
            audio_chunk = audio_chunk / max(abs(audio_chunk.max()), abs(audio_chunk.min()))

        # モデル入力の準備
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)
        speech_prob = self.model(audio_tensor, self.sample_rate).item()

        return speech_prob > 0.5

    def process_audio(self, audio_chunk: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        音声チャンクを処理し、音声部分を抽出
        
        Args:
            audio_chunk: 音声データ
            
        Returns:
            Tuple[np.ndarray, bool]: (処理済み音声データ, 音声を含むかどうか)
        """
        is_speech = self.is_speech(audio_chunk)
        return audio_chunk, is_speech 