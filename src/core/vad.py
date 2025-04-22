import numpy as np
import torch
from typing import Optional, Tuple
import os
import logging

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

        self.logger = logging.getLogger(__name__)

    def normalize_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        音声データを正規化する
        
        Args:
            audio_chunk: 音声データ
            
        Returns:
            np.ndarray: 正規化された音声データ
        """
        # DCオフセットを除去
        audio_chunk = audio_chunk - np.mean(audio_chunk)
        
        # 音量の正規化
        if np.abs(audio_chunk).max() > 0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()
        
        return audio_chunk

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
        audio_chunk = self.normalize_audio(audio_chunk)

        # ステレオの場合はモノラルに変換
        if len(audio_chunk.shape) > 1 and audio_chunk.shape[1] > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        # 次元の調整（1次元配列に変換）
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.flatten()

        # サンプル数を512に調整
        if len(audio_chunk) > self.window_size_samples:
            # オーバーラップして平均を取る
            n_chunks = len(audio_chunk) // self.window_size_samples
            chunks = [audio_chunk[i * self.window_size_samples:(i + 1) * self.window_size_samples] 
                     for i in range(n_chunks)]
            audio_chunk = np.mean(chunks, axis=0)
        elif len(audio_chunk) < self.window_size_samples:
            # 足りない部分を0で埋める
            audio_chunk = np.pad(audio_chunk, (0, self.window_size_samples - len(audio_chunk)))

        # モデル入力の準備（1次元→2次元）
        audio_tensor = torch.from_numpy(audio_chunk).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        speech_prob = self.model(audio_tensor, self.sample_rate).item()

        return speech_prob > 0.2

    def process_audio(self, audio_chunk: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        音声チャンクを処理し、音声部分を抽出
        
        Args:
            audio_chunk: 音声データ
            
        Returns:
            Tuple[np.ndarray, bool]: (処理済み音声データ, 音声を含むかどうか)
        """
        self.logger.debug(f"音声処理開始: shape={audio_chunk.shape}, dtype={audio_chunk.dtype}")
        
        # 音声の正規化処理を改善
        audio_chunk = self.normalize_audio(audio_chunk)
        self.logger.debug(f"正規化後: min={audio_chunk.min():.3f}, max={audio_chunk.max():.3f}, mean={audio_chunk.mean():.3f}")
        
        is_speech = self.is_speech(audio_chunk)
        self.logger.debug(f"音声検出結果: is_speech={is_speech}")
        
        return audio_chunk, is_speech 