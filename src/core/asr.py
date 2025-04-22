from faster_whisper import WhisperModel
import numpy as np
from typing import Optional, Tuple, List
import torch

class ASR:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None
    ):
        """
        Whisperモデルの初期化
        
        Args:
            model_size: モデルサイズ ("tiny", "base", "small", "medium", "large")
            device: 実行デバイス ("cpu" or "cuda")
            compute_type: 計算精度 ("int8", "float16", "float32")
            language: 言語コード（Noneの場合は自動検出）
        """
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root="./models"
        )
        self.language = language
        self.sample_rate = 16000

    def transcribe(
        self,
        audio: np.ndarray,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None
    ) -> Tuple[str, dict]:
        """
        音声をテキストに変換
        
        Args:
            audio: 音声データ
            beam_size: ビームサーチのサイズ
            vad_filter: VADフィルタを使用するかどうか
            vad_parameters: VADパラメータ
            
        Returns:
            Tuple[str, dict]: (転写テキスト, メタデータ)
        """
        segments, info = self.model.transcribe(
            audio,
            beam_size=beam_size,
            language=self.language,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters
        )

        # セグメントを結合
        text = " ".join([segment.text for segment in segments])
        
        return text, {
            "language": info.language,
            "language_probability": info.language_probability
        }

    def transcribe_stream(
        self,
        audio_chunk: np.ndarray,
        beam_size: int = 5
    ) -> Tuple[str, dict]:
        """
        ストリーミング音声をテキストに変換
        
        Args:
            audio_chunk: 音声チャンク
            beam_size: ビームサーチのサイズ
            
        Returns:
            Tuple[str, dict]: (転写テキスト, メタデータ)
        """
        return self.transcribe(audio_chunk, beam_size=beam_size) 