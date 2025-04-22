from faster_whisper import WhisperModel
import numpy as np
from typing import Optional, Tuple, List
import torch
import logging

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
        self.logger = logging.getLogger(__name__)
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

    def transcribe_stream(self, audio_data: np.ndarray) -> Tuple[str, dict]:
        """
        音声ストリームを文字起こし
        
        Args:
            audio_data: 音声データ
            
        Returns:
            Tuple[str, dict]: (文字起こしテキスト, 追加情報)
        """
        try:
            # 音声データの前処理
            audio_data = audio_data.astype(np.float32)
            
            # 無音部分の除去
            if np.abs(audio_data).max() < 1e-3:
                return "", {"language": "en", "language_probability": 0.0}
            
            # DCオフセットの除去
            audio_data = audio_data - np.mean(audio_data)
            
            # 正規化
            max_abs = np.abs(audio_data).max()
            if max_abs > 0:
                audio_data = audio_data / max_abs
            
            # 音声認識の実行
            segments, info = self.model.transcribe(
                audio_data,
                language=self.language or "en",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300}
            )
            
            # 結果の取得
            text = " ".join([s.text for s in segments]).strip()
            return text, info
            
        except Exception as e:
            self.logger.error(f"音声認識中にエラーが発生: {e}")
            return "", {"language": "en", "language_probability": 0.0} 