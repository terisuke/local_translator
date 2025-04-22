"""
Test script for processing audio files with the Local Translator.
This script allows testing the ASR and translation components with audio files
instead of microphone input.
"""

import os
import sys
import argparse
import logging
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.asr import ASR
from core.translator import Translator

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_audio_file(file_path, target_sr=16000):
    """
    Load an audio file and convert it to the target sample rate.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (default: 16000 Hz)
        
    Returns:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
    """
    logger.info(f"Loading audio file: {file_path}")
    try:
        import librosa
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        
        logger.info(f"Original audio: {len(audio_data)/sample_rate:.2f} seconds, {sample_rate} Hz")
        
        if len(audio_data.shape) > 1:
            logger.info(f"Converting stereo to mono, shape: {audio_data.shape}")
            audio_data = audio_data.mean(axis=1)
        
        if sample_rate != target_sr:
            logger.info(f"Resampling from {sample_rate} Hz to {target_sr} Hz")
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=target_sr
            )
            sample_rate = target_sr
            
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        logger.info(f"Audio loaded successfully: {len(audio_data)/sample_rate:.2f} seconds, {sample_rate} Hz")
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        sys.exit(1)

def process_audio_file(file_path, model_size="base", device="cpu", compute_type="float32"):
    """
    Process an audio file through the ASR and translation pipeline.
    
    Args:
        file_path: Path to the audio file
        model_size: Size of the Whisper model to use
        device: Device to use for inference (cpu or cuda)
        compute_type: Computation type (float32, float16, int8)
    """
    audio_data, sample_rate = load_audio_file(file_path)
    
    logger.info(f"Initializing ASR with model_size={model_size}, device={device}, compute_type={compute_type}")
    asr = ASR(model_size=model_size, device=device, compute_type=compute_type)
    
    logger.info("Initializing Translator")
    translator = Translator()
    
    logger.info("Processing audio with ASR")
    text, source_lang = asr.transcribe_stream(audio_data)
    logger.info(f"ASR result: '{text}' (detected language: {source_lang})")
    
    if not text:
        logger.warning("No text was transcribed from the audio")
        return
    
    logger.info(f"Translating text from {source_lang}")
    translated_text, src_lang, tgt_lang = translator.translate(text, source_lang)
    logger.info(f"Translation result: '{translated_text}' (from {src_lang} to {tgt_lang})")
    
    print("\n" + "="*50)
    print(f"Source ({src_lang}): {text}")
    print(f"Translation ({tgt_lang}): {translated_text}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Test audio file processing with Local Translator")
    parser.add_argument("--file", "-f", type=str, help="Path to audio file", 
                        default=str(Path(__file__).parents[1] / "resources" / "sample.m4a"))
    parser.add_argument("--model", "-m", type=str, default="small", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--device", "-d", type=str, default="cpu", 
                        choices=["cpu", "cuda", "auto"],
                        help="Device to use for inference")
    parser.add_argument("--compute_type", "-c", type=str, default="float32",
                        choices=["float32", "float16", "int8"],
                        help="Computation type")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
    
    process_audio_file(
        args.file,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type
    )

if __name__ == "__main__":
    main()
