"""
Test script for the full Local Translator application with audio files.
This script simulates microphone input using a pre-recorded audio file.
"""

import os
import sys
import time
import argparse
import logging
import threading
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.app import TranslatorApp

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

def on_text(text, lang):
    """Callback for when text is recognized"""
    print(f"\n[SOURCE] ({lang}): {text}")

def on_translation(text, source_lang, target_lang):
    """Callback for when text is translated"""
    print(f"[TRANSLATION] ({target_lang}): {text}\n")

def process_audio_file(file_path, model_size="base", device="cpu", compute_type="float32"):
    """
    Process an audio file through the full TranslatorApp pipeline.
    
    Args:
        file_path: Path to the audio file
        model_size: Size of the Whisper model to use
        device: Device to use for inference
        compute_type: Computation type
    """
    audio_data, sample_rate = load_audio_file(file_path, target_sr=16000)
    
    logger.info(f"Initializing TranslatorApp with model_size={model_size}, device={device}")
    app = TranslatorApp(
        on_text=on_text,
        on_translation=on_translation,
        model_size=model_size,
        device=device,
        compute_type=compute_type
    )
    
    logger.info("Starting TranslatorApp")
    with app:
        chunk_size = int(sample_rate * 0.1)  # 100ms chunks
        num_chunks = len(audio_data) // chunk_size
        
        logger.info(f"Processing audio in {num_chunks} chunks of {chunk_size} samples")
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio_data[start:end]
            
            app.audio_processor.process_audio(chunk, is_speech=True)
            
            time.sleep(0.05)
            
            if i % 10 == 0:
                logger.info(f"Processing progress: {i}/{num_chunks} chunks ({i/num_chunks*100:.1f}%)")
        
        remaining = audio_data[num_chunks*chunk_size:]
        if len(remaining) > 0:
            app.audio_processor.process_audio(remaining, is_speech=True)
        
        logger.info("Waiting for processing to complete...")
        time.sleep(3)
        
        logger.info("Processing complete")

def main():
    parser = argparse.ArgumentParser(description="Test full application with audio file")
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
