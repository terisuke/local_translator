"""
Test script for the Translator component.
This script tests the translation of Japanese text to English.
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.translator import Translator

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_translation():
    """Test the translation of Japanese text to English."""
    japanese_text = "本日はご来場いただき誠にありがとうございます。 ご来場の目的をお伺いしてもよろしいでしょうか。"
    
    logger.info(f"Testing translation of: {japanese_text}")
    
    translator = Translator()
    
    translated_text, src_lang, tgt_lang = translator.translate(japanese_text, "ja")
    
    logger.info(f"Source language: {src_lang}")
    logger.info(f"Target language: {tgt_lang}")
    logger.info(f"Translation result: {translated_text}")
    
    print("\n" + "="*50)
    print(f"Source (ja): {japanese_text}")
    print(f"Translation (en): {translated_text}")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_translation()
