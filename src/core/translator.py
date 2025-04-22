import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import Optional
import langdetect

class Translator:
    def __init__(self, device: str = "cpu"):
        """
        翻訳モデルの初期化
        
        Args:
            device: 実行デバイス（"cpu" または "cuda"）
        """
        self.device = device
        
        # 英語→日本語のモデル
        self.en_ja_model = "Helsinki-NLP/opus-mt-en-jap"
        self.en_ja_tokenizer = MarianTokenizer.from_pretrained(self.en_ja_model)
        self.en_ja_model = MarianMTModel.from_pretrained(self.en_ja_model).to(device)
        
        # 日本語→英語のモデル
        self.ja_en_model = "Helsinki-NLP/opus-mt-jap-en"
        self.ja_en_tokenizer = MarianTokenizer.from_pretrained(self.ja_en_model)
        self.ja_en_model = MarianMTModel.from_pretrained(self.ja_en_model).to(device)

    def detect_language(self, text: str) -> str:
        """
        テキストの言語を検出
        
        Args:
            text: 入力テキスト
            
        Returns:
            str: 言語コード ("ja" or "en")
        """
        try:
            lang = langdetect.detect(text)
            return lang
        except:
            return "en"  # デフォルトは英語

    def translate(self, text: str, source_lang: str) -> str:
        """
        テキストを翻訳
        
        Args:
            text: 翻訳するテキスト
            source_lang: 入力言語（"en" または "ja"）
            
        Returns:
            str: 翻訳されたテキスト
        """
        if not text:
            return ""
            
        if source_lang == "en":
            # 英語→日本語
            tokenizer = self.en_ja_tokenizer
            model = self.en_ja_model
        else:
            # 日本語→英語
            tokenizer = self.ja_en_tokenizer
            model = self.ja_en_model
            
        # テキストのトークン化
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # 翻訳の生成
        with torch.no_grad():
            outputs = model.generate(**inputs)
            
        # トークンのデコード
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return translated 