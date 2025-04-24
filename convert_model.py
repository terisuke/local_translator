import ctranslate2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

def convert_model():
    # モデルとトークナイザーをダウンロード
    model_name = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 日本語から英語への変換モデルを作成
    os.makedirs("models/nllb-ja-en-q8_0", exist_ok=True)
    converter = ctranslate2.converters.TransformersConverter(model.config)
    converter.convert(
        "models/nllb-ja-en-q8_0",
        model,
        quantization="int8",
        force=True
    )

    # 英語から日本語への変換モデルを作成
    os.makedirs("models/nllb-en-ja-q8_0", exist_ok=True)
    converter.convert(
        "models/nllb-en-ja-q8_0",
        model,
        quantization="int8",
        force=True
    )

    # トークナイザーモデルをコピー
    tokenizer.save_pretrained("models")
    os.rename("models/sentencepiece.bpe.model", "models/ja.model")
    os.system("cp models/ja.model models/en.model")

if __name__ == "__main__":
    convert_model() 