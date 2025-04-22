import torch
import os

def download_model():
    torch.hub.set_dir(os.path.dirname(os.path.abspath(__file__)))
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True,
                                trust_repo=True)
    
    # モデルをONNX形式でエクスポート
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "silero_vad.onnx")
    
    dummy_input = torch.randn(1, 1500)
    torch.onnx.export(model,
                     dummy_input,
                     model_path,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {1: 'length'}})
    
    print(f"モデルを保存しました: {model_path}")

if __name__ == "__main__":
    download_model() 