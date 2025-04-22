import tkinter as tk
import argparse
import sys
import os

# パスの設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.main_window import MainWindow

def main():
    """メインエントリーポイント"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="リアルタイム日英翻訳アプリ")
    parser.add_argument("--cli", action="store_true", help="CLIモードで起動")
    args = parser.parse_args()
    
    if args.cli:
        # CLIモード
        from core.app import TranslatorApp
        
        app = TranslatorApp()
        app.on_text = lambda text, lang: print(f"原文 [{lang}]: {text}")
        app.on_translation = lambda text, lang: print(f"訳文 [{lang}]: {text}")
        
        print("翻訳を開始します。Ctrl+Cで終了。")
        try:
            app.start()
            while True:
                input()
        except KeyboardInterrupt:
            print("\n翻訳を終了します。")
            app.stop()
    else:
        # GUIモード
        root = tk.Tk()
        app = MainWindow(root)
        root.mainloop()

if __name__ == "__main__":
    main() 