import argparse
import sys
import os

# パスの設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tkinter as tk
    from gui.main_window import MainWindow
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

def main():
    """メインエントリーポイント"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="リアルタイム日英翻訳アプリ")
    parser.add_argument("--cli", action="store_true", help="CLIモードで起動")
    args = parser.parse_args()
    
    if args.cli or not HAS_TKINTER:
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
    elif HAS_TKINTER:
        # GUIモード
        root = tk.Tk()
        app = MainWindow(root)
        root.mainloop()
    else:
        print("tkinterが利用できないため、GUIモードを起動できません。--cliオプションを使用してください。")
        sys.exit(1)

if __name__ == "__main__":
    main()    