import tkinter as tk
from tkinter import ttk
import threading
from typing import Optional
import json
import os
import logging

from core.app import TranslatorApp

class MainWindow:
    def __init__(self, root: tk.Tk):
        """
        メインウィンドウの初期化
        
        Args:
            root: Tkinterのルートウィンドウ
        """
        # ロギングの設定
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        self.root = root
        self.root.title("リアルタイム日英翻訳")
        self.root.geometry("800x600")
        
        # 設定の読み込み
        self.config = self.load_config()
        
        # アプリケーションの初期化
        self.app = TranslatorApp(
            device=self.config.get("device", "cpu"),
            model_size=self.config.get("model_size", "small"),
            compute_type=self.config.get("compute_type", "int8")
        )
        self.app.set_root(self.root)  # ルートウィンドウを設定
        
        # GUIの初期化
        self.setup_ui()
        
        # コールバックの設定
        self.app.on_text = self.on_text
        self.app.on_translation = self.on_translation
        
        # 状態管理
        self.is_running = False
        
        # 定期的な更新処理の開始
        self.root.after(100, self.periodic_update)

    def periodic_update(self):
        """定期的な更新処理"""
        try:
            # GUIの状態を確認
            self.logger.debug("GUI更新チェック - テキストエリアの状態:")
            self.logger.debug(f"原文: {self.source_text.get('1.0', tk.END).strip()}")
            self.logger.debug(f"訳文: {self.target_text.get('1.0', tk.END).strip()}")
            
            # 次の更新をスケジュール
            self.root.after(100, self.periodic_update)
        except Exception as e:
            self.logger.error(f"GUI更新チェック中にエラーが発生: {e}")

    def setup_ui(self):
        """UIの初期化"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 原文表示エリア
        ttk.Label(main_frame, text="原文:").grid(row=0, column=0, sticky=tk.W)
        self.source_text = tk.Text(main_frame, height=10, width=80, wrap=tk.WORD)
        self.source_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 訳文表示エリア
        ttk.Label(main_frame, text="訳文:").grid(row=2, column=0, sticky=tk.W)
        self.target_text = tk.Text(main_frame, height=10, width=80, wrap=tk.WORD)
        self.target_text.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # スクロールバーの追加
        source_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.source_text.yview)
        source_scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))
        self.source_text.configure(yscrollcommand=source_scrollbar.set)
        
        target_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.target_text.yview)
        target_scrollbar.grid(row=3, column=2, sticky=(tk.N, tk.S))
        self.target_text.configure(yscrollcommand=target_scrollbar.set)
        
        # 制御ボタン
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="開始", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.state(["disabled"])
        
        self.settings_button = ttk.Button(button_frame, text="設定", command=self.show_settings)
        self.settings_button.pack(side=tk.LEFT, padx=5)
        
        # ステータスバー
        self.status_var = tk.StringVar(value="停止中...")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # グリッドの設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

    def update_text(self, widget: tk.Text, text: str, lang: str):
        """
        テキストウィジェットの更新（スレッドセーフ）
        
        Args:
            widget: 更新するテキストウィジェット
            text: 表示するテキスト
            lang: 言語コード
        """
        try:
            self.logger.debug(f"テキスト更新開始: widget={widget}, text={text}, lang={lang}")
            
            if threading.current_thread() is not threading.main_thread():
                self.logger.debug("メインスレッド以外から呼び出されたため、after()を使用")
                self.root.after(0, lambda: self.update_text(widget, text, lang))
                return
                
            # 既存のテキストをクリア
            widget.delete("1.0", tk.END)
            
            # 新しいテキストを挿入
            display_text = f"[{lang}] {text}"
            widget.insert("1.0", display_text)
            
            # スクロールを最新位置に
            widget.see(tk.END)
            
            self.logger.debug(f"テキスト更新完了: {display_text}")
            
            # 更新を強制
            widget.update_idletasks()
            
        except Exception as e:
            self.logger.error(f"テキスト更新中にエラーが発生: {e}")
            self.status_var.set(f"エラー: テキスト更新に失敗しました - {str(e)}")

    def on_text(self, text: str, lang: str):
        """
        テキスト認識時のコールバック
        
        Args:
            text: 認識されたテキスト
            lang: 言語コード
        """
        self.logger.debug(f"音声認識コールバック: text={text}, lang={lang}")
        try:
            self.update_text(self.source_text, text, lang)
            self.logger.debug("音声認識テキストの更新完了")
        except Exception as e:
            self.logger.error(f"音声認識コールバックでエラー: {e}")

    def on_translation(self, text: str, lang: str):
        """
        翻訳完了時のコールバック
        
        Args:
            text: 翻訳されたテキスト
            lang: 言語コード
        """
        self.logger.debug(f"翻訳コールバック: {text} ({lang})")
        self.root.after(0, lambda: self.update_text(self.target_text, text, lang))

    def start(self):
        """翻訳の開始"""
        if not self.is_running:
            self.is_running = True
            self.start_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
            self.status_var.set("翻訳中...")
            
            # テキストエリアのクリア
            self.source_text.delete("1.0", tk.END)
            self.target_text.delete("1.0", tk.END)
            
            # アプリケーションの開始
            self.app.start()

    def stop(self):
        """翻訳の停止"""
        if self.is_running:
            self.is_running = False
            self.start_button.state(["!disabled"])
            self.stop_button.state(["disabled"])
            self.status_var.set("停止中...")
            
            # アプリケーションの停止
            self.app.stop()

    def show_settings(self):
        """設定ダイアログの表示"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("設定")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # デバイス選択
        ttk.Label(settings_window, text="実行デバイス:").grid(row=0, column=0, padx=5, pady=5)
        device_var = tk.StringVar(value=self.config.get("device", "cpu"))
        device_combo = ttk.Combobox(settings_window, textvariable=device_var)
        device_combo["values"] = ("cpu", "cuda")
        device_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # モデルサイズ選択
        ttk.Label(settings_window, text="モデルサイズ:").grid(row=1, column=0, padx=5, pady=5)
        model_size_var = tk.StringVar(value=self.config.get("model_size", "small"))
        model_size_combo = ttk.Combobox(settings_window, textvariable=model_size_var)
        model_size_combo["values"] = ("tiny", "base", "small", "medium", "large")
        model_size_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # 計算精度選択
        ttk.Label(settings_window, text="計算精度:").grid(row=2, column=0, padx=5, pady=5)
        compute_type_var = tk.StringVar(value=self.config.get("compute_type", "int8"))
        compute_type_combo = ttk.Combobox(settings_window, textvariable=compute_type_var)
        compute_type_combo["values"] = ("int8", "float16", "float32")
        compute_type_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # 保存ボタン
        def save_settings():
            self.config.update({
                "device": device_var.get(),
                "model_size": model_size_var.get(),
                "compute_type": compute_type_var.get()
            })
            self.save_config()
            settings_window.destroy()
            
            # アプリケーションの再初期化
            if self.is_running:
                self.stop()
            self.app = TranslatorApp(
                device=self.config["device"],
                model_size=self.config["model_size"],
                compute_type=self.config["compute_type"]
            )
            self.app.on_text = self.on_text
            self.app.on_translation = self.on_translation
        
        ttk.Button(settings_window, text="保存", command=save_settings).grid(row=3, column=0, columnspan=2, pady=20)

    def load_config(self) -> dict:
        """
        設定の読み込み
        
        Returns:
            dict: 設定
        """
        config_file = "config.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                return json.load(f)
        return {}

    def save_config(self):
        """設定の保存"""
        config_file = "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=4)    