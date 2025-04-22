import tkinter as tk
from tkinter import ttk
import threading
from typing import Optional
import json
import os

from core.app import TranslatorApp

class MainWindow:
    def __init__(self, root: tk.Tk):
        """
        メインウィンドウの初期化
        
        Args:
            root: Tkinterのルートウィンドウ
        """
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
        
        # GUIの初期化
        self.setup_ui()
        
        # コールバックの設定
        self.app.on_text = self.on_text
        self.app.on_translation = self.on_translation
        
        # 状態管理
        self.is_running = False

    def setup_ui(self):
        """UIの初期化"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 原文表示エリア
        ttk.Label(main_frame, text="原文:").grid(row=0, column=0, sticky=tk.W)
        self.source_text = tk.Text(main_frame, height=10, width=80)
        self.source_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 訳文表示エリア
        ttk.Label(main_frame, text="訳文:").grid(row=2, column=0, sticky=tk.W)
        self.target_text = tk.Text(main_frame, height=10, width=80)
        self.target_text.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 制御ボタン
        self.start_button = ttk.Button(main_frame, text="開始", command=self.start)
        self.start_button.grid(row=4, column=0, pady=10)
        
        self.stop_button = ttk.Button(main_frame, text="停止", command=self.stop)
        self.stop_button.grid(row=4, column=1, pady=10)
        self.stop_button.state(["disabled"])
        
        # 設定ボタン
        self.settings_button = ttk.Button(main_frame, text="設定", command=self.show_settings)
        self.settings_button.grid(row=4, column=2, pady=10)
        
        # ステータスバー
        self.status_var = tk.StringVar(value="準備完了")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))

    def start(self):
        """翻訳の開始"""
        if not self.is_running:
            self.is_running = True
            self.start_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
            self.status_var.set("翻訳中...")
            
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

    def on_text(self, text: str, lang: str):
        """
        テキスト認識時のコールバック
        
        Args:
            text: 認識されたテキスト
            lang: 言語コード
        """
        self.source_text.delete("1.0", tk.END)
        self.source_text.insert("1.0", f"[{lang}] {text}")

    def on_translation(self, text: str, lang: str):
        """
        翻訳完了時のコールバック
        
        Args:
            text: 翻訳されたテキスト
            lang: 言語コード
        """
        self.target_text.delete("1.0", tk.END)
        self.target_text.insert("1.0", f"[{lang}] {text}")

    def show_settings(self):
        """設定ダイアログの表示"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("設定")
        settings_window.geometry("400x300")
        
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