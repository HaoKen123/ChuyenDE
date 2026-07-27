# -*- coding: utf-8 -*-
"""
Script chuyên dụng tốc độ cao (hf-mirror.com) để tải Base Model Qwen/Qwen2.5-VL-3B-Instruct.
"""
import os, sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Sử dụng hf-mirror.com để tải tốc độ cao không bị throttle
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("🚀 Bắt đầu tải tốc độ cao Base Model Qwen/Qwen2.5-VL-3B-Instruct qua Mirror...")
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    print(f"\n🎉 THÀNH CÔNG! ĐÃ TẢI HOÀN TẤT BASE MODEL QWEN2.5-VL-3B-INSTRUCT!")
    print(f"📁 Đường dẫn lưu: {path}")
except Exception as e:
    print(f"\n❌ Lỗi trong quá trình tải: {e}")
