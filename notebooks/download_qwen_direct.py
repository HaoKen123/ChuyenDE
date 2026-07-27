# -*- coding: utf-8 -*-
"""
Script tải trực tiếp từng file của Qwen/Qwen2.5-VL-3B-Instruct qua hf-mirror.com
"""
import os, sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

repo_id = "Qwen/Qwen2.5-VL-3B-Instruct"
files = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
]

print(f"🚀 Bắt đầu tải lần lượt {len(files)} tệp tin của {repo_id} qua hf-mirror.com...")

for idx, filename in enumerate(files, 1):
    try:
        print(f"[{idx}/{len(files)}] Đang tải tệp: {filename} ...", flush=True)
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
        print(f"   ✅ Đã xong: {filename} -> {file_path}", flush=True)
    except Exception as e:
        print(f"   ⚠️ Bỏ qua/Lỗi {filename}: {e}", flush=True)

print("\n🎉 HOÀN THÀNH TẢI TẤT CẢ FILE BASE MODEL!")
