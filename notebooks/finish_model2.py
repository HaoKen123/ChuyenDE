# -*- coding: utf-8 -*-
"""
Script hoàn tất ghép tệp model-00002-of-00002.safetensors
"""
import os, sys, io, urllib.request

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SNAPSHOT_DIR = r"C:\Users\honha\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots\66285546d2b821cf421d4f5eb2576359d3770cd3"
url = "https://hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00002-of-00002.safetensors"
dest_file = os.path.join(SNAPSHOT_DIR, "model-00002-of-00002.safetensors")

# Xóa file 0-byte cũ nếu có
if os.path.exists(dest_file) and os.path.getsize(dest_file) == 0:
    os.remove(dest_file)

total_size = 1399742000
num_workers = 8
chunk_size = total_size // num_workers

print("📦 Đang kiểm tra 8 phần của model-00002...")
part_files = []
for i in range(num_workers):
    start_b = i * chunk_size
    end_b = (i + 1) * chunk_size - 1 if i < num_workers - 1 else total_size - 1
    expected_len = end_b - start_b + 1
    
    pfn = os.path.join(SNAPSHOT_DIR, f"model-00002-of-00002.safetensors.part{i}")
    part_files.append(pfn)
    
    if not os.path.exists(pfn) or os.path.getsize(pfn) != expected_len:
        print(f"⏳ Đang tải lại mảnh {i} ({expected_len / (1024**2):.1f} MB)...", flush=True)
        headers = {"User-Agent": "Mozilla/5.0", "Range": f"bytes={start_b}-{end_b}"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(pfn, "wb") as f:
            f.write(resp.read())
        print(f"   ✓ Xong mảnh {i}")
    else:
        print(f"  ✓ Mảnh {i} đã hoàn chỉnh ({expected_len / (1024**2):.1f} MB)")

print(f"🔄 Đang ghép 8 mảnh vào tệp hoàn chỉnh {dest_file}...")
with open(dest_file, "wb") as outfile:
    for pfn in part_files:
        with open(pfn, "rb") as infile:
            outfile.write(infile.read())

for pfn in part_files:
    if os.path.exists(pfn):
        os.remove(pfn)

print(f"🎉 THÀNH CÔNG! File model-00002-of-00002.safetensors hoàn chỉnh đạt {os.path.getsize(dest_file) / (1024**2):.1f} MB!")
