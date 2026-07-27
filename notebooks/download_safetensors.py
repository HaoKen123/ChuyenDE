# -*- coding: utf-8 -*-
"""
Script chuyên dụng tiếp tục tải 2 file safetensors của Qwen2.5-VL-3B-Instruct trực tiếp bằng HTTP Range (hỗ trợ resume)
"""
import os, sys, io, time
import urllib.request

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SNAPSHOT_DIR = r"C:\Users\honha\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots\66285546d2b821cf421d4f5eb2576359d3770cd3"

urls = {
    "model-00001-of-00002.safetensors": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00002-of-00002.safetensors",
}

# Thử qua hf-mirror nếu huggingface.co bị rớt
mirror_urls = {
    "model-00001-of-00002.safetensors": "https://hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors": "https://hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00002-of-00002.safetensors",
}

def download_file(filename, primary_url, fallback_url):
    dest_path = os.path.join(SNAPSHOT_DIR, filename)
    part_path = dest_path + ".downloading"
    
    downloaded = 0
    if os.path.exists(part_path):
        downloaded = os.path.getsize(part_path)
    
    url = fallback_url  # dùng mirror cho nhanh
    print(f"\n📦 Bắt đầu tải {filename}...")
    print(f"   Lưu tại: {dest_path}")
    print(f"   Đã có sẵn: {downloaded / (1024**2):.1f} MB")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_range = resp.headers.get("Content-Range")
            if content_range:
                total_bytes = int(content_range.split("/")[-1])
            else:
                total_bytes = int(resp.headers.get("Content-Length", 0)) + downloaded
            
            mode = "ab" if downloaded > 0 else "wb"
            last_print = time.time()
            
            with open(part_path, mode) as f:
                while True:
                    chunk = resp.read(1024 * 512)  # 512KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if time.time() - last_print > 3.0:
                        pct = (downloaded / total_bytes * 100) if total_bytes else 0
                        mb_done = downloaded / (1024**2)
                        mb_total = total_bytes / (1024**2) if total_bytes else 0
                        print(f"   ⏳ {filename}: {mb_done:.1f} MB / {mb_total:.1f} MB ({pct:.1f}%)", flush=True)
                        last_print = time.time()
                        
        os.rename(part_path, dest_path)
        print(f"✅ Hoàn thành tải {filename}!")
        return True
    except Exception as e:
        print(f"⚠️ Lỗi khi tải {filename}: {e}")
        return False

print("🚀 Bắt đầu tải 2 file safetensors lớn của Qwen2.5-VL-3B-Instruct...")
for fn in urls:
    download_file(fn, urls[fn], mirror_urls[fn])

print("\n🎉 XONG TẤT CẢ FILE SAFETENSORS!")
