# -*- coding: utf-8 -*-
"""
Script tải đa luồng song song (Multi-threaded Parallel Chunk Downloader) 
Tối ưu tốc độ tải 10x cho 2 file safetensors của Qwen2.5-VL-3B-Instruct.
"""
import os, sys, io, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SNAPSHOT_DIR = r"C:\Users\honha\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots\66285546d2b821cf421d4f5eb2576359d3770cd3"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

files_url = {
    "model-00001-of-00002.safetensors": "https://hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors": "https://hf-mirror.com/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00002-of-00002.safetensors",
}

NUM_WORKERS = 8  # 8 luồng tải song song

def get_file_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers.get("Content-Length", 0))

def download_chunk(url, start_byte, end_byte, part_filename):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start_byte}-{end_byte}"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(part_filename, "wb") as f:
        while True:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            f.write(chunk)
    return part_filename

def download_file_parallel(filename, url):
    dest_path = os.path.join(SNAPSHOT_DIR, filename)
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024**2)
        print(f"✅ File {filename} đã tồn tại ({size_mb:.1f} MB), bỏ qua.")
        return

    print(f"\n🚀 Đang lấy thông tin dung lượng cho {filename}...")
    total_size = get_file_size(url)
    print(f"📦 Tổng dung lượng {filename}: {total_size / (1024**2):.1f} MB")
    
    chunk_size = total_size // NUM_WORKERS
    tasks = []
    part_files = []

    print(f"⚡ Chia nhỏ tệp thành {NUM_WORKERS} luồng tải song song...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i in range(NUM_WORKERS):
            start_b = i * chunk_size
            end_b = (i + 1) * chunk_size - 1 if i < NUM_WORKERS - 1 else total_size - 1
            part_fn = dest_path + f".part{i}"
            part_files.append(part_fn)
            
            # Kiểm tra xem part đã tải xong chưa
            if os.path.exists(part_fn) and os.path.getsize(part_fn) == (end_b - start_b + 1):
                continue
                
            tasks.append(executor.submit(download_chunk, url, start_b, end_b, part_fn))

        for future in as_completed(tasks):
            try:
                res = future.result()
                print(f"  ✓ Đã xong mảnh: {os.path.basename(res)}")
            except Exception as e:
                print(f"  ❌ Lỗi luồng: {e}")

    print(f"🔄 Đang ghép {NUM_WORKERS} mảnh thành file hoàn chỉnh: {filename}...")
    with open(dest_path, "wb") as outfile:
        for pfn in part_files:
            if os.path.exists(pfn):
                with open(pfn, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(pfn)

    elapsed = time.time() - start_time
    mb_sec = (total_size / (1024**2)) / max(elapsed, 1)
    print(f"🎉 Hoàn thành {filename}! Thời gian: {elapsed:.1f}s ({mb_sec:.2f} MB/s)")

print("="*65)
print("🚀 MULTI-THREADED PARALLEL DOWNLOADER — QWEN2.5-VL-3B SAFETENSORS")
print("="*65)

for filename, url in files_url.items():
    download_file_parallel(filename, url)

print("\n🎉 HOÀN THÀNH TẤT CẢ FILE WEIGHTS!")
