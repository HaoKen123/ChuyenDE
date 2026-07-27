# -*- coding: utf-8 -*-
"""
Script kiểm tra & suy luận trực tiếp từ LoRA Checkpoint-1500 + Qwen2.5-VL-3B-Instruct
"""

import sys, os, io
from pathlib import Path
import torch

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoint-1500"
if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR = Path("checkpoint-1500")

print(f"📌 Thư mục Checkpoint: {CHECKPOINT_DIR}")
print(f"📌 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📌 GPU Device: {torch.cuda.get_device_name(0)}")

print("\n📥 Đang nạp Base Model Qwen/Qwen2.5-VL-3B-Instruct & ghép LoRA Adapter checkpoint-1500...")

try:
    from peft import PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image

    processor = AutoProcessor.from_pretrained(str(CHECKPOINT_DIR), trust_remote_code=True)
    
    print("⏳ Nạp Base Model từ HuggingFace Cache (cần khoảng 6GB)...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    print("⏳ Ghép LoRA Adapter từ checkpoint-1500...")
    model = PeftModel.from_pretrained(base_model, str(CHECKPOINT_DIR))
    model.eval()
    print("✅ ĐÃ NẠP THÀNH CÔNG CHECKPOINT-1500 QWEN2.5-VL-3B!")

    # Tìm thử 1 ảnh mẫu để test
    sample_images = list(Path(".").glob("**/*.jpg")) + list(Path(".").glob("**/*.png"))
    if sample_images:
        test_img_path = sample_images[0]
        print(f"\n🖼️ Chạy thử nghiệm trên ảnh: {test_img_path}")
        image = Image.open(test_img_path).convert("RGB")
        
        prompt = "Identify the blood cell type in this image and explain its key morphological features."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        print("\n🤖 KẾT QUẢ TỪ QWEN2.5-VL LORA (CHECKPOINT-1500):")
        print("="*60)
        print(output_text)
        print("="*60)
    else:
        print("⚠️ Không tìm thấy ảnh test trong thư mục.")

except Exception as e:
    print(f"\n❌ Lỗi khi nạp/chạy Checkpoint-1500: {e}")
    import traceback
    traceback.print_exc()
