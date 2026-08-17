import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

# =====================================================================
# CẤU HÌNH DÀNH CHO BẠN (Thay đổi thông tin tại đây)
# =====================================================================

# 1. Biến cấu hình địa chỉ file ảnh test trên laptop của bạn
IMAGE_PATH = r"D:\.chuyende\code\dataset_split\dataset_split\test\MY\MY_img32668.jpg"  # <-- Sửa đường dẫn này

# 2. Đường dẫn tới thư mục checkpoint bạn đã train (ví dụ checkpoint-4000)
LORA_CHECKPOINT_PATH = r"D:\.chuyende\code\custom_models\qwen_blood_cell\checkpoint-5500" 

# 3. Prompt (Câu lệnh) để hỏi mô hình. 
# QUAN TRỌNG: Hãy nhập đúng câu prompt mà bạn đã dùng trong file dataset lúc train!
PROMPT_TEXT = "Hãy phân loại tế bào máu trong bức ảnh này."

# Model gốc (giữ nguyên)
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# =====================================================================

def main():
    print(f"Bắt đầu tải ảnh từ: {IMAGE_PATH}")
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
    except Exception as e:
        print(f"LỖI: Không thể mở ảnh. Vui lòng kiểm tra lại đường dẫn IMAGE_PATH.\nChi tiết lỗi: {e}")
        return

    print("Đang tải Processor và cấu hình nén 4-bit...")
    # Khởi tạo cấu hình 4-bit giống hệt lúc train để tránh lỗi lệch trọng số
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    print("Đang tải Base Model (Sẽ mất một chút thời gian)...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print(f"Đang đắp bản vá LoRA từ {LORA_CHECKPOINT_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT_PATH)
    model.eval() # Đưa mô hình về chế độ đánh giá (không cập nhật trọng số)

    # Chuẩn bị dữ liệu đầu vào theo định dạng của Qwen-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]

    # Xử lý text và hình ảnh
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    print("Đang phân tích hình ảnh và dự đoán kết quả...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        # Loại bỏ phần token của input prompt để chỉ lấy câu trả lời
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    print("\n" + "="*50)
    print("🤖 KẾT QUẢ TỪ MÔ HÌNH:")
    print("="*50)
    print(output_text[0])
    print("="*50)

if __name__ == "__main__":
    main()