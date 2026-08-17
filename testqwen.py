import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Tùy chỉnh ở đây)
# ==========================================
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Đường dẫn đến thư mục checkpoint LoRA của bạn
LORA_MODEL_PATH = r"D:\.chuyende\code\custom_models\qwen_blood_cell\checkpoint-5500" 

# Đường dẫn đến bức ảnh cần dự đoán
IMAGE_PATH = r"D:\.chuyende\code\dataset_split\dataset_split\test\MY\MY_img32668.jpg"
# ==========================================
# 2. KHỞI TẠO PROMPT (Giống hệt lúc train)
# ==========================================
CELL_LABELS = ["BA", "BNE", "EO", "ERB", "LY", "MMY", "MO", "MY", "MYO", "PLT", "PMY", "SNE"]
CELL_NAMES = {
    "BA": "Basophil", "BNE": "Band Neutrophil", "EO": "Eosinophil",
    "ERB": "Erythroblast", "LY": "Lymphocyte", "MMY": "Metamyelocyte",
    "MO": "Monocyte", "MY": "Myelocyte", "MYO": "Myeloblast",
    "PLT": "Platelet", "PMY": "Promyelocyte", "SNE": "Segmented Neutrophil"
}

def build_prompt():
    cell_list = ", ".join(CELL_LABELS)
    prompt = (
        f"This is a microscopic image of a blood cell. "
        f"Classify this cell into exactly one of these categories: {cell_list}. "
        f"These are abbreviations for: "
    )
    for cl in CELL_LABELS:
        prompt += f"{cl}={CELL_NAMES[cl]}, "
    prompt = prompt.rstrip(", ") + ". "
    prompt += "Reply with ONLY the abbreviation code (e.g., 'LY')."
    return prompt

# ==========================================
# 3. TẢI MÔ HÌNH VÀ DỰ ĐOÁN
# ==========================================
def predict_cell():
    print("⏳ Đang tải mô hình gốc (chế độ 4-bit để tối ưu VRAM)...")
    
    # Cấu hình load model 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Tải model gốc với cấu hình 4-bit và thư mục offload dự phòng
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config,
        offload_folder="offload"
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

    print(f"🔗 Đang áp dụng trọng số LoRA từ: {LORA_MODEL_PATH}")
    # Load trọng số LoRA (Adapter) đè lên model gốc
    model = PeftModel.from_pretrained(
        base_model, 
        LORA_MODEL_PATH,
        offload_folder="offload"
    )
    model.eval() # Chuyển sang chế độ dự đoán

    print(f"🖼️ Đang xử lý ảnh: {IMAGE_PATH}")
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
    except Exception as e:
        print(f"❌ Không tìm thấy hoặc không mở được ảnh! Lỗi: {e}")
        return

    # Chuẩn bị dữ liệu đầu vào
    prompt = build_prompt()
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=False, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("🧠 Mô hình đang suy nghĩ...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    # Giải mã đầu ra
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], outputs)]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    result = generated_text.strip()
    
    print("\n" + "="*40)
    print(f"🎯 KẾT QUẢ DỰ ĐOÁN: {result}")
    if result in CELL_NAMES:
        print(f"🩺 Tên đầy đủ: {CELL_NAMES[result]}")
    else:
        print("⚠️ Cảnh báo: Mô hình trả về kết quả không nằm trong danh sách nhãn chuẩn!")
    print("="*40)

if __name__ == "__main__":
    # Chỉ chạy nếu đường dẫn tồn tại
    import os
    if not os.path.exists(LORA_MODEL_PATH):
        print(f"❌ Không tìm thấy mô hình tại: {LORA_MODEL_PATH}")
    else:
        predict_cell()