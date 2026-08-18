# 🩸 HemoAI: Hệ Thống Phát Hiện & Phân Loại Tế Bào Máu Đa Mô Hình Kết Hợp XAI

Hệ thống ứng dụng Trí tuệ Nhân tạo Đa phương thức (Multimodal Vision-Language Models) và Thị giác Máy tính (Computer Vision) phục vụ phát hiện, phân loại chuyên sâu 12 dòng tế bào máu và giải thích quyết định y khoa (Explainable AI - XAI).

---

## 🌟 Tính Năng Nổi Bật

1. **Phát hiện tế bào (Cell Detection - YOLO)**:
   - Tự động nhận diện, định vị và khoanh vùng tế bào từ tiêu bản máu vi thể (BCCD dataset).
   - Phân loại 3 lớp cơ bản: **WBC** (Bạch cầu), **RBC** (Hồng cầu), **Platelets** (Tiểu cầu).

2. **Phân loại tế bào chuyên sâu (Fine-grained Classification - Qwen-VL)**:
   - Ứng dụng các mô hình thị giác - ngôn ngữ tiên tiến (**Qwen2-VL-2B** và **Qwen2.5-VL-3B**).
   - Phân loại chính xác **12 phân lớp tế bào máu** (Bạch cầu, Tiền thân dòng tủy, Tiền hồng cầu).
   - **Tính toán Độ tin cậy Toán học (Mathematical Softmax Confidence)**: Trích xuất trực tiếp logits từ token đầu ra của mô hình và tính phân phối xác suất Softmax thực tế (Top-1, Top-3), hoàn toàn không dùng dữ liệu giả lập.

3. **Hỗ trợ Đa Phương Pháp Fine-Tuning (PEFT)**:
   - **DoRA (Weight-Decomposed Low-Rank Adaptation)**: Checkpoint 3315 mới nhất & Final model ($r=8, \alpha=16$) tối ưu hóa độ chính xác và tách biệt độ lớn/hướng trọng số.
   - **QLoRA (Quantized Low-Rank Adaptation)**: Mô hình Quốc Huy Qwen2-VL-2B ($r=8, \alpha=16$) tối ưu bộ nhớ.
   - **LoRA (Low-Rank Adaptation)**: Mô hình Qwen2.5-VL-3B Checkpoint 5500 ($r=16, \alpha=32$).

4. **Tối ưu hóa Phần cứng (4-bit BitsAndBytes Quantization)**:
   - Hỗ trợ nén **4-bit** chạy mượt mà trên GPU phổ thông (chỉ chiếm ~3.2GB VRAM, tương thích hoàn hảo card 4GB như GTX 1050 Ti).
   - Hỗ trợ chuyển đổi linh hoạt sang chế độ **Full 32-bit trên CPU**.
   - Cơ chế **In-Memory Caching / Keep-Alive**: Chỉ nạp weights lần đầu (~25-35s), các lượt phân tích tiếp theo diễn ra tức thì (~0.2-0.5s).

5. **Giải thích quyết định mô hình (Explainable AI - XAI)**:
   - Tích hợp 4 kỹ thuật trực quan hóa vùng chú ý:
     - **HiResCAM**: Độ phân giải cao, hiển thị sắc nét ranh giới tế bào.
     - **XGrad-CAM**: Trực quan hóa tập trung dựa trên gradient chuẩn hóa.
     - **EigenCAM**: Phân tích thành phần chính (PCA), triệt tiêu nhiễu nền.
     - **Integrated Gradients**: Giải thích chính xác theo từng pixel.

6. **Đối chiếu Lịch sử 3 Mô hình (Model Comparison & History)**:
   - Tự động ghi nhận và hiển thị đối chiếu kết quả 3 dòng mô hình (Quốc Huy QLoRA, LoRA Ckpt-5500, Nhật Hào DoRA Ckpt-3315).
   - Bảng lịch sử phân loại chi tiết có thể lưu vết và xóa nhanh.

7. **Luồng phân tích tự động (End-to-End Pipeline)**:
   - Quy trình khép kín: Tiêu bản thô &rarr; YOLO Detect & Bounding Box &rarr; Tự động Crop &rarr; Qwen Classify &rarr; XAI Heatmap.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
code/
├── src/
│   ├── api/
│   │   ├── server.py              # FastAPI backend server
│   │   └── schemas.py             # Pydantic data schemas
│   └── models/
│       ├── model_registry.py      # Bộ quản lý đăng ký các adapter Qwen (DoRA, QLoRA, LoRA)
│       ├── qwen_classifier.py     # Module inference Qwen-VL & tính toán Softmax logits
│       ├── yolo_detector.py       # Module phát hiện tế bào YOLO
│       └── xai_engine.py          # Module sinh bản đồ giải thích XAI (CAM, IntGrad)
├── web/
│   ├── index.html                 # Giao diện Web SPA (Dashboard, Detect, Classify, XAI, Pipeline, Compare)
│   ├── styles.css                 # Hệ thống CSS Design System
│   └── app.js                     # Logic điều khiển Frontend & lưu lịch sử đối chiếu
├── KetQuaMoiNhat_DoRa/            # Weights mô hình DoRA mới nhất (Checkpoint 3315 & Final)
├── FileTrainByQuocHuy_QLoRa_Qwen_2_2B/ # Weights mô hình Quốc Huy QLoRA (Checkpoint 2000 & Final)
├── custom_models/                 # Weights mô hình LoRA r=16 Checkpoint 5500
├── training/
│   ├── training_qwen/             # Scripts & Notebooks huấn luyện Qwen
│   └── training_yolo/             # Scripts & Weights huấn luyện YOLO
├── Dataset-Crop/                  # Tập dữ liệu 12 lớp tế bào phục vụ kiểm thử
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── run.py                         # File chạy khởi động ứng dụng Web
└── README.md                      # Tài liệu hướng dẫn dự án
```

---

## 🏷 Danh Mục Các Lớp Tế Bào Hỗ Trợ

### 1. YOLO Detection (3 lớp tế bào cơ bản)
| Mã | Tên tiếng Việt | Tên khoa học |
| :--- | :--- | :--- |
| **WBC** | Bạch cầu | White Blood Cells / Leukocytes |
| **RBC** | Hồng cầu | Red Blood Cells / Erythrocytes |
| **Platelets** | Tiểu cầu | Platelets / Thrombocytes |

### 2. Qwen-VL Classification (12 lớp tế bào máu chuyên sâu)
| Mã nhãn | Tên lớp tế bào | Ý nghĩa lâm sàng |
| :---: | :--- | :--- |
| **BA** | Basophil | Bạch cầu ái kiềm |
| **BNE** | Band Neutrophil | Bạch cầu đoạn trung tính dạng đũa |
| **EO** | Eosinophil | Bạch cầu ái toan |
| **ERB** | Erythroblast | Tiền hồng cầu có nhân |
| **LY** | Lymphocyte | Bạch cầu Lympho |
| **MMY** | Metamyelocyte | Hậu tủy bào |
| **MO** | Monocyte | Bạch cầu Mono |
| **MY** | Myelocyte | Tủy bào dòng hạt |
| **MYO** | Myeloblast | Nguyên tủy bào |
| **PLT** | Platelet | Tiểu cầu |
| **PMY** | Promyelocyte | Tiền tủy bào |
| **SNE** | Segmented Neutrophil | Bạch cầu trung tính phân đoạn |

---

## ⚙️ Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Cài đặt môi trường Python
Yêu cầu **Python 3.10 - 3.12** và GPU NVIDIA (khuyên dùng CUDA 12.x).

```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Khởi chạy Web Server
Chạy lệnh khởi động máy chủ FastAPI:

```bash
python run.py
```

Sau khi khởi động, mở trình duyệt web và truy cập:
👉 **[http://localhost:8000](http://localhost:8000)** (hoặc `http://127.0.0.1:8000`)

---

## 🔬 Danh Sách Mô Hình Được Đăng Ký Trong Hệ Thống

| Model ID | Tên hiển thị | Kiến trúc Base | Kỹ thuật PEFT | Siêu tham số |
| :--- | :--- | :--- | :---: | :--- |
| `qwen2.5-vl-3b-dora-r8-checkpoint-3315` | **Nhật Hào — Checkpoint 3315** *(Mặc định)* | Qwen2.5-VL-3B | **DoRA** | $r=8, \alpha=16, \text{dropout}=0.1$ |
| `qwen2.5-vl-3b-dora-r8-final` | **Nhật Hào — Final Model** | Qwen2.5-VL-3B | **DoRA** | $r=8, \alpha=16, \text{dropout}=0.1$ |
| `quoc-huy-qwen2-vl-2b-qlora-r8-final` | **Quốc Huy — Final Model** | Qwen2-VL-2B | **QLoRA** | $r=8, \alpha=16, \text{dropout}=0.05$ |
| `quoc-huy-qwen2-vl-2b-qlora-r8-checkpoint-2000` | **Quốc Huy — Checkpoint 2000** | Qwen2-VL-2B | **QLoRA** | $r=8, \alpha=16, \text{dropout}=0.05$ |
| `qwen2.5-vl-3b-lora-r16-checkpoint-5500` | **LoRA Checkpoint 5500** | Qwen2.5-VL-3B | **LoRA** | $r=16, \alpha=32, \text{dropout}=0.05$ |
