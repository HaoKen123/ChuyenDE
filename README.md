# Hệ Thống Phát Hiện & Phân Loại Tế Bào Máu Đa Mô Hình Kết Hợp XAI

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Transformers-Qwen--VL-yellow.svg)](https://huggingface.co/docs/transformers/)
[![PEFT](https://img.shields.io/badge/PEFT-DoRA%20%7C%20QLoRA%20%7C%20LoRA-brightgreen.svg)](https://github.com/huggingface/peft)
[![YOLO](https://img.shields.io/badge/YOLO-Detection-orange.svg)](https://github.com/ultralytics/ultralytics)

Hệ thống ứng dụng **Trí tuệ Nhân tạo Đa phương thức (Multimodal Vision-Language Models - Qwen-VL)** kết hợp **Thị giác Máy tính (Computer Vision - YOLO)** phục vụ phát hiện, phân loại chuyên sâu 12 dòng tế bào máu vi thể và trực quan hóa quyết định y khoa (**Explainable AI - XAI**).

---

## 📑 Mục Lục
1. [Tính Năng Nổi Bật](#-tính-năng-nổi-bật)
2. [Cấu Trúc Thư Mục Dự Án](#-cấu-trúc-thư-mục-dự-án)
3. [Danh Mục 12 Lớp Tế Bào Máu](#-danh-mục-12-lớp-tế-bào-máu)
4. [Danh Sách Mô Hình Được Đăng Ký](#-danh-sách-mô-hình-được-đăng-ký)
5. [Hướng Dẫn Cài Đặt Chi Tiết](#-hướng-dẫn-cài-đặt-chi-tiết)
6. [Hướng Dẫn Khởi Chạy & Sử Dụng](#-hướng-dẫn-khởi-chạy--sử-dụng)
7. [Hướng Dẫn Huấn Luyện Mô Hình](#-hướng-dẫn-huấn-luyện-mô-hình)
8. [Câu Hỏi Thường Gặp & Xử Lý Sự Cố (FAQ)](#-câu-hỏi-thường-gặp--xử-lý-sự-cố-faq)

---

## 🌟 Tính Năng Nổi Bật

### 1. Phát hiện Tế bào (Cell Detection - YOLO)
- Tự động nhận diện, định vị và tạo hộp bao (Bounding Box) cho các tế bào từ tiêu bản máu vi thể toàn cảnh (BCCD Dataset).
- Đếm và phân loại 3 thành phần chính: **WBC** (Bạch cầu), **RBC** (Hồng cầu), **Platelets** (Tiểu cầu).
- Hỗ trợ cắt tự động (Auto-crop) các tế bào bạch cầu để chuyển tiếp vào module phân loại chuyên sâu.

### 2. Phân loại Chuyên sâu 12 Lớp (Fine-grained Classification - Qwen-VL)
- Tích hợp các Vision-Language Foundation Models tiên tiến: **Qwen2-VL-2B** và **Qwen2.5-VL-3B**.
- Phân loại chính xác 12 dòng tế bào máu (Bạch cầu trưởng thành, tiền thân dòng tủy, tiền hồng cầu).
- **Tính toán Độ tin cậy Toán học (Mathematical Softmax Confidence)**: Trích xuất trực tiếp logits từ token đầu ra của mô hình qua hàm Softmax chuẩn $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{12} e^{z_j}}$, hiển thị độ tin cậy thực tế (%) và phân phối Top-3 xác suất cao nhất (hoàn toàn không dùng số liệu giả lập).

### 3. Đa Dạng Kỹ Thuật Fine-Tuning (PEFT)
- **DoRA (Weight-Decomposed Low-Rank Adaptation)**: Phiên bản Checkpoint 3315 mới nhất và Final model ($r=8, \alpha=16$), phân rã ma trận trọng số thành độ lớn (magnitude) và hướng (direction), mang lại độ chính xác phân loại vượt trội.
- **QLoRA (Quantized Low-Rank Adaptation)**: Mô hình Quốc Huy Qwen2-VL-2B ($r=8, \alpha=16$) tối ưu hóa bộ nhớ RAM/VRAM.
- **LoRA (Low-Rank Adaptation)**: Mô hình baseline Qwen2.5-VL-3B Checkpoint 5500 ($r=16, \alpha=32$).

### 4. Tối Ưu Hóa Bộ Nhớ & Tăng Tốc Phần Cứng
- **Nén 4-bit (BitsAndBytes NF4 Quantization)**: Cho phép mô hình 3B tham số chạy mượt mà trên GPU chỉ 4GB VRAM (như NVIDIA GeForce GTX 1050 Ti) với mức chiếm dụng bộ nhớ chỉ ~3.2GB VRAM.
- **Chế độ CPU (Full 32-bit)**: Tự động tương thích và chạy ổn định trên các máy không trang bị GPU rời.
- **Cơ chế In-Memory Caching (Keep-Alive)**: Chỉ nạp weights một lần duy nhất lúc khởi đầu (~20–35 giây), các lượt phân loại ảnh tiếp theo diễn ra tức thì (~0.2–0.5 giây).

### 5. Trực Quan Hóa Giải Thích Quyết Định (Explainable AI - XAI)
- **HiResCAM**: Giữ nguyên độ phân giải cao của feature map, thể hiện sắc nét ranh giới tế bào tác động đến dự đoán.
- **XGrad-CAM**: Trực quan hóa tập trung dựa trên gradient chuẩn hóa theo tỷ lệ kích hoạt.
- **EigenCAM**: Sử dụng phân tích thành phần chính (PCA) trích xuất đặc trưng thị giác chủ đạo, loại bỏ nhiễu nền tiêu bản.
- **Integrated Gradients**: Tích phân gradient dọc theo đường đi tuyến tính từ ảnh baseline đến ảnh gốc, giải thích chính xác đến từng pixel.

### 6. Đối Chiếu Lịch Sử 3 Mô Hình (Model Comparison & History)
- Tự động lưu vết và đối chiếu kết quả dự đoán của 3 dòng mô hình (**Quốc Huy QLoRA**, **LoRA Ckpt-5500**, **Nhật Hào DoRA Ckpt-3315**) từ tab Phân loại.
- Bảng lịch sử phân loại chi tiết có hình ảnh thumbnail, nhãn dự đoán, độ tin cậy Softmax %, tốc độ xử lý (ms) và nút xóa lịch sử tiện lợi.

### 7. Quy Trình Tự Động Toàn Diện (End-to-End Pipeline)
- Luồng phân tích 4 bước khép kín chỉ với 1 click: Tiêu bản thô &rarr; YOLO Detect & Bounding Box &rarr; Tự động Crop từng tế bào &rarr; Qwen Classify &rarr; Sinh bản đồ giải thích XAI.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
code/
├── src/
│   ├── api/
│   │   ├── server.py              # FastAPI backend server & RESTful API endpoints
│   │   └── schemas.py             # Pydantic data models & request/response schemas
│   └── models/
│       ├── model_registry.py      # Bộ điều phối & quản lý metadata các adapter Qwen
│       ├── qwen_classifier.py     # Module inference Qwen-VL & trích xuất Softmax logits
│       ├── yolo_detector.py       # Module phát hiện tế bào YOLO (Inference & Bounding Box)
│       └── xai_engine.py          # Module tính toán Heatmap XAI (CAM & Integrated Gradients)
├── web/
│   ├── index.html                 # Giao diện SPA (Single Page Application)
│   ├── styles.css                 # CSS Design System chuẩn y tế (Dark/Light Responsive)
│   └── app.js                     # Điều khiển frontend, gọi API & lưu lịch sử đối chiếu
├── KetQuaMoiNhat_DoRa/            # Trọng số mô hình DoRA (Checkpoint 3315 Mới nhất & Final)
├── FileTrainByQuocHuy_QLoRa_Qwen_2_2B/ # Trọng số mô hình Quốc Huy QLoRA (Ckpt 2000 & Final)
├── custom_models/                 # Trọng số mô hình LoRA r=16 Checkpoint 5500
├── training/
│   ├── training_qwen/             # Mã nguồn & Notebooks huấn luyện mô hình Qwen
│   └── training_yolo/             # Mã nguồn & Trọng số huấn luyện mô hình YOLO
├── Dataset-Crop/                  # Tập dữ liệu mẫu 12 lớp tế bào máu
├── requirements.txt               # Danh sách thư viện Python phụ thuộc
├── run.py                         # File thực thi khởi động ứng dụng Web
└── README.md                      # Tài liệu hướng dẫn sử dụng dự án
```

---

## 🏷 Danh Mục 12 Lớp Tế Bào Máu

| Mã Nhãn | Tên Tiếng Việt | Tên Khoa Học | Ý Nghĩa Y Khoa / Lâm Sàng |
| :---: | :--- | :--- | :--- |
| **BA** | Bạch cầu ái kiềm | Basophil | Bạch cầu hạt chứa histamine, tham gia phản ứng viêm và dị ứng. |
| **BNE** | Bạch cầu đũa trung tính | Band Neutrophil | Bạch cầu trung tính non chưa phân đoạn hoàn toàn, nhân hình chữ U. |
| **EO** | Bạch cầu ái toan | Eosinophil | Bạch cầu hạt bắt màu acid cam đỏ, chống nhiễm ký sinh trùng. |
| **ERB** | Tiền hồng cầu | Erythroblast | Tế bào tiền thân hồng cầu có nhân trong tủy xương. |
| **LY** | Bạch cầu Lympho | Lymphocyte | Tế bào miễn dịch chủ chốt (Lympho T, B), nhân tròn lớn. |
| **MMY** | Hậu tủy bào | Metamyelocyte | Tế bào tủy dòng hạt giai đoạn sau, nhân bắt đầu lõm hình hạt đậu. |
| **MO** | Bạch cầu Mono | Monocyte | Bạch cầu kích thước lớn nhất, sau khi vào mô biến thành đại thực bào. |
| **MY** | Tủy bào dòng hạt | Myelocyte | Tế bào tủy trung gian dòng bạch cầu hạt, nhân tròn/bầu dục. |
| **MYO** | Nguyên tủy bào | Myeloblast | Tế bào đầu dòng non nhất của dòng bạch cầu hạt trong tủy xương. |
| **PLT** | Tiểu cầu | Platelet / Thrombocyte | Mảnh tế bào không nhân, đóng vai trò chính trong đông cầm máu. |
| **PMY** | Tiền tủy bào | Promyelocyte | Giai đoạn phát triển kế tiếp nguyên tủy bào, xuất hiện hạt tiên phát. |
| **SNE** | Bạch cầu phân đoạn | Segmented Neutrophil | Bạch cầu trưởng thành đông đảo nhất trong máu, nhân chia 2–5 múi. |

---

## 🔬 Danh Sách Mô Hình Được Đăng Ký

Hệ thống quản lý thống nhất các mô hình qua bộ điều phối [`src/models/model_registry.py`](file:///d:/code_QV/code/src/models/model_registry.py):

| Mã Định Danh (Model ID) | Tên Hiển Thị | Base Model | Kỹ Thuật | Tham Số PEFT |
| :--- | :--- | :--- | :---: | :--- |
| `qwen2.5-vl-3b-dora-r8-checkpoint-3315` | **Nhật Hào — DoRA Ckpt-3315** *(Mặc định)* | Qwen2.5-VL-3B | **DoRA** | $r=8, \alpha=16, \text{dropout}=0.1$ |
| `qwen2.5-vl-3b-dora-r8-final` | **Nhật Hào — DoRA Final Model** | Qwen2.5-VL-3B | **DoRA** | $r=8, \alpha=16, \text{dropout}=0.1$ |
| `quoc-huy-qwen2-vl-2b-qlora-r8-final` | **Quốc Huy — Qwen2-VL-2B (Final)** | Qwen2-VL-2B | **QLoRA** | $r=8, \alpha=16, \text{dropout}=0.05$ |
| `quoc-huy-qwen2-vl-2b-qlora-r8-checkpoint-2000` | **Quốc Huy — Qwen2-VL-2B (Ckpt-2000)** | Qwen2-VL-2B | **QLoRA** | $r=8, \alpha=16, \text{dropout}=0.05$ |
| `qwen2.5-vl-3b-lora-r16-checkpoint-5500` | **LoRA Checkpoint 5500** | Qwen2.5-VL-3B | **LoRA** | $r=16, \alpha=32, \text{dropout}=0.05$ |

---

## 🛠 Hướng Dẫn Cài Đặt Chi Tiết

### 1. Yêu Cầu Hệ Thống
- **Hệ điều hành**: Windows 10/11 (64-bit) hoặc Linux (Ubuntu 20.04+).
- **Python**: Phiên bản **3.10**, **3.11** hoặc **3.12** (khuyên dùng Python 3.11).
- **Phần cứng khuyên dùng**:
  - GPU: NVIDIA GeForce GTX 1050 Ti trở lên (VRAM $\ge$ 4GB, hỗ trợ CUDA 11.8 hoặc 12.x).
  - RAM: Tối thiểu 8GB (khuyên dùng 16GB).
  - Ổ cứng trống: Tối thiểu 10GB.

### 2. Thiết Lập Môi Trường Ảo (Virtual Environment)
Mở Terminal / PowerShell tại thư mục gốc của dự án (`d:\code_QV\code`):

```bash
# Tạo môi trường ảo tên .venv
python -m venv .venv

# Kích hoạt môi trường ảo trên Windows:
.venv\Scripts\activate

# (Hoặc kích hoạt trên Linux/macOS: source .venv/bin/activate)
```

### 3. Cài Đặt PyTorch Hỗ Trợ GPU CUDA
> ⚠️ **QUAN TRỌNG**: Cài đặt PyTorch tương thích CUDA trước khi cài đặt các thư viện khác:

```bash
# Đối với GPU NVIDIA (CUDA 12.1 / 12.4):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# (Hoặc chỉ dùng CPU nếu máy không có GPU rời):
# pip install torch torchvision
```

### 4. Cài Đặt Các Thư Viện Phụ Thuộc
Chạy lệnh sau để cài đặt toàn bộ dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Khởi Chạy & Sử Dụng

### 1. Khởi Động Server Ứng Dụng
Chạy lệnh thực thi sau tại thư mục gốc:

```bash
python run.py
```

Khi server khởi động thành công, màn hình sẽ hiển thị:
```text
[SYSTEM] Starting HemoAI Medical Research Web Server...
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 2. Truy Cập Giao Diện Web
Mở trình duyệt (Chrome, Edge, Firefox, Brave) và truy cập vào địa chỉ:
👉 **[http://localhost:8000](http://localhost:8000)** (hoặc `http://127.0.0.1:8000`)

---

### 3. Hướng Dẫn Sử Dụng Chi Tiết Từng Tab Chức Năng

#### 📊 Tab 1: Dashboard (Tổng quan hệ thống)
- Theo dõi trạng thái hoạt động của Backend API (`Online` / `Offline`).
- Xem cấu hình phần cứng hiện tại (Tên GPU, bộ nhớ VRAM khả dụng, Model đang kích hoạt).
- Xem thống kê nhanh về các mô hình đã đăng ký và danh mục 12 lớp tế bào máu.

#### 🎯 Tab 2: Phát hiện Tế bào (YOLO Detection)
1. **Tải ảnh**: Kéo thả ảnh tiêu bản máu thô hoặc bấm **"chọn từ máy tính"** (hoặc chọn 1 trong các ảnh mẫu có sẵn bên dưới).
2. **Cấu hình**:
   - Chọn Model YOLO: `yolo26n.pt` hoặc `best.pt`.
   - Chỉnh **Ngưỡng tự tin (Confidence Threshold)**: Mặc định `0.25`.
3. **Thực thi**: Bấm nút **"Chạy phát hiện tế bào"**.
4. **Kết quả**:
   - Hình ảnh hiển thị Bounding Box và nhãn từng tế bào (`WBC`, `RBC`, `Platelets`).
   - Bảng thống kê số lượng tế bào theo từng loại.
   - Danh sách các tế bào bạch cầu đã cắt tự động (Crops) có thể chọn để gửi trực tiếp sang tab **Phân loại** hoặc **XAI**.

#### 🧠 Tab 3: Phân loại Tế bào Chuyên sâu (Qwen-VL)
1. **Tải ảnh**: Tải lên ảnh tế bào đơn lẻ (đã crop) hoặc bấm chọn mẫu nhanh trong 12 phân lớp tế bào máu (`BA`, `LY`, `MO`, `EO`,...).
2. **Cấu hình mô hình**:
   - **Chọn Model Qwen**: Chọn trong dropdown (ví dụ: *Nhật Hào — DoRA Ckpt-3315*, *Quốc Huy — QLoRA Final*, hoặc *LoRA Ckpt-5500*).
   - **Chất lượng nén (Quantization)**: Chọn **4-bit (GPU CUDA — Tối ưu VRAM)** cho card 4GB hoặc **Full 32-bit (CPU)**.
   - **Thiết bị**: Chọn `GPU (CUDA)` hoặc `CPU`.
3. **Thực thi**: Bấm nút **"QWen phân loại tế bào"**.
4. **Kết quả chi tiết**:
   - **Nhãn dự đoán**: Badge màu đặc trưng kèm tên khoa học của loại tế bào.
   - **Độ tin cậy mô hình (%)**: Phần trăm xác suất Softmax toán học thực tế và thanh tiến trình trực quan.
   - **Top-3 Xác suất phân loại**: Biểu đồ phân phối 3 lớp có xác suất cao nhất.
   - **Thông tin kỹ thuật**: Model đã dùng, thời gian suy luận (`ms`) và raw text output.

#### ✨ Tab 4: Giải Thích Mô Hình Trực Quan (XAI)
1. **Tải ảnh**: Tải ảnh tế bào đơn lẻ hoặc chọn ảnh mẫu.
2. **Chọn phương pháp XAI**:
   - `HiResCAM`: Độ nét cao, bám sát cấu trúc hình thái tế bào.
   - `XGrad-CAM`: Tập trung vào các vùng quyết định chính.
   - `EigenCAM`: Phân tích đặc trưng chủ đạo, loại bỏ nhiễu.
   - `Integrated Gradients`: Độ chính xác cấp độ pixel.
3. **Tùy chỉnh**: Chỉnh độ trong suốt Heatmap Overlay ($\alpha \in [0.1, 0.9]$).
4. **Thực thi**: Bấm **"Tạo bản đồ giải thích"** hoặc **"So sánh tất cả phương pháp"** để xem song song 4 Heatmaps đối chiếu.

#### ⚡ Tab 5: Quy Trình Toàn Diện (End-to-End Pipeline)
1. Tải 1 ảnh tiêu bản vi thể toàn cảnh.
2. Thiết lập cấu hình YOLO và Qwen.
3. Bấm **"Chạy toàn bộ Pipeline"**.
4. Hệ thống tự động xử lý qua 4 bước: **Phát hiện &rarr; Cắt ảnh &rarr; Phân loại &rarr; Sinh Heatmap XAI**, hiển thị bảng tổng hợp kết quả của toàn bộ tế bào trong ảnh.

#### 🔄 Tab 6: Đối Chiếu Lịch Sử 3 Mô Hình (Model Comparison)
- **3 Thẻ Đối Chiếu Trực Tiếp**: Hiển thị kết quả của 3 lượt chạy gần nhất tương ứng với:
  1. Mô hình **Quốc Huy — Qwen2-VL-2B (QLoRA)**.
  2. Mô hình **LoRA Checkpoint 5500 (LoRA)**.
  3. Mô hình **Nhật Hào — DoRA Checkpoint 3315 (DoRA)**.
- **Bảng Lịch Sử Toàn Diện**: Ghi nhận chi tiết thời gian, hình ảnh, mô hình, nhãn dự đoán, độ tin cậy (%) và thời gian thực thi của mọi lượt test.
- **Nút "Xóa lịch sử"**: Cho phép làm mới dữ liệu đối chiếu bất cứ lúc nào.

#### 📚 Tab 7: Khám Phá Tập Dữ Liệu (Dataset Explorer)
- Duyệt qua danh mục 12 dòng tế bào máu.
- Xem mô tả sinh học, đặc điểm nhân, bào tương và thư viện ảnh mẫu thực tế từ tập dữ liệu.

---

## 🏋️ Hướng Dẫn Huấn Luyện Mô Hình

### 1. Huấn luyện Mô hình Phân loại Qwen-VL (DoRA / QLoRA)
Các script huấn luyện được lưu trữ trong thư mục [`training/training_qwen/`](file:///d:/code_QV/code/training/training_qwen/):

```bash
# Chạy script huấn luyện DoRA trên tập Dataset-Crop:
python training/training_qwen/23004023_train.py
```
*Hoặc mở và chạy file Jupyter Notebook:* [`TrainNhatHao.ipynb`](file:///d:/code_QV/code/TrainNhatHao.ipynb).

### 2. Huấn luyện Mô hình Phát hiện YOLO
Các script huấn luyện YOLO được lưu trữ trong thư mục [`training/training_yolo/`](file:///d:/code_QV/code/training/training_yolo/):

```bash
# Huấn luyện YOLO trên tập dữ liệu BCCD:
python training/training_yolo/train.py
```

---

## ❓ Câu Hỏi Thường Gặp & Xử Lý Sự Cố (FAQ)

### Q1: Tại sao lần đầu up ảnh phân loại mất khoảng 25–35 giây, nhưng các lần tiếp theo lại chạy rất nhanh (<0.5 giây)?
> **Trả lời**: Đây là cơ chế **In-Memory Caching (Keep-Alive)** chuẩn mực của hệ thống.
> - **Lần đầu tiên**: Backend phải đọc file weights (~3–4GB) từ ổ cứng và nạp vào VRAM của GPU (hiển thị thanh `Loading weights: 100%|...| 824/824`).
> - **Các lần sau**: Model đã nằm sẵn trong bộ nhớ GPU, server chỉ việc đưa ảnh vào tính toán ngay lập tức mà không cần nạp lại từ ổ cứng.
> - Hệ thống chỉ nạp lại weights khi bạn **đổi sang model khác**, **đổi thiết bị tính toán (GPU/CPU)** hoặc **khởi động lại server**.

### Q2: Gặp lỗi tràn bộ nhớ `CUDA out of memory` (OOM) khi phân loại?
> **Khắc phục**:
> 1. Đảm bảo bạn đang chọn chất lượng nén: **`4-bit (GPU CUDA — Tối ưu VRAM)`**.
> 2. Đóng các ứng dụng đồ họa nặng khác đang chiếm VRAM GPU.
> 3. Nếu máy không có GPU $\ge$ 4GB VRAM, hãy chuyển sang thiết bị **`CPU`** với nén **`Full 32-bit`**.

### Q3: Muốn đổi cổng (Port) chạy Web Server sang cổng khác?
> **Khắc phục**: Mở file [`run.py`](file:///d:/code_QV/code/run.py) và sửa tham số `port=8000` thành cổng mong muốn (ví dụ `port=8080`), sau đó chạy lại `python run.py`.

---

## 👥 Tác Giả & Đóng Góp
- **Sinh viên thực hiện**: Hồ Nhật Hào (23004023) & Nguyễn Quốc Huy
- **Đề tài**: Nghiên cứu & Ứng dụng Vision-Language Models kết hợp XAI trong Phân tích Tế bào Máu Vi thể.
- **Trường / Đơn vị**: Khoa Công nghệ Thông tin.
