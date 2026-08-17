# HemoAI: Hệ thống Phát hiện và Phân loại Tế bào Máu kết hợp XAI

Dự án này triển khai một pipeline hoàn chỉnh ứng dụng Deep Learning để phân tích hình ảnh huyết học, bao gồm phát hiện tế bào (YOLO), phân loại chi tiết (QWen-VL) và trực quan hóa quyết định mô hình (Explainable AI - XAI). Dự án cung cấp kèm giao diện Web Demo trực quan, dễ dàng theo dõi.

## 🌟 Tính năng chính

1. **Phát hiện Tế bào (Detection)**: Sử dụng mô hình **YOLO26** (kiến trúc YOLO) để định vị và phân loại các tế bào cơ bản (Hồng cầu - RBC, Bạch cầu - WBC, Tiểu cầu - Platelets) từ tiêu bản máu thô.
2. **Phân loại Chuyên sâu (Classification)**: Sử dụng mô hình ngôn ngữ lớn đa phương thức **QWen2.5-VL** (được fine-tune bằng QLoRA) để phân loại từng tế bào (sau khi crop) thành 12 loại tế bào máu chuyên sâu (Basophil, Eosinophil, Lymphocyte, Monocyte,...).
3. **Giải thích Mô hình (XAI - Explainable AI)**: Tích hợp nhiều phương pháp trực quan hóa (HiresCAM, XGrad-CAM, EigenCAM, Integrated Gradients) để giải thích vùng ảnh tác động đến quyết định phân loại của QWen.
4. **Pipeline Tích hợp**: Chạy xuyên suốt một pipeline: Tải ảnh gốc -> YOLO phát hiện & cắt ảnh -> QWen phân loại -> XAI sinh heatmap.
5. **Giao diện Web Trực quan**: Backend viết bằng FastAPI, Frontend xây dựng bằng HTML/CSS/JS thuần, cung cấp Dashboard tổng quan, công cụ tương tác mạnh mẽ.

---

## 📁 Cấu trúc Dự án

```text
chuyende/
├── Dataset-Crop/         # Tập dữ liệu 12 loại tế bào để huấn luyện QWen
├── BCCD/                 # Tập dữ liệu phát hiện tế bào (người dùng tự thêm)
├── src/
│   ├── api/
│   │   └── server.py     # Backend FastAPI server
│   └── models/
│       ├── qwen_classifier.py  # Inference QWen2.5-VL
│       ├── yolo_detector.py    # Inference YOLO26
│       └── xai_engine.py       # Thuật toán XAI
├── web/
│   ├── index.html        # Giao diện chính
│   ├── styles.css        # File CSS
│   └── app.js            # File xử lý logic Web
├── train_yolo.py         # Script huấn luyện YOLO trên tập BCCD
├── train_qwen.py         # Script fine-tune QWen trên tập Dataset-Crop
├── requirements.txt      # Các thư viện Python cần thiết
├── run.py                # Script khởi chạy Web Server
└── README.md             # File hướng dẫn này
```

---

## 🛠 Hướng dẫn Cài đặt & Chuẩn bị Dữ liệu

### 1. Cài đặt Môi trường
Yêu cầu Python 3.9 trở lên. Mở Terminal và chạy lệnh:
```bash
pip install -r requirements.txt
pip install ultralytics transformers peft bitsandbytes accelerate datasets qwen_vl_utils
```

### 2. Chuẩn bị Dữ liệu (Dành cho việc Huấn luyện)
Để huấn luyện mô hình, bạn cần chuẩn bị 2 tập dữ liệu và đặt chúng ở thư mục gốc của dự án:
- **Tập Dataset BCCD**: Tải về và giải nén thành thư mục tên `BCCD`. Tập dữ liệu này chứa ảnh và nhãn yolo cho việc detect. (Script sẽ tự tạo `bccd.yaml` tương ứng).
- **Tập Dataset-Crop**: Chứa 12 thư mục con (BA, BNE, EO,...), mỗi thư mục chứa ảnh của các tế bào tương ứng. Thư mục này dùng để fine-tune QWen.

---

## 🚀 Hướng dẫn Sử dụng

### 1. Huấn luyện Mô hình (Training)

**Huấn luyện YOLO26 (Phát hiện 3 loại tế bào cơ bản):**
```bash
python train_yolo.py
```
*Kết quả (weights) sẽ được lưu tại: `outputs/yolo26_bccd/weights/best.pt`*

**Fine-tune QWen-VL (Phân loại 12 loại tế bào chuyên sâu):**
```bash
python train_qwen.py
```
*Kết quả (adapter model) sẽ được lưu tại: `outputs/qwen_blood_cell/`*

### 2. Khởi chạy Giao diện Web (Demo)
Khi bạn đã có file weights (hoặc muốn sử dụng chế độ Mocking Demo - mô phỏng), hãy chạy lệnh sau để khởi động Server FastAPI:
```bash
python run.py
```
Sau đó, mở trình duyệt web và truy cập vào địa chỉ: **http://127.0.0.1:8000**

Giao diện Web cho phép bạn:
- Tải ảnh tiêu bản lên để chạy YOLO phát hiện tế bào.
- Tải ảnh từng tế bào đơn lẻ để chạy QWen phân loại.
- Sinh Heatmap XAI từ các ảnh tế bào.
- Xem mô phỏng Pipeline đầy đủ từ đầu đến cuối.

---

## 🧩 Các Lớp (Classes) Được Hỗ trợ

### YOLO (3 Lớp)
1. **RBC**: Hồng cầu
2. **WBC**: Bạch cầu
3. **Platelets**: Tiểu cầu

### QWen (12 Lớp Bạch cầu & Tế bào dòng tủy)
* BA (Basophil)
* BNE (Band Neutrophil)
* EO (Eosinophil)
* ERB (Erythroblast)
* LY (Lymphocyte)
* MMY (Metamyelocyte)
* MO (Monocyte)
* MY (Myelocyte)
* MYO (Myeloblast)
* PLT (Platelet)
* PMY (Promyelocyte)
* SNE (Segmented Neutrophil)
