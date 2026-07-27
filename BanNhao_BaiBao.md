# TRIỂN KHAI MÔ HÌNH QWEN CHO BÀI TOÁN PHÁT HIỆN VÀ PHÂN LOẠI TẾ BÀO KẾT HỢP XAI

**Nguyễn Quốc Vinh, Hồ Nhật Hào, Lê Trần Quốc Huy**
Khoa Công nghệ Thông tin, Trường Đại học Sư phạm Kỹ thuật Vĩnh Long
Email: 23004008@st.vlute.edu.vn, 23004023@st.vlute.edu.vn, 23004050@st.vlute.edu.vn

**TÓM TẮT**— Phát hiện và phân loại tế bào máu là một bài toán quan trọng trong chẩn đoán y sinh, giúp hỗ trợ đắc lực cho các bác sĩ trong việc chẩn đoán các bệnh lý về máu một cách nhanh chóng và chính xác. Nghiên cứu này đề xuất phương pháp kết hợp mô hình YOLO26 cho bài toán phát hiện vị trí tế bào trên tập dữ liệu BCCD (Blood Cell Count and Detection Dataset), phối hợp cùng mô hình ngôn ngữ lớn đa phương tiện Qwen2.5-VL đóng vai trò bộ phân loại học sâu để phân tích chi tiết đặc điểm tế bào. Nhằm giải quyết tính chất "hộp đen" của các mô hình học sâu truyền thống và nâng cao mức độ tin cậy trong các ứng dụng y tế, chúng tôi tích hợp các phương pháp trí tuệ nhân tạo có thể giải thích được (Explainable AI - XAI), bao gồm HiresCAM, XGrad-CAM, EigenCAM và Integrated Gradients. Các kỹ thuật XAI này cung cấp các bản đồ nhiệt trực quan hóa vùng chú ý, giúp giải thích rõ ràng căn cứ ra quyết định của mô hình. Kết quả thử nghiệm trên tập dữ liệu tế bào thực tế chứng minh mô hình đề xuất đạt hiệu năng vượt trội cả về độ chính xác phân đoạn và tính minh bạch của kết quả dự báo, mở ra nhiều triển vọng ứng dụng thực tiễn trong hỗ trợ chẩn đoán lâm sàng.

**Từ khóa**— Phát hiện tế bào, Qwen2.5-VL, YOLO26, Explainable AI (XAI), BCCD dataset.

---

## I. GIỚI THIỆU

### A. Giới thiệu bài toán
Việc phân tích và đếm tế bào máu đóng vai trò quan trọng trong việc chẩn đoán các bệnh lý huyết học như thiếu máu, nhiễm trùng, và đặc biệt là các bệnh ung thư máu (Leukemia). Phương pháp thủ công đếm tế bào dưới kính hiển vi thường tốn nhiều thời gian và phụ thuộc lớn vào kinh nghiệm của kỹ thuật viên. Do đó, việc ứng dụng Trí tuệ nhân tạo (AI) vào tự động hóa quy trình đếm và phân loại tế bào máu đang thu hút được nhiều sự quan tâm.

Nghiên cứu này đề xuất một luồng xử lý (pipeline) toàn diện từ khâu phát hiện (Detection) các tế bào trên ảnh toàn cảnh đến khâu phân loại chi tiết (Classification) từng loại bạch cầu. Đặc biệt, nghiên cứu tích hợp Trí tuệ nhân tạo có thể giải thích (XAI) nhằm trực quan hóa các vùng đặc trưng mà mô hình tập trung vào, qua đó nâng cao độ tin cậy của AI trong các ứng dụng y tế.

### B. Những nghiên cứu liên quan
Trong những năm gần đây, thị giác máy tính trong y tế đã phát triển vượt bậc nhờ các mô hình phát hiện đối tượng thời gian thực như YOLO (Redmon et al.) và các kiến trúc mạng tích chập tiên tiến. Tập dữ liệu BCCD (Blood Cell Count and Detection) đã trở thành một chuẩn so sánh (benchmark) phổ biến cho việc đánh giá các thuật toán phát hiện tế bào máu (RBC, WBC, Platelets).

Tuy nhiên, các phương pháp phân loại tế bào truyền thống thường coi mô hình học sâu như một "hộp đen" (black box), thiếu khả năng giải thích nguyên nhân dẫn đến quyết định phân loại. Sự xuất hiện của các phương pháp CAM (Class Activation Mapping) như Grad-CAM (Selvaraju et al.), XGrad-CAM (Fu et al.), HiresCAM (Draelos et al.) và Integrated Gradients (Sundararajan et al.) đã tạo cơ sở lý thuyết vững chắc cho việc giải thích mô hình (XAI).

Bên cạnh đó, các mô hình ngôn ngữ đa phương tiện thị giác (Vision-Language Models - VLMs) như series Qwen2.5-VL (Alibaba Group) đã chứng minh khả năng vượt trội trong việc hiểu ảnh và sinh lời giải thích bằng ngôn ngữ tự nhiên. Việc kết hợp fine-tuning LoRA (Low-Rank Adaptation) trên mô hình VLM cùng các kỹ thuật XAI trực quan mang lại hướng tiếp cận toàn diện chưa từng có cho phân tích tế bào y học.

---

## II. CƠ SỞ LÝ THUYẾT

### A. Nhận diện tế bào máu trên tập BCCD
Mô hình được huấn luyện để phát hiện 3 lớp đối tượng chính trong ảnh phết máu: Hồng cầu (RBC), Bạch cầu (WBC) và Tiểu cầu (Platelets) dựa trên tập dữ liệu BCCD.

### B. Các phương pháp nền tảng

#### 1. Mô hình phát hiện đối tượng YOLO26
Bài toán phát hiện đối tượng được giải quyết bằng mô hình YOLO26 (kiến trúc NMS-Free). Mô hình định vị xuất sắc các tế bào nhỏ và phân bố dày đặc như hồng cầu và tiểu cầu.

#### 2. Mô hình đa phương tiện thị giác-ngôn ngữ Qwen2.5-VL-3B
Bạch cầu sau khi được trích xuất (crop) sẽ được xử lý bởi 2 nhánh mô hình:
- **QwenCellClassifier (EfficientNet-B2 backbone):** Phân loại 12 lớp tế bào kết hợp với module XAI trực quan hóa bản đồ nhiệt.
- **Qwen2.5-VL-3B-Instruct Fine-tuned (LoRA):** Mô hình đa phương tiện Vision-Language Model (VLM) được tinh chỉnh bằng kỹ thuật LoRA (Low-Rank Adaptation, $r=16, \alpha=32$) áp dụng lên các ma trận trọng số chú ý và FFN (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Mô hình nhận ảnh tế bào trực tiếp và sinh ra văn bản phân loại cùng lời giải thích y khoa tự động theo ngôn ngữ tự nhiên.

#### 3. Các lý thuyết giải thích mô hình (XAI)
Để diễn giải quyết định của mạng nơ-ron đen (black-box), bốn phương pháp XAI được triển khai:
- **HiresCAM và XGrad-CAM:** Hiển thị bản đồ nhiệt (heatmap) dựa trên gradient của lớp tích chập cuối cùng.
- **EigenCAM:** Sử dụng thành phần chính của các đặc trưng học được để định vị đối tượng.
- **Integrated Gradients (IG):** Tính toán tích phân gradient từ ảnh gốc đến ảnh nền (baseline) để chỉ ra mức độ đóng góp của từng điểm ảnh.

---

## III. PHƯƠNG PHÁP ĐỀ XUẤT

Hệ thống đề xuất hoạt động theo một luồng xử lý hai giai đoạn (Two-Stage Pipeline) khép kín kết hợp giải thích đa phương thức:

1. **Giai đoạn 1 - Định vị & Trích xuất (Detection & Cropping):**
   - Ảnh phết máu đầu vào được đưa qua mô hình **YOLO26**. Mô hình dự đoán tọa độ hộp giới hạn (Bounding Box) và nhãn cho 3 lớp: Hồng cầu (RBC), Bạch cầu (WBC) và Tiểu cầu (Platelets).
   - Các vùng chứa tế bào (đặc biệt là Bạch cầu) được cắt tự động (Auto-crop) và tiền xử lý kích thước chuẩn để chuẩn bị cho giai đoạn phân loại chuyên sâu.

2. **Giai đoạn 2 - Phân loại Chuyên sâu & Giải thích Đa phương thức (Classification & XAI):**
   - **Phân loại ảnh & Trực quan Heatmap (EfficientNet-B2 + XAI Module):** Ảnh tế bào được phân loại vào 12 lớp hình thái tế bào máu chi tiết. Đồng thời, 4 thuật toán XAI (HiresCAM, XGrad-CAM, EigenCAM, Integrated Gradients) trích xuất gradient từ lớp tích chập cuối để tạo bản đồ nhiệt biểu diễn mức độ quan tâm của mô hình vào vùng nhân và bào tương tế bào.
   - **Sinh Chẩn đoán & Lời giải thích Y khoa (Qwen2.5-VL-3B LoRA):** Ảnh tế bào được chuyển tới mô hình Vision-Language Qwen2.5-VL-3B đã được fine-tune bằng LoRA. Qwen sinh ra phản hồi ngôn ngữ tự nhiên gồm: Tên loại tế bào, phân tích đặc điểm nhân/bào tương và đưa ra kết luận chẩn đoán lâm sàng.

3. **Giao diện Tương tác Triển khai (Gradio Web Interface):**
   - Toàn bộ pipeline được tích hợp lên giao diện Web Gradio thân thiện, hỗ trợ hiển thị đồng thời kết quả phát hiện vị trí, phân loại tế bào, so sánh 4 bản đồ nhiệt XAI và văn bản chẩn đoán y khoa từ Qwen2.5-VL.

---

## IV. KẾT QUẢ THỰC NGHIỆM

### 4.1. Dữ liệu thực nghiệm
- Mô hình YOLO26 được huấn luyện trên tập dữ liệu BCCD (Blood Cell Count and Detection) với nhãn dạng JSON Supervisely (205 ảnh huấn luyện).
- Mô hình phân loại được huấn luyện trên tập dữ liệu tế bào máu đã crop (gồm 12 lớp với 17,000+ mẫu dữ liệu), chia tỷ lệ 70% Train, 15% Val, 15% Test.

### 4.2. Kết quả phát hiện đối tượng (Detection)
Sau quá trình huấn luyện, mô hình YOLO26 đạt được các chỉ số ấn tượng trên tập validation:
- mAP@50: **89.72%**
- mAP@50-95: **62.55%**

### 4.3. Kết quả phân loại (Classification) & Tinh chỉnh Qwen2.5-VL
Kết quả đánh giá trên các mô hình phân loại:
- **QwenCellClassifier (EfficientNet-B2):** Đạt độ chính xác tổng thể (Accuracy) **97.03%** và F1-score (Weighted) **0.9703** trên tập kiểm thử. Đối với các loại tế bào khó phân biệt như Eosinophil, Lymphocyte hay Monocyte, F1-score duy trì ở mức trên 0.95.
- **Qwen2.5-VL-3B LoRA (`checkpoint-1500`):** Sau 1,500 bước tinh chỉnh (global step 1500), mô hình Vision-Language đạt độ chính xác đánh giá **Eval Accuracy: 97.27%** và **Eval Loss: 0.2841**. Mô hình sinh lời giải thích y khoa tự động chính xác dựa trên các đặc trưng hình thái học nhân và bào tương của tế bào.

### 4.4. Đánh giá XAI và Faithfulness Test
Bản đồ nhiệt sinh ra từ các phương pháp XAI cho thấy mô hình tập trung chính xác vào vùng nhân (nucleus) của bạch cầu để đưa ra quyết định thay vì học các nhiễu nền. Phương pháp XGrad-CAM và Integrated Gradients cho kết quả sắc nét nhất. Kiểm thử Faithfulness Deletion Test chứng minh rằng khi xóa dần các vùng ảnh có điểm số XAI cao, độ chính xác của mô hình giảm mạnh, khẳng định các phương pháp XAI đã đánh giá đúng đặc trưng quan trọng.

---

## KẾT LUẬN
Nghiên cứu đã triển khai thành công một hệ thống toàn diện cho việc phát hiện và phân loại tế bào máu. Sự kết hợp giữa YOLO26, EfficientNet và Qwen2.5-VL-3B không chỉ mang lại độ chính xác cao (97%) mà còn cung cấp khả năng diễn giải mạnh mẽ thông qua XAI và ngôn ngữ tự nhiên. Giao diện Web Gradio được xây dựng thành công, cho phép triển khai ứng dụng vào thực tế, đóng góp công cụ hỗ trợ đắc lực cho các bác sĩ huyết học.

---

## V. TÀI LIỆU THAM KHẢO
1. Ultralytics, "YOLO Vision Models", 2024.
2. Qwen Team, "Qwen2.5-VL Technical Report", Alibaba Group, 2024.
3. Ramprasaath R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV, 2017.
4. Mukund Sundararajan et al., "Axiomatic Attribution for Deep Networks" (Integrated Gradients), ICML, 2017.
5. Dữ liệu BCCD Dataset: https://www.kaggle.com/datasets/orvile/bccd-blood-cell-count-and-detection-dataset

---

# DEVELOPMENT AND DEPLOYMENT OF A QWEN-BASED MODEL FOR CELL DETECTION AND CLASSIFICATION WITH EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI)
**Nguyen Quoc Vinh, Ho Nhat Hao, Le Tran Quoc Huy**

**ABSTRACT**— Blood cell detection and classification play a critical role in biomedical diagnostics, assisting physicians in making fast and accurate clinical decisions regarding hematological disorders. This study proposes an integrated framework that leverages the YOLO26 object detection model to identify cell locations using the BCCD dataset, combined with the Qwen2.5-VL multimodal vision-language model acting as a deep learning classifier to analyze detailed cellular features. To address the "black box" nature of traditional deep neural networks and build clinical trust, we incorporate Explainable AI (XAI) methodologies, specifically HiresCAM, XGrad-CAM, EigenCAM, and Integrated Gradients. These XAI techniques generate visual heatmaps of the model's focus, providing clear clinical explanations for its decision-making process. Experimental results on the designated dataset demonstrate that our proposed model achieves superior performance in both segmentation accuracy and interpretability of predictions, highlighting its potential for practical integration into clinical decision-support systems.

**Keywords**— Cell detection, Qwen2.5-VL, YOLO26, Explainable AI (XAI), BCCD dataset.
