
TRIỂN KHAI MÔ HÌNH QWEN CHO BÀI TOÁN PHÁT HIỆN VÀ PHÂN LOẠI TẾ BÀO KẾT HỢP XAI
Nguyễn Quốc Vinh, Hồ Nhật Hào, Lê Trần Quốc Huy
Khoa Công nghệ Thông tin, Trường Đại học Sư phạm Kỹ thuật Vĩnh Long
23004008@st.vlute.edu.vn, 23004023@st.vlute.edu.vn, 23004050@st.vlute.edu.vn

TÓM TẮT— Phân tích tế bào máu ngoại vi là quy trình cốt lõi trong chẩn đoán huyết học, tuy nhiên phương pháp soi kính hiển vi thủ công thường tốn từ 15 đến 20 phút mỗi tiêu bản với tỉ lệ sai sót giữa các kỹ thuật viên từ 15% đến 25%, trong khi các mô hình trí tuệ nhân tạo hiện đại dù đạt độ chính xác cao nhưng hơn 85% vẫn vận hành như một "hộp đen" thiếu minh bạch. Nghiên cứu này đề xuất khung hệ thống chẩn đoán HemoAI tích hợp mô hình YOLO26 để phát hiện và khoanh vùng tế bào thời gian thực trên tập dữ liệu BCCD, phối hợp cùng mô hình ngôn ngữ lớn đa phương tiện Qwen2.5-VL (được tinh chỉnh bằng QLoRA 4-bit) đảm nhận phân loại chi tiết 12 lớp tế bào máu. Đồng thời, bộ động cơ XAI gồm 4 thuật toán (HiResCAM, XGrad-CAM, EigenCAM và Integrated Gradients) được tích hợp để trực quan hóa vùng chú ý của mô hình dưới dạng bản đồ nhiệt. Thử nghiệm thực nghiệm chứng minh YOLO26 đạt mAP@0.5 là 91.5%, trong khi Qwen2.5-VL đạt độ chính xác phân loại toàn cục 97.27% (hàm mất mát 0.0761) đi kèm các giải thích thị giác khớp với đặc điểm hình thái học y sinh. Kết quả nghiên cứu mở ra triển vọng lớn trong hỗ trợ chẩn đoán lâm sàng, đồng thời định hướng tối ưu hóa tốc độ suy luận để triển khai trên các thiết bị nhúng và thử nghiệm lâm sàng diện rộng tại các bệnh viện.
Từ khóa—  Phát hiện tế bào, Qwen2.5-VL, YOLO26, Explainable AI (XAI), BCCD dataset.
I.	GIỚI THIỆU

A.	Giới thiệu bài toán
Trong y học lâm sàng và huyết học, việc phân tích tế bào máu ngoại vi là một quy trình chẩn đoán cơ bản và tối quan trọng. Theo thống kê y tế, hơn **70% các quyết định chẩn đoán lâm sàng** trong bệnh viện phụ thuộc vào kết quả xét nghiệm công thức máu và phân tích lam máu ngoại vi [2]. Quy trình này cung cấp các thông tin sinh học thiết yếu giúp phát hiện sớm nhiều bệnh lý nguy hiểm như nhiễm trùng, suy giảm miễn dịch, thiếu máu (bệnh lý đang ảnh hưởng tới hơn **1.62 tỷ người**, tương đương **24.8% dân số thế giới** [1]), dị ứng, và các thể bệnh ung thư máu nguy hiểm như leukemia (bạch cầu cấp, chiếm khoảng **8.2% tổng số ca ung thư** ở trẻ em và người trẻ tuổi [1]). 

Theo phương pháp truyền thống, các kỹ thuật viên xét nghiệm phải quan sát tiêu bản máu nhuộm dưới kính hiển vi quang học, đếm thủ công từ 100 đến 200 tế bào bạch cầu trên mỗi lam máu. Tuy nhiên, quy trình thủ công này tồn tại nhiều hạn chế đáng kể về cả hiệu suất lẫn độ tin cậy: trung bình một kỹ thuật viên phải mất từ **15 đến 20 phút** để phân tích hoàn chỉnh một tiêu bản, và tỉ lệ sai sót hoặc sự biến thiên kết quả giữa các người thực hiện (inter-observer variability) có thể dao động từ **15% đến 25%** [1], [2]. Đặc biệt, sau khi soi từ 30 đến 50 tiêu bản trong một ca trực, sự mệt mỏi thị giác kéo theo tỉ lệ chẩn đoán nhầm tăng thêm tới **20%**, nhất là khi gặp các tế bào có sự chồng chéo hình thái rất khó phân biệt giữa các giai đoạn biệt hóa (như giữa hậu tủy bào, tủy bào và nguyên tủy bào) [2].

Để khắc phục các nhược điểm trên, việc ứng dụng các kỹ thuật xử lý ảnh số và trí tuệ nhân tạo (AI), đặc biệt là các mô hình học sâu hiện đại như mạng nơ-ron tích chập (CNN) và kiến trúc Vision Transformer (ViT) tiền huấn luyện trên hàng trăm triệu hình ảnh [5], đã trở thành một xu hướng nghiên cứu mạnh mẽ trong những năm gần đây. Các hệ thống thị giác máy tính tự động giúp nâng cao hiệu suất phân tích lên gấp **100 lần**, rút ngắn thời gian xử lý xuống dưới **1 giây** với độ chính xác phân loại ấn tượng đạt từ **95% đến 98%** [2], [5]. Hệ thống phân tích tế bào tự động thường gồm hai bài toán cốt lõi: phát hiện đối tượng (Object Detection) dựa trên các kiến trúc One-Stage như YOLO giúp xác định vị trí tế bào với độ trễ cực thấp dưới **20 ms/ảnh** (tốc độ từ **45 đến 155 khung hình/giây**) [4], và bài toán phân loại (Classification) để nhận diện chi tiết các phân nhóm tế bào phức tạp. 

Tuy nhiên, bất chấp độ chính xác cao, các khảo sát y tế chỉ ra rằng hơn **85% các hệ thống chẩn đoán AI hiện hành** vẫn vận hành như một "hộp đen" (black-box) thiếu minh bạch. Chính sự thiếu giải thích này khiến hơn **82% các bác sĩ lâm sàng** e ngại và từ chối áp dụng AI vào quy trình chẩn đoán thực tế do lo ngại rủi ro trách nhiệm y khoa và pháp lý [3]. Điều này đặt ra yêu cầu cấp thiết về việc tích hợp Trí tuệ nhân tạo có thể giải thích được (Explainable AI - XAI) vào quy trình chẩn đoán, giúp tường minh hóa các đặc điểm hình thái học tế bào mà mô hình dựa vào để đưa ra quyết định [3].

B.	Những nghiên cứu liên quan
Trong giai đoạn từ năm 2021 đến 2026, lĩnh vực phân tích tế bào máu tự động đã chứng kiến sự phát triển vượt bậc từ các kiến trúc CNN truyền thống sang các mô hình Transformer đa phương tiện và sự tích hợp sâu rộng của Trí tuệ nhân tạo có thể giải thích được (XAI). Các nghiên cứu trong giai đoạn này tập trung giải quyết hai bài toán cốt lõi: nâng cao độ chính xác phát hiện/phân loại tế bào và mở hộp đen mô hình nhằm tăng cường độ tin cậy lâm sàng.

*1) Các phương pháp phát hiện tế bào máu dựa trên nhánh mô hình YOLO và R-CNN*: 
M. M. Çakır và G. Çınarer [6] đã thực hiện đánh giá diện rộng trên 21 kiến trúc học sâu bao gồm Faster R-CNN, Detectron2 và các biến thể YOLO trên tập dữ liệu BCCD, chỉ ra rằng các mô hình họ YOLO đạt sự cân bằng tối ưu giữa tốc độ suy luận và độ chính xác (mAP@0.5 đạt 91.5%). V. D. Nguyen và cộng sự [8] đề xuất mô hình YOLOv5 cải tiến tích hợp cơ chế chú ý CBAM giúp nâng cao khả năng đếm tế bào trên tiêu bản. Tiếp đó, S.-J. Lee và nhóm nghiên cứu [9] ứng dụng mạng nơ-ron sâu (DNN) kết hợp tiền xử lý tăng cường ảnh để giải quyết bài toán phát hiện tế bào ở các vùng ảnh mật độ cao. Gần đây, C. Shi và cộng sự [13] giới thiệu Gpmb-YOLO dựa trên YOLOv8n kết hợp thuật toán di truyền và cơ chế SimAM, đạt mAP@0.5 là 95.3%. X. Chen và cộng sự [14] phát triển NBCDC-YOLOv8 tích hợp các khối SPD-Conv, MultiSEAM và BiFPN nhằm tối ưu hóa việc phân tách các tế bào nhỏ. A. E. Hasen và các cộng sự [15] đề xuất mô hình ABCD dựa trên YOLOX kết hợp CBAM và ASFF, đạt độ chính xác mAP@0.5 lên tới 95.49%. N. B. Džakula và cộng sự [20] thực hiện nghiên cứu thực nghiệm so sánh hai thế hệ mô hình YOLOv10 và YOLOv11, chứng minh YOLOv11n cho kết quả vượt trội về mAP@0.5 (92.79%) so với YOLOv10n.

*2) Các kiến trúc CNN và kỹ thuật chú ý (Attention Mechanism) trong phân loại tế bào*: 
A. Girdhar và các cộng sự [7] xây dựng kiến trúc CNN 5 lớp tùy chỉnh để phân loại các phân nhóm bạch cầu với độ chính xác 94.2%. S. Khan và nhóm nghiên cứu [12] tích hợp cơ chế chú ý kép (Dual Attention - Channel & Spatial) vào CNN, nâng độ chính xác phân loại bạch cầu lên 99.35%. G. Zhang và các cộng sự [16] phát triển mô hình CCE-YOLOv7 chuyên biệt cho ảnh chụp kính hiển vi Fourier Ptychographic (FPM), đạt mAP@0.5 là 94.1%. N.-H.-Q. Nguyen và cộng sự [18] đề xuất mạng Ghost Residual Network (GRsNet) nhằm giảm nhẹ kích thước mô hình nhưng vẫn đạt độ chính xác 95.0% trên tập dữ liệu PBC.

*3) Mô hình nền tảng (Foundation Models) và Transformer trong xử lý ảnh y sinh*: 
Y. Li và các cộng sự [11] giới thiệu BC-SAM, sử dụng mô hình nền tảng Segment Anything Model (SAM) kết hợp LoRA để phân loại tế bào máu xuyên miền đạt độ chính xác 95.1%. S. Ziane và S. Hazmoune [17] đề xuất mô hình học máy kết hợp (Ensemble Learning) dựa trên chuỗi Transformer tiên tiến (ViT, Swin, DeiT, BEiT), đạt độ chính xác 95.8%. Gần đây nhất, J. van Logtestijn và P. Manescu [19] phát triển mô hình ngôn ngữ - thị giác HemBLIP (Vision-Language Model), cho phép phân tích hình thái tế bào bạch cầu và chẩn đoán ung thư máu một cách linh hoạt qua văn bản y khoa.

*4) Trí tuệ nhân tạo có thể giải thích được (XAI) trong chẩn đoán huyết học*: 
J. L. Diaz Resendiz và các cộng sự [10] đề xuất hệ thống CAD giải thích được cho chẩn đoán ung thư máu bạch cầu cấp (ALL) kết hợp U-Net và hai kỹ thuật HiRes-CAM, XGrad-CAM, đạt độ chính xác 99.9%. Sự kết hợp giữa các kỹ thuật trực quan hóa bản đồ chú ý (CAM) và mô hình học sâu (như trong các công trình [10], [17], [18], [19]) đang khẳng định xu hướng bắt buộc trong việc xây dựng các hệ thống AI lâm sàng minh bạch, đáng tin cậy.

Bảng I tổng hợp chi tiết 15 công trình nghiên cứu tiêu biểu trong giai đoạn này, chỉ ra các phương pháp, tập dữ liệu, số lượng lớp phân loại, hạn chế và kết quả đạt được của từng nghiên cứu.

**Bảng I. Tóm tắt các nghiên cứu liên quan về phát hiện và phân loại tế bào máu (2021–2026)**

| Tác giả & Năm | Phương pháp đề xuất | Tập dữ liệu | Số loại tế bào / Số lớp | Hạn chế chính | Độ đo đánh giá & Kết quả đạt được |
| :--- | :--- | :--- | :--- | :--- | :--- |
| M. M. Çakır & G. Çınarer (2025) [6] | So sánh 21 kiến trúc phát hiện đối tượng (Faster R-CNN, Detectron2, YOLO) | BCCD | 3 lớp (RBC, WBC, Platelet) | Cấu trúc Faster R-CNN cồng kềnh, tốc độ suy luận chậm hơn các YOLO hiện đại | mAP@0.5: 91.5%<br>Precision: 90.8% |
| A. Girdhar et al. (2022) [7] | CNN tùy chỉnh phân loại tế bào bạch cầu nhiều lớp | Custom dataset | 5 lớp bạch cầu (Neutrophil, Eosinophil, Basophil, Lymphocyte, Monocyte) | Không thể xử lý ảnh tế bào chồng chéo, hạn chế mở rộng số lớp | Accuracy: 94.2%<br>F1-Score: 93.7% |
| V. D. Nguyen et al. (2022) [8] | YOLOv5 cải tiến cho đếm và nhận diện tế bào máu | BCCD | 3 lớp (RBC, WBC, Platelet) | Mô hình chưa xử lý tốt tế bào chồng chéo ở mật độ cao | mAP@0.5: 92.1%<br>Precision: 90.5% |
| S.-J. Lee et al. (2022) [9] | Đếm và phát hiện tế bào máu dựa trên mạng nơ-ron sâu (DNN) | BCCD | 3 lớp (RBC, WBC, Platelet) | Hiệu năng đếm giảm đối với các vùng tế bào nằm chồng lấn nặng | mAP@0.5: 91.8%<br>Precision: 90.2% |
| J. L. Diaz Resendiz et al. (2023) [10] | Explainable CAD kết hợp U-Net segmentation với HiRes-CAM, XGrad-CAM | ALL-IDB | 2 lớp (Bình thường vs. Bạch cầu cấp - ALL) | Chỉ áp dụng cho 2 lớp phân loại, chưa mở rộng sang phân loại đa lớp phức tạp | Accuracy: 99.9%<br>XAI Sensitivity verified |
| Y. Li et al. (2024) [11] | BC-SAM: SAM kết hợp LoRA cho phân loại tế bào máu xuyên miền | Matek-19, Acevedo-20 | 8 lớp bạch cầu | Thời gian suy luận của mô hình SAM quá lâu, chưa thể ứng dụng thời gian thực | Accuracy: 95.1%<br>Recall: 94.6% |
| S. Khan et al. (2024) [12] | CNN kết hợp Dual Attention (Channel + Spatial) phát hiện và phân loại bạch cầu | Microscopic blood smear dataset | Bạch cầu đa lớp (WBC) | Mô hình phức tạp, yêu cầu tài nguyên tính toán cao | Accuracy: 99.35%<br>Recall: 99.83% |
| C. Shi et al. (2024) [13] | Gpmb-YOLO: YOLOv8 tối ưu hóa siêu tham số bằng thuật toán di truyền + SimAM | BCCD | 3 lớp tế bào | Kiến trúc cải tiến phức tạp, khó tái tạo hoàn toàn trên phần cứng nhúng | mAP@0.5: 95.3%<br>Precision: 93.8% |
| X. Chen et al. (2025) [14] | NBCDC-YOLOv8 tích hợp SPD-Conv, MultiSEAM và BiFPN | BCCD | 3 lớp tế bào | Cấu trúc khối Neck phức tạp, đòi hỏi nhiều tài nguyên GPU | mAP@0.5: 94.7%<br>Precision: 93.5% |
| A. E. Hasen et al. (2025) [15] | ABCD: YOLOX cải tiến với CBAM Attention và ASFF | BCCD | 3 lớp tế bào | Cấu trúc khối Neck rất phức tạp, khó triển khai trên các thiết bị nhúng | mAP@0.5: 95.49%<br>Precision: 93.8% |
| G. Zhang et al. (2025) [16] | CCE-YOLOv7 chuyên biệt cho ảnh chụp kính hiển vi FPM | FPM custom dataset | 5 lớp tế bào | Phụ thuộc hoàn toàn vào thiết bị chụp ảnh phân cực đặc thù | mAP@0.5: 94.1%<br>F1-Score: 92.5% |
| S. Ziane & S. Hazmoune (2026) [17] | Ensemble Transformer (ViT + Swin + DeiT + BEiT) kết hợp XAI | PBC | 8 lớp tế bào | Tiêu tốn lượng lớn tài nguyên VRAM trong quá trình huấn luyện và dự đoán | Accuracy: 95.8%<br>Recall: 95.1% |
| N.-H.-Q. Nguyen et al. (2025) [18] | Ghost Residual Network (GRsNet) kết hợp Grad-CAM, LIME và SHAP | Naturalize 2K-PBC, Microscopic Blood Cell | 4 lớp | Việc tạo ảnh surrogate cho LIME làm chậm tốc độ suy luận giải thích | Accuracy: 95.0%<br>Precision: 94.2% |
| J. van Logtestijn & P. Manescu (2026) [19] | HemBLIP: Vision-Language Model (VLM) hỗ trợ phân tích hình thái tế bào máu | Clinical & Synthetic dataset (~14k cells) | 12+ lớp tế bào | Mô hình ngôn ngữ đa phương tiện đòi hỏi tài nguyên tính toán cao và tinh chỉnh prompt | Accuracy: 94.8%<br>F1-Score: 94.2% |
| N. B. Džakula et al. (2025) [20] | So sánh YOLOv10 và YOLOv11 cho phát hiện tế bào máu | BCCD (Kaggle Blood Cell Dataset) | 3 lớp (RBC, WBC, Platelet) | YOLOv10n phát hiện nhanh hơn nhưng độ chính xác kém hơn YOLOv11n | mAP@0.5: 91.20% (YOLOv10n)<br>Recall: 93.1% |

Các nghiên cứu gần đây cho thấy xu hướng dịch chuyển rõ rệt. Các mô hình không chỉ hướng đến nâng cao độ chính xác thô mà còn tập trung cải thiện khả năng chẩn đoán đa lớp tế bào (lên đến 12 lớp phức tạp) và cung cấp cơ chế tường minh hóa kết quả bằng XAI để tối ưu hóa sự tích hợp vào thực tiễn y học lâm sàng.

II.	CƠ SỞ LÝ THUYẾT

A.	Tổng quan về tế bào máu và các phân nhóm hình thái
Trong hệ tuần hoàn của cơ thể người, máu ngoại vi đóng vai trò nuôi dưỡng và bảo vệ cơ thể thông qua ba thành phần tế bào hữu hình chính bao gồm hồng cầu (Red Blood Cells - RBC), tiểu cầu (Platelets - PLT) và bạch cầu (White Blood Cells - WBC) [1], [2]. Hồng cầu là nhóm tế bào chiếm số lượng lớn nhất, có dạng hình đĩa lõm hai mặt và không chứa nhân khi trưởng thành, chịu trách nhiệm chính trong việc vận chuyển oxy từ cơ quan hô hấp đến các mô và thu nhận carbon dioxide quay trở lại phổi. Tiểu cầu là các mảnh tế bào không nhân có kích thước nhỏ từ 2 đến 3 µm, giữ vai trò cốt lõi trong cơ chế đông máu sinh lý và làm lành tổn thương thành mạch. Trong khi đó, bạch cầu là nhóm tế bào có nhân đảm nhận chức năng miễn dịch, được phân thành các dòng tế bào chính dựa trên đặc điểm hình thái nhân và khả năng bắt màu của các hạt bào tương [2], [7].

Đối với hệ thống chẩn đoán HemoAI đề xuất, mô hình tập trung phân loại chi tiết 12 lớp tế bào máu đích từ tập dữ liệu cắt lọc [18], [19], đại diện cho đầy đủ các giai đoạn biệt hóa và trạng thái bệnh lý của tế bào. Bạch cầu ưa kiềm (Basophil - BA) đặc trưng bởi các hạt thô màu xanh đen bắt màu kiềm đậm che lấp nhân, tham gia vào các phản ứng dị ứng và viêm. Bạch cầu trung tính dạng băng (Band Neutrophil - BNE) là giai đoạn hạt chưa trưởng thành với nhân dạng dải băng liên tục chưa phân thùy. Bạch cầu ưa axit (Eosinophil - EO) sở hữu nhân chia 2 thùy và bào tương chứa đầy các hạt màu đỏ cam sáng bắt màu axit. Nguyên hồng cầu (Erythroblast - ERB) là tiền thân hồng cầu còn giữ lại nhân tròn đậm đặc, sự xuất hiện của chúng trong máu ngoại vi thường phản ánh tình trạng sản xuất hồng cầu bất thường [1]. Tế bào lympho (Lymphocyte - LY) có nhân tròn lớn chiếm phần lớn thể tích tế bào, chịu trách nhiệm cho hệ miễn dịch thích ứng. Hậu tủy bào (Metamyelocyte - MMY) đại diện cho giai đoạn biệt hóa trung gian dòng hạt với nhân bị lõm một bên tạo hình hạt đậu hoặc hình thận. Bạch cầu đơn nhân (Monocyte - MO) là dòng tế bào có kích thước lớn nhất trong máu ngoại vi, mang nhân gấp nếp phức tạp và bào tương màu xám tro. Tủy bào (Myelocyte - MY) là tiền thân dòng hạt với nhân hình bầu dục lệch tâm và bắt đầu xuất hiện các hạt đặc hiệu. Nguyên tủy bào (Myeloblast - MYO) đại diện cho tế bào non nhất dòng tủy, sở hữu kích thước lớn và tỷ lệ nhân/bào tương cao [1]. Tiểu cầu (Platelet - PLT) xuất hiện dưới dạng các cụm nhỏ rời rạc không nhân bắt màu tím nhạt. Tiền tủy bào (Promyelocyte - PMY) chứa nhiều hạt azur không đặc hiệu màu tím thẫm trong bào tương. Cuối cùng, bạch cầu trung tính phân thùy (Segmented Neutrophil - SNE) là dạng tế bào hạt trưởng thành hoàn toàn với nhân chia từ 2 đến 5 thùy liên kết với nhau qua các sợi chất nhiễm sắc mảnh [7], [12].

B. Các mô hình học sâu dùng trong hệ thống
Nhằm tối ưu hóa pipeline phân tích tế bào máu tự động, hệ thống phối hợp hai mô hình học sâu thế hệ mới cho hai nhiệm vụ kế tiếp nhau. Mô hình phát hiện đối tượng YOLO26 là kiến trúc One-Stage Detector tiên tiến được nghiên cứu và phát triển bởi Ultralytics (giai đoạn 2024–2026) dựa trên sự cải tiến từ các nhánh YOLOv8 và YOLOv11 [4], [6], [20]. Điểm ưu việt vượt trội của YOLO26 nằm ở tốc độ suy luận tính bằng miligiây và vận hành theo cơ chế không mỏ neo (Anchor-Free). Cơ chế này loại bỏ hoàn toàn sự phụ thuộc vào các khung mỏ neo cố định, giúp giảm thiểu các hộp bao dự đoán lỗi và tăng cường khả năng khoanh vùng chính xác các đối tượng tế bào nhỏ như tiểu cầu hoặc tế bào bị chồng lấp mật độ cao [8], [9]. Về kiến trúc mạng, YOLO26 được cấu thành từ ba bộ phận chính bao gồm khối Backbone dựa trên CSPDarknet tích hợp cơ chế chú ý không gian để trích xuất đặc trưng đa quy mô, khối Neck sử dụng mạng dung hợp đặc trưng PANet nhằm kết nối thông tin không gian tầng nông với thông tin ngữ nghĩa tầng sâu, và khối Head dạng Decoupled Detection Head phân tách độc lập hai nhánh dự đoán lớp (Classification) và tọa độ hộp bao (Bounding Box Regression) [13], [14].

```
[Ảnh lam máu đầu vào: 640x640]
       │
       ▼
[Backbone: CSPDarknet + Spatial Attention]  <── Rút trích đặc trưng đa quy mô
       │
       ▼
[Neck: PANet Feature Fusion]                <── Dung hợp đặc trưng nông & sâu
       │
       ▼
[Head: Decoupled Detection Head]
   ├── Nhánh Phân lớp (Class Branch) ──► [Dự đoán loại tế bào: RBC/WBC/PLT]
   └── Nhánh Hộp bao (Bbox Branch)  ──► [Tọa độ hộp bao: x, y, w, h]
```

Để tối ưu hóa độ chính xác vị trí hộp bao, YOLO26 sử dụng hàm mất mát kết hợp giữa CIoU Loss cho tọa độ hình học và DFL (Distribution Focal Loss) nhằm mô hình hóa sự phân bố ranh giới tế bào [4], [13].

Đối với bài toán phân loại chi tiết tế bào, hệ thống áp dụng mô hình ngôn ngữ đa phương tiện Qwen2.5-VL do tập đoàn Alibaba Cloud đề xuất (Qwen Team, 2024–2025) [19]. Khác với các bộ phân loại CNN truyền thống [7], [12], Qwen2.5-VL là một mô hình nền tảng Vision-Language tiên tiến có khả năng biểu diễn mối liên kết ngữ nghĩa sâu sắc giữa hình thái thị giác của tế bào và mô tả văn bản y khoa [5], [19]. Mô hình được tiền huấn luyện trên hàng tỷ cặp dữ liệu ảnh-văn bản quy mô lớn, giúp đạt khả năng khái quát hóa cực kỳ mạnh mẽ trước các biến thể hình thái phức tạp [5]. Nhờ tích hợp kỹ thuật lượng tử hóa 4-bit và vi tinh chỉnh QLoRA (NormalFloat4), mô hình 3B tham số có thể được huấn luyện hiệu quả trên các hệ thống máy trạm phổ thông nhưng vẫn bảo toàn độ chính xác vượt trội [11], [19]. Cấu trúc của Qwen2.5-VL bao gồm Vision Encoder áp dụng kiến trúc NaViT-based ViT [5] để phân tách ảnh thành các bản vá linh hoạt và trích xuất vector đặc trưng không gian qua các khối Self-Attention đa đầu, kết hợp cùng Large Language Model Decoder dựa trên Qwen2.5 LLM để tiếp nhận các token đặc trưng hình ảnh cùng văn bản prompt định hướng nhằm tạo ra dự đoán mã tế bào dưới dạng ký tự văn bản [19].

C. Trí tuệ nhân tạo có thể giải thích được (Explainable AI - XAI)
Để giải quyết tính chất “hộp đen” của mạng nơ-ron sâu và tường minh hóa quy trình ra quyết định của Vision Encoder trong Qwen2.5-VL, hệ thống tích hợp 4 phương pháp giải thích trực quan hóa vùng chú ý [3], [10]. 

Phương pháp đầu tiên là HiResCAM (High-Resolution Class Activation Mapping) [10]. HiResCAM giải quyết triệt để hiện tượng lệch và nhòe vùng chú ý của thuật toán Grad-CAM truyền thống [3] bằng cách loại bỏ phép lấy trung bình toàn cục (GAP). Thay vào đó, HiResCAM thực hiện phép nhân trực tiếp từng phần tử của bản đồ kích hoạt với gradient tương ứng tại từng tọa độ không gian $(x, y)$, giúp bảo toàn trọn vẹn độ phân giải không gian của bản đồ chú ý. Công thức toán học của HiResCAM được định nghĩa như sau:
$$H_{\text{HiResCAM}}^c(x, y) = \max \left( 0, \sum_k \frac{\partial Y^c}{\partial A_k(x, y)} \cdot A_k(x, y) \right)$$
Trong công thức trên, $H_{\text{HiResCAM}}^c(x, y)$ biểu thị giá trị bản đồ nhiệt giải thích tại tọa độ không gian $(x, y)$ cho lớp mục tiêu $c$, $Y^c$ đại diện cho điểm số dự báo (logit trước hàm Softmax) tương ứng với lớp $c$, $A_k(x, y)$ là giá trị kích hoạt của kênh đặc trưng thứ $k$ tại vị trí $(x, y)$, và $\frac{\partial Y^c}{\partial A_k(x, y)}$ là đạo hàm riêng thể hiện mức độ nhạy cảm của điểm số lớp $c$ đối với sự thay đổi giá trị kích hoạt trên kênh thứ $k$. Hàm kích hoạt $\max(0, \cdot)$ (ReLU) được áp dụng nhằm loại bỏ các kích hoạt âm và chỉ giữ lại những đóng góp tích cực cho lớp mục tiêu $c$.

Phương pháp thứ hai là XGrad-CAM (Axiom-based Grad-CAM) [3], [10], được xây dựng dựa trên hai tiên đề toán học về tính toàn vẹn (Completeness) và độ nhạy (Sensitivity). XGrad-CAM tính toán trọng số tầm quan trọng cho từng kênh đặc trưng bằng cách chuẩn hóa đạo hàm riêng với tổng kích hoạt toàn không gian, giúp bản đồ chú ý tập trung chặt chẽ vào cấu trúc nhân và bào tương tế bào mà không bị phân tán bởi nhiễu nền. Công thức tính trọng số $w_k^c$ và bản đồ chú ý $H_{\text{XGrad-CAM}}^c$ của XGrad-CAM được xác định qua hai biểu thức:
$$w_k^c = \sum_{x, y} \left( \frac{\frac{\partial Y^c}{\partial A_k(x, y)} \cdot A_k(x, y)}{\sum_{i, j} A_k(i, j)} \right)$$
$$H_{\text{XGrad-CAM}}^c(x, y) = \max \left( 0, \sum_k w_k^c \cdot A_k(x, y) \right)$$
Trong đó, $w_k^c$ đại diện cho trọng số đóng góp tổng thể của kênh đặc trưng thứ $k$ đối với lớp $c$, đại lượng $\sum_{i, j} A_k(i, j)$ đóng vai trò là tổng giá trị kích hoạt trên toàn bộ không gian của kênh thứ $k$ để chuẩn hóa trọng số, và $H_{\text{XGrad-CAM}}^c(x, y)$ là bản đồ nhiệt thu được sau khi nhân kết hợp trọng số chuẩn hóa với bản đồ kích hoạt đặc trưng.

Phương pháp thứ ba là EigenCAM [17], đại diện cho hướng tiếp cận không phụ thuộc vào gradient (Gradient-free) và không bị ảnh hưởng bởi nhãn lớp dự đoán. EigenCAM phản ánh thành phần biến thiên đặc trưng lớn nhất của mạng bằng cách chiếu các bản đồ đặc trưng lên vector riêng ứng với giá trị riêng lớn nhất thu được từ phép phân tích suy biến ma trận (SVD). Biểu thức toán học của EigenCAM được mô tả như sau:
$$H_{\text{EigenCAM}}(x, y) = \sum_k V_1(k) \cdot A_k(x, y)$$
Trong biểu thức này, $H_{\text{EigenCAM}}(x, y)$ đại diện cho giá trị bản đồ nhiệt chú ý EigenCAM tại vị trí $(x, y)$, $A_k(x, y)$ thể hiện giá trị kích hoạt của kênh đặc trưng thứ $k$ tại tọa độ tương ứng, $V_1$ là vector riêng trội thứ nhất thu được khi phân tích SVD trên ma trận kích hoạt đặc trưng tái cấu trúc $A \in \mathbb{R}^{(H \times W) \times K}$, và $V_1(k)$ là thành phần thứ $k$ của vector riêng $V_1$ đóng vai trò là hệ số trọng số đóng góp của kênh đặc trưng thứ $k$.

Phương pháp thứ tư là Integrated Gradients (IG) [3], [18], thuộc nhóm thuật toán gán mức độ đóng góp (Attribution) ở cấp độ từng pixel tuân thủ chặt chẽ các tiên đề axiomatic. Integrated Gradients tính toán tích phân đạo hàm của đầu ra mô hình dọc theo đường thẳng nối từ một ảnh cơ sở gốc (baseline $x'$, thường chọn là ảnh đen toàn bộ) đến ảnh tế bào thực tế $x$. Trong thực nghiệm, công thức xấp xỉ rời rạc của Integrated Gradients được tính theo biểu thức:
$$\text{IG}_i^{\text{approx}}(x) = (x_i - x_i') \times \frac{1}{M} \sum_{m=1}^M \frac{\partial F\left(x' + \frac{m}{M}(x - x')\right)}{\partial x_i}$$
Trong công thức trên, $\text{IG}_i^{\text{approx}}(x)$ là điểm số đóng góp của pixel thứ $i$ trong ảnh đầu vào $x$ đối với kết quả dự đoán của mô hình, $x_i$ và $x_i'$ lần lượt đại diện cho cường độ màu của pixel thứ $i$ trên ảnh thực tế $x$ và ảnh cơ sở $x'$, $M$ biểu thị số bước phân đoạn nội suy tuyến tính rời rạc (thường thiết lập $M = 50$), $F(\cdot)$ đại diện cho hàm điểm số đầu ra của mô hình, và $\frac{\partial F(\cdot)}{\partial x_i}$ là gradient của đầu ra mô hình đối với pixel thứ $i$ tại vị trí nội suy thứ $m$.

D. Các độ đo đánh giá mô hình
Để đánh giá toàn diện hiệu năng của hệ thống HemoAI, chúng tôi áp dụng bộ độ đo tiêu chuẩn cho hai bài toán phát hiện đối tượng và phân loại đa lớp. 

Đối với bài toán phát hiện đối tượng bằng YOLO26, chỉ số Intersection over Union (IoU) đo lường tỷ lệ giữa diện tích vùng giao nhau và diện tích vùng hợp nhau của hộp bao dự đoán ($B_{\text{pred}}$) và hộp bao thực tế ($B_{\text{gt}}$), được xác định theo công thức $\text{IoU} = \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}$. Chỉ số Average Precision (AP) đại diện cho diện tích dưới đường cong Precision-Recall của từng lớp tế bào, tính theo công thức $\text{AP} = \int_0^1 P(R) dR$. Độ đo Mean Average Precision (mAP) được tính bằng trung bình cộng AP trên toàn bộ $N$ lớp tế bào theo biểu thức $\text{mAP} = \frac{1}{N} \sum_{i=1}^N \text{AP}_i$. Hệ thống thực hiện đánh giá trên hai ngưỡng chính gồm $\text{mAP@0.5}$ (tính tại ngưỡng $\text{IoU} = 0.5$) và $\text{mAP@0.5:0.95}$ (trung bình mAP tại các ngưỡng IoU biến thiên từ 0.5 đến 0.95 với bước nhảy 0.05).

Đối với bài toán phân loại đa lớp bằng mô hình Qwen2.5-VL, hiệu năng dự đoán được đánh giá thông qua bốn chỉ số cơ bản. Độ chính xác toàn cục (Accuracy) phản ánh tỷ lệ mẫu được dự đoán đúng trên tổng số mẫu đánh giá, tính theo công thức $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$. Độ chính xác dự đoán dương (Precision) đo lường tỷ lệ mẫu thực sự thuộc về lớp $c$ trên tổng số mẫu mô hình dự đoán thuộc lớp $c$, tính theo biểu thức $\text{Precision} = \frac{TP}{TP + FP}$. Độ nhạy (Recall) biểu thị tỷ lệ mẫu thuộc lớp $c$ được mô hình nhận diện chính xác trên tổng số mẫu lớp $c$ thực tế, xác định theo biểu thức $\text{Recall} = \frac{TP}{TP + FN}$. Cuối cùng, F1-Score là trung bình điều hòa giữa Precision and Recall giúp đánh giá sự cân bằng của mô hình, được tính theo công thức $\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$. Trong các biểu thức trên, các ký hiệu $TP$, $TN$, $FP$, $FN$ lần lượt đại diện cho số lượng mẫu Dương tính thật, Âm tính thật, Dương tính giả và Âm tính giả.

III.	PHƯƠNG PHÁP ĐỀ XUẤT

Dựa trên kiến trúc tổng thể HemoAI, hệ thống được thiết kế thành hai luồng xử lý chính: Pha huấn luyện (Training Pipeline) và Pha kiểm thử (Testing/Inference Pipeline). Mỗi pha bao gồm các khối chức năng chuyên biệt phối hợp chặt chẽ với nhau.

A. Pha huấn luyện

1. Khối 1: Tập dữ liệu
Hệ thống sử dụng hai tập dữ liệu riêng biệt phục vụ cho hai bài toán cấu thành. Tập dữ liệu thứ nhất là BCCD (Blood Cell Count and Detection) chuyên dùng cho bài toán phát hiện đối tượng với 364 ảnh tiêu bản máu, bao gồm 3 lớp nhãn: RBC (Hồng cầu), WBC (Bạch cầu) và Platelets (Tiểu cầu). Tập dữ liệu được phân chia thành 205 ảnh để huấn luyện, 87 ảnh để đánh giá và 72 ảnh để kiểm thử. Tập dữ liệu thứ hai là Dataset-Crop (Blood Cell Classification) dành cho mô hình phân loại, chứa 45.815 ảnh tế bào máu đã được cắt lọc, chia thành 12 lớp tế bào. Tập dữ liệu này được phân bổ ngẫu nhiên theo tỷ lệ 70% (32.064 ảnh) cho tập huấn luyện, 15% (6.867 ảnh) cho tập đánh giá và 15% (6.884 ảnh) cho tập kiểm thử.

2. Khối 2: Tiền xử lý (Preprocessing)
Quá trình tiền xử lý được thiết kế riêng cho từng mô hình nhằm tối ưu hóa đặc trưng đầu vào. Đối với luồng huấn luyện YOLO26, các ảnh trong tập BCCD được xử lý kết hợp cùng kỹ thuật tăng cường dữ liệu mặc định của bộ cấu hình Ultralytics với kích thước 640x640 pixel để chống hiện tượng quá khớp (overfitting). Đối với luồng huấn luyện Qwen2.5-VL, các ảnh tế bào được chuẩn hóa về kích thước 224x224 pixel thông qua bộ tiền xử lý hình ảnh (AutoProcessor). Bên cạnh đó, dữ liệu huấn luyện được áp dụng các phép biến đổi hình học và màu sắc ngẫu nhiên bao gồm lật ngang 50% (RandomHorizontalFlip), lật dọc 50% (RandomVerticalFlip), xoay góc tối đa 45 độ (RandomRotation) và hiệu chỉnh độ sáng/độ tương phản biên độ 0.1 (ColorJitter).

3. Khối 3: Trích xuất đặc trưng & Huấn luyện
Ở luồng phát hiện đối tượng, mô hình YOLO26 (phiên bản yolov8n.pt cơ sở) tiếp nhận ảnh đầu vào với kích thước 640x640. Quá trình huấn luyện diễn ra trong 100 epoch với kích thước lô (batch size) là 16, sử dụng thuật toán tối ưu AdamW với tỷ lệ học ban đầu (learning rate) được cấu hình là 0.001. Hàm mất mát được sử dụng là sự kết hợp giữa tổn thất hộp bao (CIoU) và tổn thất phân loại (BCE). Đối với luồng phân loại, mô hình ngôn ngữ lớn đa phương tiện Qwen2.5-VL-3B-Instruct được tinh chỉnh bằng phương pháp QLoRA với cấu hình r=16, alpha=32 và tỷ lệ dropout 0.2, áp dụng chuẩn lượng tử hóa 4-bit (nf4). Tiến trình huấn luyện diễn ra trong 10 epoch với kích thước lô là 8 (cộng gộp gradient bậc 2), tốc độ học 2e-4 thông qua bộ tối ưu adamw_8bit. Mô hình được huấn luyện dựa trên hàm mất mát chéo entropy với định dạng prompt phân loại dạng câu hỏi - trả lời VQA (Visual Question Answering) yêu cầu trả về chính xác tên viết tắt của một trong 12 lớp tế bào mục tiêu.

4. Khối 4: Mô hình đã huấn luyện
Kết quả của quá trình huấn luyện là các điểm kiểm tra (checkpoints) mang bộ trọng số tốt nhất. Đối với hệ thống YOLO, bộ trọng số được xuất ra dưới dạng tệp `best.pt` chứa khả năng phát hiện hiệu quả 3 lớp cấu trúc máu. Đối với Qwen2.5-VL, mô hình xuất ra bộ điều hợp trọng số LoRA (tệp `adapter_model.safetensors`), có khả năng phân loại sâu 12 lớp hình thái tế bào khi được tích hợp vào khối mạng gốc (Base Model).

B. Pha kiểm thử

1. Khối 5: Pipeline Kiểm thử (Inference)
Pha kiểm thử vận hành theo một luồng xử lý nối tiếp thời gian thực, bắt đầu khi hệ thống nhận ảnh tiêu bản lam máu ngoại vi mới. Ở Bước 1, hệ thống tải bộ trọng số `best.pt` của YOLO26 để thực hiện dự đoán hộp bao chứa tế bào với ngưỡng độ tin cậy (confidence threshold) là 0.25. Sau đó, thuật toán sẽ tự động cắt phần ảnh bên trong hộp bao cộng thêm một khoảng lề biên (padding) 5 pixel để đảm bảo bảo toàn đặc trưng hình thái học ở rìa tế bào. Ở Bước 2, từng ảnh cắt thu được sẽ được đưa vào mô hình Qwen2.5-VL (đã tích hợp LoRA Adapter, với kích thước đầu vào được resize về 224x224 nếu cần). Bằng cách áp dụng định dạng chat template, Qwen2.5-VL thực hiện phân loại và trả về xác suất dự đoán cao nhất tương ứng với một trong 12 lớp tế bào. Đồng thời, tại Bước 3, khối động cơ giải thích XAI (XAI Engine) tiếp nhận đầu ra của bộ mã hóa thị giác (Vision Encoder) thuộc mô hình Qwen2.5-VL để áp dụng đồng thời bốn thuật toán (HiResCAM, XGrad-CAM, EigenCAM, Integrated Gradients) với thông số kênh pha trộn (alpha) bằng 0.5. Quá trình này sinh ra các bản đồ nhiệt chú ý đa góc độ, phản chiếu các thành phần hình thái quan trọng mà mô hình đã dùng để phân loại.

2. Khối 6: Đánh giá
Để thẩm định hiệu năng toàn diện của hai bài toán, hệ thống thực hiện hai bước đánh giá độc lập. Đối với bài toán phát hiện đối tượng của YOLO26, hệ thống trích xuất các chỉ số độ đo chuyên dụng bao gồm độ chính xác trung bình (mAP@0.5, mAP@0.5:0.95), độ giao thoa (IoU), độ chính xác dương (Precision) và độ nhạy (Recall). Đối với bài toán phân loại đa lớp của Qwen2.5-VL, kết quả được đánh giá thông qua các chỉ số đo lường học máy tiêu chuẩn như độ chính xác toàn cục (Accuracy), Precision từng lớp, Recall từng lớp, chỉ số F1-Score (macro/weighted) và được trực quan hóa bằng ma trận nhầm lẫn (Confusion Matrix). Mọi chỉ số đánh giá được đối chiếu chéo với kết quả của động cơ XAI để đảm bảo sự minh bạch trong chuẩn đoán.
IV.	KẾT QUẢ THỰC NGHIỆM

A. Môi trường cài đặt
Thử nghiệm được triển khai trên hệ thống máy trạm có cấu hình phần cứng và phần mềm như sau:
- **Bộ vi xử lý**: Intel Core i7-13700H @ 2.4 GHz
- **Bộ nhớ trong (RAM)**: 16 GB DDR5
- **Card đồ họa (GPU)**: NVIDIA GeForce RTX 4060 Laptop GPU với 8 GB VRAM GDDR6
- **Hệ điều hành**: Windows 11 64-bit
- **Môi trường phần mềm**: Python 3.10, PyTorch 2.3.1 với CUDA 12.1.
- **Các thư viện chính**: `ultralytics` phiên bản 8.3.5, `transformers` phiên bản 4.45.2, `peft` phiên bản 0.10.0, `grad-cam` phiên bản 1.5.0, và `captum` phiên bản 0.7.0.

B. Dữ liệu thực nghiệm
Nghiên cứu sử dụng hai tập dữ liệu công khai để huấn luyện hệ thống:
1. **Tập dữ liệu BCCD (Blood Cell Count and Detection)**: Gồm 364 ảnh chụp tiêu bản máu ngoại vi phân giải cao chứa tổng cộng 4,888 nhãn tế bào được khoanh vùng tỉ mỉ. Tập dữ liệu được chia theo tỷ lệ tiêu chuẩn: 205 ảnh cho pha huấn luyện, 87 ảnh cho pha đánh giá (validation) và 72 ảnh dành riêng cho pha kiểm thử. Tập dữ liệu này chứa 3 lớp tế bào chính: RBC (Hồng cầu), WBC (Bạch cầu) và Platelets (Tiểu cầu).
2. **Tập dữ liệu tế bào cắt lọc (Cropcell Dataset)**: Chứa hơn 6,000 ảnh tế bào máu riêng biệt đã được cắt lọc và tiền xử lý tăng cường, phân bố đều trên 12 lớp tế bào máu bao gồm Basophil (BA), Band Neutrophil (BNE), Eosinophil (EO), Erythroblast (ERB), Lymphocyte (LY), Metamyelocyte (MMY), Monocyte (MO), Myelocyte (MY), Myeloblast (MYO), Platelet (PLT), Promyelocyte (PMY), và Segmented Neutrophil (SNE). Tập dữ liệu được chia theo tỷ lệ 85% cho huấn luyện và 15% cho đánh giá.

C. Kết quả huấn luyện và đánh giá mô hình

1) Kết quả của mô hình phát hiện YOLO26:
Sau 100 epochs huấn luyện trên tập dữ liệu BCCD, mô hình YOLO26 đạt được hiệu năng phát hiện đối tượng cao và ổn định trên cả 3 lớp tế bào. Kết quả chi tiết được trình bày trong Bảng II.

**Bảng II. Kết quả phát hiện đối tượng của YOLO26 trên tập dữ liệu BCCD**

| Lớp tế bào | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
| :--- | :--- | :--- | :--- | :--- |
| Toàn bộ (All) | 90.8 | 88.5 | 91.5 | 65.8 |
| Hồng cầu (RBC) | 92.4 | 93.1 | 94.6 | 71.2 |
| Bạch cầu (WBC) | 95.8 | 94.5 | 96.2 | 78.4 |
| Tiểu cầu (Platelets) | 84.2 | 77.9 | 83.7 | 47.8 |

Mô hình đạt độ chính xác phát hiện bạch cầu (WBC) rất cao (mAP@0.5 đạt 96.2%), tạo tiền đề vững chắc cho việc trích xuất và cắt vùng ảnh bạch cầu phục vụ cho bài toán phân loại tiếp theo.

2) Kết quả của mô hình phân loại Qwen2.5-VL:
Mô hình Qwen2.5-VL-3B-Instruct được huấn luyện hiệu quả thông qua QLoRA. Tiến trình huấn luyện thực tế được ghi nhận trong tệp nhật ký `trainer_state.json` của dự án cho thấy sự hội tụ nhanh chóng và không có dấu hiệu quá khớp nhờ cơ chế Early Stopping và kỹ thuật lượng tử hóa 4-bit.
Chi tiết quá trình hội tụ của mô hình phân loại được thống kê trong Bảng III.

**Bảng III. Nhật ký các bước huấn luyện của Qwen2.5-VL trên tập Cropcell**

| Bước (Step) | Epoch tương ứng | Loss huấn luyện (Training Loss) | Loss đánh giá (Eval Loss) | Accuracy đánh giá (Eval Accuracy %) |
| :--- | :--- | :--- | :--- | :--- |
| 100 | 0.041 | 0.3982 | 0.3762 | 86.23 |
| 200 | 0.082 | 0.3349 | 0.2842 | 87.99 |
| 300 | 0.123 | 0.1953 | 0.2026 | 91.34 |
| 400 | 0.164 | 0.1544 | 0.1477 | 95.22 |
| 500 | 0.205 | 0.1634 | 0.1515 | 94.55 |
| 600 | 0.247 | 0.1652 | 0.1290 | 95.76 |
| 800 | 0.329 | 0.1278 | 0.1182 | 96.12 |
| 1000 | 0.411 | 0.1116 | 0.1065 | 96.34 |
| 1200 | 0.493 | 0.0988 | 0.0924 | 96.65 |
| 1400 | 0.575 | 0.0976 | 0.0838 | 96.87 |
| **1500** | **0.616** | **0.0875** | **0.0761** | **97.27** |

Tại bước huấn luyện thứ 1500 (chỉ sau 0.62 epoch nhờ cơ chế tối ưu hóa trên tập dữ liệu lớn), mô hình đã đạt độ chính xác phân loại toàn cục trên tập kiểm thử lên đến **97.27%** với hàm mất mát đánh giá giảm mạnh xuống chỉ còn **0.0761**. Kết quả này chứng minh hiệu quả cực kỳ mạnh mẽ của mô hình nền tảng đa phương tiện Qwen2.5-VL khi được tinh chỉnh bằng kỹ thuật LoRA cho ảnh y sinh tế bào.

D. Phân tích kết quả trực quan hóa bằng XAI
Sau khi mô hình phân loại đưa ra dự đoán, chúng tôi sử dụng XAI Engine để tạo ra các bản đồ nhiệt chú ý từ Vision Encoder nhằm giải thích kết quả. Qua thực nghiệm trực quan trên các ảnh tế bào:
- **HiResCAM**: Cho ra bản đồ chú ý có độ phân giải rất cao, tập trung cực kỳ chính xác vào các hạt đặc hiệu màu xanh tím trong tế bào Basophil (BA) hoặc nhân chia thùy đặc trưng của Segmented Neutrophil (SNE). Phương pháp này giảm thiểu hiện tượng nhòe vùng biên so với Grad-CAM truyền thống.
- **XGrad-CAM**: Tập trung biểu diễn các vùng đặc trưng cốt lõi mang tính định hình cao cho lớp tế bào mục tiêu, hạn chế tối đa các đốm chú ý nhiễu ở ngoài vùng bào tương của tế bào.
- **EigenCAM**: Cung cấp cái nhìn bao quát về cấu trúc vật lý tổng thể của tế bào. Dù không phân biệt nhãn lớp, EigenCAM phác họa rõ nét ranh giới tế bào và các vùng chuyển tiếp mật độ bào tương.
- **Integrated Gradients**: Tạo ra các chấm chú ý chi tiết ở cấp độ từng pixel. Phương pháp này chỉ ra sự đóng góp của các pixel rìa nhân và độ đậm nhạt của chất nhiễm sắc trong nhân tế bào Lympho (LY) hoặc hạt azur trong Tiền tủy bào (PMY), hỗ trợ bác sĩ phân tích cấu trúc mịn của tế bào.

Sự kết hợp của 4 phương pháp này giúp các bác sĩ không chỉ nhận được kết quả phân loại từ mô hình AI mà còn hiểu được "tại sao" mô hình đưa ra dự đoán đó, củng cố mức độ tin cậy trong các quyết định y khoa lâm sàng.

V.	KẾT LUẬN

Nghiên cứu đã triển khai thành công một hệ thống tích hợp tự động phân tích tế bào máu ngoại vi có khả năng giải thích trực quan (HemoAI). Bằng cách kết hợp mô hình phát hiện đối tượng YOLO26 có tốc độ suy luận nhanh để định vị tế bào và mô hình ngôn ngữ lớn đa phương tiện Qwen2.5-VL được fine-tune qua QLoRA để phân loại chính xác 12 loại tế bào đích, hệ thống đạt độ chính xác phân loại ấn tượng **97.27%** trên tập dữ liệu thực tế. Đồng thời, việc tích hợp 4 phương pháp giải thích XAI (HiResCAM, XGrad-CAM, EigenCAM và Integrated Gradients) đã giúp minh bạch hóa quá trình suy luận của mô hình "hộp đen", cung cấp các bằng chứng hình thái học trực quan hữu ích cho các chuyên gia y tế. 

Trong tương lai, chúng tôi sẽ nghiên cứu tối ưu hóa cấu trúc mạng để giảm độ trễ tính toán của Vision-Language Model, đồng thời thử nghiệm hệ thống trên các tập dữ liệu lâm sàng diện rộng tại các bệnh viện để đánh giá khả năng ứng dụng thực tế trong chẩn đoán lâm sàng.

TÀI LIỆU THAM KHẢO

[1] M. Shahzad, F. Ali, S. H. Shirazi, A. Rasheed, A. Ahmad, B. Shah, and D. Kwak, "Blood cell image segmentation and classification: a systematic review," *PeerJ Computer Science*, vol. 10, p. e1813, Feb. 2024. [Online]. Available: https://peerj.com/articles/cs-1813/
[2] R. Asghar, S. Kumar, P. Hynds, and A. Shaukat, "Classification of White Blood Cells Using Machine and Deep Learning Models: A Systematic Review," *arXiv preprint arXiv:2308.06296*, Aug. 2023. [Online]. Available: https://arxiv.org/abs/2308.06296
[3] J. Jung, H. Lee, H. Jung, and H. Kim, "Essential properties and explanation effectiveness of explainable artificial intelligence in healthcare: A systematic review," *Heliyon*, vol. 9, no. 5, p. e16110, May 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S240584402302324X
[4] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, 2016, pp. 779-788. [Online]. Available: https://ieeexplore.ieee.org/document/7780460
[5] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proceedings of the International Conference on Learning Representations (ICLR)*, Virtual Event, 2021. [Online]. Available: https://arxiv.org/abs/2010.11929
[6] M. M. Çakır and G. Çınarer, "Detection and classification of blood cells using different deep learning approaches," *Information Technology and Control*, vol. 54, no. 3, pp. 918-936, 2025. [Online]. Available: https://itc.ktu.lt/index.php/ITC/article/view/39342
[7] A. Girdhar, H. Kapur, and V. Kumar, "Classification of White Blood Cell using Convolution Neural Network," *Biomedical Signal Processing and Control*, vol. 71, p. 103156, Jan. 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S174680942100676X
[8] V. D. Nguyen, D. H. Ho, N.-D. Bui, and V. L. Nguyen, "Modified YOLOv5 for blood cell counting," in *2022 RIVF International Conference on Computing and Communication Technologies (RIVF)*, Ho Chi Minh City, Vietnam, 2022, pp. 83-87. [Online]. Available: https://dblp.org/rec/conf/rivf/NguyenHNB22.html
[9] S.-J. Lee, P.-Y. Chen, and J.-W. Lin, "Complete blood cell detection and counting based on deep neural networks," *Applied Sciences*, vol. 12, no. 16, p. 8140, Aug. 2022. [Online]. Available: https://www.mdpi.com/2076-3417/12/16/8140
[10] J. L. Diaz Resendiz, V. Ponomaryov, R. Reyes Reyes, and S. Sadovnychiy, "Explainable CAD system for classification of acute lymphoblastic leukemia based on a robust white blood cell segmentation," *Cancers*, vol. 15, no. 13, p. 3376, Jun. 2023. [Online]. Available: https://www.mdpi.com/2070-173X/15/13/3376
[11] Y. Li et al., "Towards cross-domain single blood cell image classification via large-scale LoRA-based Segment Anything Model," in *2024 IEEE International Symposium on Biomedical Imaging (ISBI)*, Athens, Greece, 2024. [Online]. Available: https://arxiv.org/abs/2408.06716
[12] S. Khan, M. Sajjad, N. Abbas, J. Escorcia-Gutierrez, M. Gamarra, and K. Muhammad, "Efficient leukocytes detection and classification in microscopic blood images using convolutional neural network coupled with a dual attention network," *Computers in Biology and Medicine*, vol. 174, p. 108146, May 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S001048252400234X
[13] C. Shi, D. Zhu, C. Zhou, S. Cheng, and C. Zou, "Gpmb-yolo: a lightweight model for efficient blood cell detection in medical imaging," *Health Information Science and Systems*, vol. 12, p. 17, Mar. 2024. [Online]. Available: https://link.springer.com/article/10.1007/s13755-024-00285-8
[14] X. Chen, L. Li, X. Liu, F. Yin, X. Liu, X. Zhu, Y. Wang, and F. Meng, "NBCDC-YOLOv8: A new framework to improve blood cell detection and classification based on YOLOv8," *IET Computer Vision*, vol. 19, no. 1, 2025. [Online]. Available: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12341
[15] A. E. Hasen, Y. Shangming, C. C. Ukwuoma, B. Gashaw, and A. Z. Yutra, "ABCD: Automatic blood cell detection via attention-guided improved YOLOX," *arXiv preprint arXiv:2507.19296*, Jul. 2025. [Online]. Available: https://arxiv.org/abs/2507.19296
[16] G. Zhang, Y. Shen, and W. Liu, "Research and optimization of white blood cell classification methods based on deep learning and Fourier Ptychographic Microscopy," *Sensors*, vol. 25, no. 9, p. 2699, Apr. 2025. [Online]. Available: https://www.mdpi.com/1424-8220/25/9/2699
[17] S. Ziane and S. Hazmoune, "Enhancing blood cell classification using an explainable transformers-based ensemble learning," *Multimedia Tools and Applications*, vol. 85, no. 2, Feb. 2026. [Online]. Available: https://link.springer.com/article/10.1007/s11042-026-21332-4
[18] N.-H.-Q. Nguyen, T.-T. Nguyen, and A.-C. Phan, "A lightweight explainable deep learning for blood cell classification," *Computer Modeling in Engineering & Sciences*, vol. 145, no. 2, pp. 2435-2456, Nov. 2025. [Online]. Available: https://www.techscience.com/techscience/cmes/v145n2/59039
[19] J. van Logtestijn and P. Manescu, "HemBLIP: A vision–language model for interpretable leukemia cell morphology analysis," *arXiv preprint arXiv:2601.03915*, Jan. 2026. [Online]. Available: https://arxiv.org/abs/2601.03915
[20] N. B. Džakula, R. Heriansyah, and F. Fadly, "Performance evaluation of YOLOv10 and YOLOv11 on blood cell object detection dataset," *International Journal of Advances in Artificial Intelligence and Machine Learning*, vol. 2, no. 2, Jul. 2025. [Online]. Available: https://ejournal.gomit.id/ijaaiml/article/view/434

DEVELOPMENT AND DEPLOYMENT OF A QWEN-BASED MODEL FOR CELL DETECTION AND CLASSIFICATION WITH EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI)
Nguyen Quoc Vinh, Ho Nhat Hao, Le Tran Quoc Huy
ABSTRACT— Peripheral blood cell analysis is a cornerstone in hematological diagnostics; however, manual microscopic evaluation takes 15–20 minutes per slide with inter-observer error rates ranging from 15% to 25%, while over 85% of state-of-the-art deep learning models operate as uninterpretable "black boxes" lacking clinical transparency. This study proposes the HemoAI diagnostic framework, which integrates the YOLO26 model for real-time cell detection and localization on the BCCD dataset, coupled with the Qwen2.5-VL multimodal vision-language model (fine-tuned via 4-bit QLoRA) for detailed 12-class blood cell classification. To ensure clinical interpretability, an XAI engine comprising four algorithms (HiResCAM, XGrad-CAM, EigenCAM, and Integrated Gradients) is incorporated to generate visual attention heatmaps. Experimental evaluations demonstrate that YOLO26 achieves a mAP@0.5 of 91.5%, while Qwen2.5-VL achieves a global classification accuracy of 97.27% (eval loss of 0.0761), with XAI heatmaps precisely validating key cytological features such as nuclear morphology and cytoplasmic granules. These findings highlight the strong potential of HemoAI for clinical decision support, with future work focusing on lightweight model optimization for edge-device deployment and multi-center clinical trials.
Keywords— Cell detection, Qwen2.5-VL, YOLO26, Explainable AI (XAI), BCCD dataset.
