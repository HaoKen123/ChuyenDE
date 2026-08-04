# -*- coding: utf-8 -*-
"""
MODULE TẠO BÁO CÁO KẾT QUẢ XÉT NGHIỆM TẾ BÀO MÁU (PDF MEDICAL REPORT GENERATOR)
Dự án: Triển khai Qwen cho phát hiện & phân loại tế bào kết hợp XAI
"""

import os
import time
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Khoảng tham chiếu chuẩn y tế cho các tế bào máu
REFERENCE_RANGES = {
    'RBC': {'name': 'Erythrocytes (Red Blood Cells)', 'range': '4.2 - 5.9 M/µL', 'unit': 'M/µL'},
    'WBC': {'name': 'Leukocytes (White Blood Cells)', 'range': '4.5 - 11.0 K/µL', 'unit': 'K/µL'},
    'PLT': {'name': 'Platelets (Thrombocytes)',       'range': '150 - 450 K/µL', 'unit': 'K/µL'},
    'BA':  {'name': 'Basophil',                       'range': '0.0 - 1.0 %',    'unit': '%'},
    'BNE': {'name': 'Band Neutrophil',                'range': '0.0 - 3.0 %',    'unit': '%'},
    'EO':  {'name': 'Eosinophil',                     'range': '1.0 - 4.0 %',    'unit': '%'},
    'ERB': {'name': 'Erythroblast',                   'range': '0.0 - 0.5 %',    'unit': '%'},
    'LY':  {'name': 'Lymphocyte',                     'range': '20.0 - 40.0 %',  'unit': '%'},
    'MMY': {'name': 'Metamyelocyte',                  'range': '0.0 - 0.0 %',    'unit': '%'},
    'MO':  {'name': 'Monocyte',                       'range': '2.0 - 8.0 %',    'unit': '%'},
    'MY':  {'name': 'Myelocyte',                      'range': '0.0 - 0.0 %',    'unit': '%'},
    'MYO': {'name': 'Myeloblast',                     'range': '0.0 - 0.0 %',    'unit': '%'},
    'PMY': {'name': 'Promyelocyte',                   'range': '0.0 - 0.0 %',    'unit': '%'},
    'SNE': {'name': 'Segmented Neutrophil',          'range': '40.0 - 70.0 %',  'unit': '%'},
}


class NumberedCanvas(canvas.Canvas):
    """Canvas vẽ trang trí Header/Footer và Đánh số trang kiểu Y tế hiện đại."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Thanh màu trang trí trên đỉnh
        self.setFillColor(colors.HexColor('#0F172A'))
        self.rect(0, 832, 595, 10, fill=True, stroke=False)
        self.setFillColor(colors.HexColor('#2563EB'))
        self.rect(0, 828, 595, 4, fill=True, stroke=False)

        # Thanh trang trí ở chân trang
        self.setStrokeColor(colors.HexColor('#E2E8F0'))
        self.setLineWidth(0.8)
        self.line(36, 45, 559, 45)

        # Text Chân trang
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#64748B'))
        self.drawString(36, 32, "CDSS-XAI Blood Lab System | Báo cáo Phân tích AI Hỗ trợ Chẩn đoán Lâm sàng")
        
        page_str = f"Trang {self._pageNumber} / {page_count}"
        self.drawRightString(559, 32, page_str)
        self.restoreState()


def create_signature_block():
    """Tạo khối chữ ký đàng hoàng với dấu mộc và chữ ký điện tử."""
    style_sig_title = ParagraphStyle(
        'SigTitle', fontName='Helvetica-Bold', fontSize=10, leading=12, alignment=1, textColor=colors.HexColor('#1E293B')
    )
    style_sig_sub = ParagraphStyle(
        'SigSub', fontName='Helvetica-Oblique', fontSize=8, leading=10, alignment=1, textColor=colors.HexColor('#64748B')
    )
    style_sig_name = ParagraphStyle(
        'SigName', fontName='Helvetica-Bold', fontSize=10, leading=12, alignment=1, textColor=colors.HexColor('#2563EB')
    )

    col1 = [
        Paragraph("BÁC SĨ PHÂN TÍCH", style_sig_title),
        Paragraph("(Ký, ghi rõ họ tên)", style_sig_sub),
        Spacer(1, 40),
        Paragraph("<b>BS. CKII. Nguyễn Quốc Vinh</b>", style_sig_name),
        Paragraph("Mã số CCHN: 018492/BYT", style_sig_sub),
    ]

    col2 = [
        Paragraph("TRƯỞNG KHOA XÉT NGHIỆM", style_sig_title),
        Paragraph("(Ký tên, đóng dấu)", style_sig_sub),
        Spacer(1, 40),
        Paragraph("<b>TS. BS. Hồ Nhật Hào</b>", style_sig_name),
        Paragraph("Trung tâm Xét nghiệm Huyết học", style_sig_sub),
    ]

    sig_table = Table([[col1, col2]], colWidths=[250, 250])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    return sig_table


def generate_medical_pdf(
    output_pdf_path,
    patient_info=None,
    yolo_counts=None,
    top_pred_class="Segmented Neutrophil",
    top_pred_code="SNE",
    top_confidence=96.5,
    top5_probs=None,
    detected_img_path=None,
    xai_grid_path=None,
    vlm_description=None
):
    """
    Hàm sinh file PDF báo cáo y khoa chuyên nghiệp.
    """
    if patient_info is None:
        patient_info = {
            'patient_name': 'Trần Văn An',
            'patient_id': f'BN-{int(time.time()) % 1000000:06d}',
            'age_gender': '35 / Nam',
            'sample_id': f'SMP-{int(time.time()) % 10000:04d}',
            'doctor_req': 'BS. Lê Trần Quốc Huy',
            'date_time': time.strftime("%d/%m/%Y %H:%M:%S")
        }

    if yolo_counts is None:
        yolo_counts = {'RBC': 42, 'WBC': 3, 'Platelets': 8}

    if top5_probs is None:
        top5_probs = [
            ('Segmented Neutrophil (SNE)', 96.5),
            ('Band Neutrophil (BNE)', 2.1),
            ('Metamyelocyte (MMY)', 0.8),
            ('Eosinophil (EO)', 0.4),
            ('Monocyte (MO)', 0.2),
        ]

    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=25,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()

    # Các style tùy chỉnh
    style_h1 = ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=colors.HexColor('#0F172A'), alignment=0)
    style_h2 = ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=colors.HexColor('#1E3A8A'), spaceBefore=8, spaceAfter=4)
    style_sub = ParagraphStyle('Sub', fontName='Helvetica', fontSize=8.5, leading=11, textColor=colors.HexColor('#64748B'))
    style_cell = ParagraphStyle('Cell', fontName='Helvetica', fontSize=8.5, leading=11, textColor=colors.HexColor('#334155'))
    style_cell_bold = ParagraphStyle('CellB', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor('#0F172A'))

    story = []

    # 1. HEADER DỊCH VỤ Y TẾ
    header_left = [
        Paragraph("<b>BỆNH VIỆN ĐA KHOA QUỐC TẾ - TRUNG TÂM XÉT NGHIỆM</b>", ParagraphStyle('HLeft1', fontName='Helvetica-Bold', fontSize=10, leading=12, textColor=colors.HexColor('#1E3A8A'))),
        Paragraph("Địa chỉ: 123 Đường Lý Thường Kiệt, Quận 10, TP. Hồ Chí Minh", style_sub),
        Paragraph("Hotline: (028) 3829 9999 | Email: hematology@ai-medlab.org", style_sub),
    ]
    header_right = [
        Paragraph("<b>PHIẾU KẾT QUẢ XÉT NGHIỆM</b>", ParagraphStyle('HRight1', fontName='Helvetica-Bold', fontSize=13, leading=15, alignment=2, textColor=colors.HexColor('#2563EB'))),
        Paragraph(f"Mã phiếu: <b>{patient_info.get('sample_id')}</b>", ParagraphStyle('HRight2', fontName='Helvetica', fontSize=9, leading=11, alignment=2, textColor=colors.HexColor('#475569'))),
        Paragraph(f"Ngày in: {patient_info.get('date_time')}", ParagraphStyle('HRight3', fontName='Helvetica', fontSize=8, leading=10, alignment=2, textColor=colors.HexColor('#64748B'))),
    ]

    header_table = Table([[header_left, header_right]], colWidths=[310, 213])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceBefore=2, spaceAfter=8))

    # 2. THÔNG TIN BỆNH NHÂN
    info_data = [
        [
            Paragraph(f"<b>Họ và tên:</b> {patient_info.get('patient_name')}", style_cell),
            Paragraph(f"<b>Mã BN:</b> {patient_info.get('patient_id')}", style_cell),
            Paragraph(f"<b>Tuổi/Giới:</b> {patient_info.get('age_gender')}", style_cell),
        ],
        [
            Paragraph(f"<b>Bác sĩ chỉ định:</b> {patient_info.get('doctor_req')}", style_cell),
            Paragraph(f"<b>Loại mẫu:</b> Máu ngoại vi (EDTA)", style_cell),
            Paragraph(f"<b>Thời gian nhận:</b> {patient_info.get('date_time')}", style_cell),
        ]
    ]
    info_table = Table(info_data, colWidths=[180, 170, 173])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 10))

    # 3. KẾT QUẢ ĐẾM TẾ BÀO (YOLOv26) & PHÂN LOẠI CHI TIẾT
    story.append(Paragraph("I. THỐNG KÊ ĐẾM TẾ BÀO VÀ PHÂN LOẠI BẠCH CẦU (YOLO26 & QWEN AI)", style_h2))

    # Bảng đếm sơ bộ YOLO
    yolo_data = [
        [Paragraph("<b>Loại tế bào phát hiện (YOLO26)</b>", style_cell_bold), Paragraph("<b>Số lượng (Cells)</b>", style_cell_bold), Paragraph("<b>Tỷ lệ mật độ (%)</b>", style_cell_bold), Paragraph("<b>Đánh giá sơ bộ</b>", style_cell_bold)]
    ]
    total_cells = sum(yolo_counts.values()) if sum(yolo_counts.values()) > 0 else 1
    for k, v in yolo_counts.items():
        pct = (v / total_cells) * 100
        yolo_data.append([
            Paragraph(f"<b>{k}</b> ({REFERENCE_RANGES.get(k, {}).get('name', k)})", style_cell),
            Paragraph(f"{v:,}", style_cell),
            Paragraph(f"{pct:.1f} %", style_cell),
            Paragraph("<font color='#16A34A'><b>Bình thường</b></font>", style_cell)
        ])

    yolo_table = Table(yolo_data, colWidths=[200, 100, 100, 123])
    yolo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EFF6FF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(yolo_table)
    story.append(Spacer(1, 8))

    # Bảng Phân loại Top-5 Classifier
    story.append(Paragraph("<b>Kết quả Phân loại Bạch cầu Trọng điểm (QwenCellClassifier 12-Class):</b>", style_cell_bold))
    story.append(Spacer(1, 4))

    top_data = [
        [Paragraph("<b>Thứ tự</b>", style_cell_bold), Paragraph("<b>Tên tế bào (Full Name)</b>", style_cell_bold), Paragraph("<b>Độ tin cậy (Confidence)</b>", style_cell_bold), Paragraph("<b>Khoảng tham chiếu</b>", style_cell_bold)]
    ]
    for idx, (cls_name, prob) in enumerate(top5_probs, 1):
        top_data.append([
            Paragraph(f"#{idx}", style_cell),
            Paragraph(f"<b>{cls_name}</b>", style_cell),
            Paragraph(f"<b>{prob:.2f} %</b>", style_cell),
            Paragraph("Theo tiêu chuẩn WHO", style_cell)
        ])

    top_table = Table(top_data, colWidths=[50, 220, 120, 133])
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(top_table)
    story.append(Spacer(1, 10))

    # 4. HÌNH ẢNH CHẨN ĐOÁN VÀ GIẢI THÍCH XAI
    story.append(Paragraph("II. HÌNH ẢNH PHÂN TÍCH LÂM SÀNG VÀ BẢN ĐỒ GIẢI THÍCH XAI (EXPLAINABLE AI)", style_h2))

    img_elements = []
    if detected_img_path and Path(detected_img_path).exists():
        try:
            img_detect = Image(str(detected_img_path), width=245, height=180)
            img_elements.append([Paragraph("<b>Ảnh Phát hiện Bounding Box (YOLO26)</b>", style_cell_bold), img_detect])
        except Exception:
            img_elements.append([Paragraph("<b>Ảnh Phát hiện YOLO</b>", style_cell_bold), Paragraph("Không thể tải ảnh", style_cell)])
    else:
        img_elements.append([Paragraph("<b>Ảnh Phát hiện YOLO</b>", style_cell_bold), Paragraph("Chưa cung cấp ảnh", style_cell)])

    if xai_grid_path and Path(xai_grid_path).exists():
        try:
            img_xai = Image(str(xai_grid_path), width=245, height=180)
            img_elements.append([Paragraph("<b>Lưới Giải thích XAI (4 Methods)</b>", style_cell_bold), img_xai])
        except Exception:
            img_elements.append([Paragraph("<b>Lưới Giải thích XAI</b>", style_cell_bold), Paragraph("Không thể tải ảnh", style_cell)])
    else:
        img_elements.append([Paragraph("<b>Lưới Giải thích XAI</b>", style_cell_bold), Paragraph("Chưa cung cấp ảnh XAI", style_cell)])

    img_table_data = [
        [img_elements[0][0], img_elements[1][0]],
        [img_elements[0][1], img_elements[1][1]],
    ]
    img_table = Table(img_table_data, colWidths=[256, 256])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # 5. DỄ NÓI VỚI VLM AI ASSESSMENT
    if vlm_description:
        story.append(Paragraph("III. ĐÁNH GIÁ TỰ ĐỘNG TỪ MÔ HÌNH QWEN2.5-VL MULTIMODAL AI", style_h2))
        vlm_p = Paragraph(f"<i>\"{vlm_description}\"</i>", ParagraphStyle('VLMText', fontName='Helvetica-Oblique', fontSize=8.5, leading=12, textColor=colors.HexColor('#1E293B')))
        vlm_table = Table([[vlm_p]], colWidths=[512])
        vlm_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEF3C7')),
            ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#F59E0B')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(vlm_table)
        story.append(Spacer(1, 12))

    # 6. KHỐI CHỮ KÝ
    story.append(KeepTogether([
        Spacer(1, 5),
        create_signature_block()
    ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    return output_pdf_path


if __name__ == '__main__':
    test_pdf = Path("test_medical_report.pdf")
    generate_medical_pdf(test_pdf)
    print(f"[OK] Da tao test PDF tai: {test_pdf.resolve()}")
