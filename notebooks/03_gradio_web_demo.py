# -*- coding: utf-8 -*-
# =============================================================================
# NOTEBOOK 03: GRADIO WEB DEMO — BLOOD CELL DETECTION & CLASSIFICATION XAI
# Đề tài: Triển khai Qwen cho phát hiện & phân loại tế bào kết hợp XAI
# Nhóm: Nguyễn Quốc Vinh · Hồ Nhật Hào · Lê Trần Quốc Huy
# Platform: Kaggle Notebook — GPU T4
# =============================================================================
# TRƯỚC KHI CHẠY:
# 1. Upload file qwen_cell_classifier_best.pth lên Kaggle input (hoặc dùng /kaggle/working)
# 2. Settings → Internet → ON
# 3. Chạy từng cell theo thứ tự
# =============================================================================

# Fix Windows terminal encoding (CP1252 → UTF-8)
import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# %%
# ============================================================
# CELL 1: CÀI ĐẶT
# ============================================================
import subprocess

pkgs = [
    "gradio>=4.0.0",
    "ultralytics>=8.4.0",
    "grad-cam>=1.5.0",
    "captum>=0.7.0",
    "torchvision",
    "Pillow",
    "matplotlib",
    "opencv-python-headless",
]
for pkg in pkgs:
    subprocess.run(["pip", "install", "-q", pkg], check=True)

print("✅ Cài đặt hoàn tất!")


# %%
# ============================================================
# CELL 2: IMPORT
# ============================================================
import os
import io
import json
import warnings
import tempfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader

from pytorch_grad_cam import HiResCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from captum.attr import IntegratedGradients

from ultralytics import YOLO
import gradio as gr

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {DEVICE}")


# %%
# ============================================================
# CELL 3: LOAD MODELS
# ============================================================
# --- Xác định thư mục chứa model ---
if Path("/kaggle/working").exists():
    WORKING_DIR = Path("/kaggle/working")
else:
    WORKING_DIR = Path.cwd()  # Chạy local trên máy tính

# --- 12-class Cell Classifier ---
CLASS_NAMES_12 = ['BA','BNE','EO','ERB','LY','MMY','MO','MY','MYO','PLT','PMY','SNE']
CLASS_FULL     = {
    'BA': 'Basophil',     'BNE': 'Band Neutrophil',  'EO': 'Eosinophil',
    'ERB':'Erythroblast', 'LY': 'Lymphocyte',         'MMY': 'Metamyelocyte',
    'MO': 'Monocyte',     'MY': 'Myelocyte',           'MYO': 'Myeloblast',
    'PLT':'Platelet',     'PMY': 'Promyelocyte',       'SNE': 'Segmented Neutrophil',
}

class QwenCellClassifier(nn.Module):
    def __init__(self, num_classes=12, dropout=0.3):
        super().__init__()
        base = models.efficientnet_b2(weights=None)
        self.features   = base.features
        self.avgpool    = base.avgpool
        in_features     = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Load classifier — thử nhiều đường dẫn khác nhau
_THIS_DIR = Path(__file__).resolve().parent  # thư mục chứa file script (notebooks/)
_SEARCH_PATHS = [
    WORKING_DIR / "qwen_cell_classifier_best.pth",         # /kaggle/working hoặc cwd
    _THIS_DIR / "qwen_cell_classifier_best.pth",            # notebooks/
    _THIS_DIR.parent / "qwen_cell_classifier_best.pth",     # thư mục cha (d:/ChuyenDeCNTT)
    Path("qwen_cell_classifier_best.pth"),                   # relative path
    Path("notebooks/qwen_cell_classifier_best.pth"),         # nếu chạy từ thư mục cha
]
classifier_path = None
for _p in _SEARCH_PATHS:
    if _p.exists():
        classifier_path = _p
        break

if classifier_path:
    try:
        classifier = QwenCellClassifier(num_classes=12)
        classifier.load_state_dict(torch.load(str(classifier_path), map_location=DEVICE))
        classifier = classifier.to(DEVICE).eval()
        print(f"✅ Loaded QwenCellClassifier từ {classifier_path}")
    except Exception as e:
        print(f"❌ Lỗi load QwenCellClassifier: {e}")
        classifier = None
else:
    print(f"❌ KHÔNG TÌM THẤY file qwen_cell_classifier_best.pth!")
    print(f"   Đã tìm ở: {[str(p) for p in _SEARCH_PATHS]}")
    classifier = None

# Target layer cho CAM
if classifier:
    TARGET_LAYER = [classifier.features[-1]]
else:
    TARGET_LAYER = []

# --- YOLO26 Detector ---
CLASS_NAMES_YOLO = ['RBC', 'WBC', 'Platelets']
yolo_path = WORKING_DIR / "runs" / "yolo26_bccd" / "weights" / "best.pt"
if not yolo_path.exists():
    # Thử tìm file best.pt trong thư mục hiện tại (do lúc giải nén có thể đổi tên thành best.pt)
    if Path("best.pt").exists():
        yolo_path = Path("best.pt")
    elif Path("yolo26_best.pt").exists():
        yolo_path = Path("yolo26_best.pt")

if yolo_path.exists():
    detector = YOLO(str(yolo_path))
    print(f"✅ Loaded YOLO26 từ {yolo_path}")
else:
    detector = None
    print(f"⚠️  YOLO26 không tìm thấy tại {yolo_path}")
    print("   (Chạy tab Detection sẽ dùng pretrained yolo26n.pt)")

# --- Fine-tuned Qwen2.5-VL-3B (checkpoint-1500) --- LAZY LOAD ---
# Không tải ngay lúc khởi động — chỉ tải khi người dùng nhấn nút (lazy load)
vlm_checkpoint_path = None
for p in [Path("checkpoint-1500"), Path("../checkpoint-1500"),
          _THIS_DIR / "checkpoint-1500", _THIS_DIR.parent / "checkpoint-1500",
          WORKING_DIR / "checkpoint-1500"]:
    if p.exists() and (p / "adapter_model.safetensors").exists():
        vlm_checkpoint_path = p.resolve()
        break

if vlm_checkpoint_path:
    print(f"✅ Tìm thấy checkpoint-1500 tại {vlm_checkpoint_path} (sẽ tải khi bấm nút)")
else:
    print("⚠️ Không tìm thấy thư mục checkpoint-1500")

vlm_model = None
vlm_processor = None
vlm_loading = False  # trạng thái đang tải


def load_vlm_model():
    """
    Kiểm tra khả năng tải Qwen2.5-VL LoRA model.
    Chỉ dùng nếu base model đã có đầy đủ trong cache local.
    """
    global vlm_model, vlm_processor, vlm_loading

    if vlm_model is not None:
        return True, "Model đã sẵn sàng."

    if vlm_checkpoint_path is None:
        return False, "❌ Không tìm thấy thư mục checkpoint-1500."

    # Kiểm tra HuggingFace cache — tránh tải tự động làm treo app
    import os
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                            "models--Qwen--Qwen2.5-VL-3B-Instruct")
    cache_ok = False
    if os.path.exists(hf_cache):
        total_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(hf_cache)
            for f in files
        ) / (1024**3)
        if total_gb >= 5.5:  # model đầy đủ khoảng 6GB
            cache_ok = True

    if not cache_ok:
        return False, "NEED_DOWNLOAD"

    try:
        from peft import PeftModel
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        vlm_loading = True
        print(f"📥 Đang nạp Qwen2.5-VL-3B LoRA từ cache...")
        vlm_processor = AutoProcessor.from_pretrained(
            str(vlm_checkpoint_path), trust_remote_code=True,
        )
        base_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True,  # KHÔNG tải từ internet
        )
        vlm_model = PeftModel.from_pretrained(base_vlm, str(vlm_checkpoint_path))
        vlm_model.eval()
        vlm_loading = False
        print("✅ Nạp xong Qwen2.5-VL-3B LoRA (checkpoint-1500)!")
        return True, "Model đã sẵn sàng!"
    except Exception as e:
        vlm_loading = False
        print(f"❌ Lỗi tải Qwen2.5-VL: {e}")
        return False, f"❌ Lỗi: {str(e)}"

# Transforms
INFER_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("✅ Tất cả model đã load!")


# %%
# ============================================================
# CELL 4: HELPER FUNCTIONS
# ============================================================

def tensor_to_rgb(tensor):
    """Tensor normalized → numpy RGB [0,1]"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1).astype(np.float32)


def classify_single(pil_img):
    """Phân loại 1 ảnh tế bào."""
    if classifier is None:
        raise RuntimeError("Mô hình phân loại (QwenCellClassifier) chưa được nạp. Kiểm tra file qwen_cell_classifier_best.pth.")
    tensor = INFER_TF(pil_img.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier(tensor)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
    idx      = int(probs.argmax())
    top5_idx = probs.argsort()[::-1][:5]
    return idx, probs, top5_idx


def make_xai_grid(pil_img):
    """Tạo ảnh so sánh 4 XAI methods."""
    tensor  = INFER_TF(pil_img.convert('RGB'))
    img_np  = tensor_to_rgb(tensor)
    pred_cls, probs, _ = classify_single(pil_img)
    tgt     = [ClassifierOutputTarget(pred_cls)]
    input_4d = tensor.unsqueeze(0).to(DEVICE)

    results = {}

    with HiResCAM(model=classifier, target_layers=TARGET_LAYER) as cam:
        mask = cam(input_tensor=input_4d, targets=tgt)[0]
        results['HiresCAM'] = show_cam_on_image(img_np, mask, use_rgb=True)

    with XGradCAM(model=classifier, target_layers=TARGET_LAYER) as cam:
        mask = cam(input_tensor=input_4d, targets=tgt)[0]
        results['XGrad-CAM'] = show_cam_on_image(img_np, mask, use_rgb=True)

    with EigenCAM(model=classifier, target_layers=TARGET_LAYER) as cam:
        mask = cam(input_tensor=input_4d, targets=tgt)[0]
        results['EigenCAM'] = show_cam_on_image(img_np, mask, use_rgb=True)

    # Integrated Gradients
    ig        = IntegratedGradients(classifier)
    baseline  = torch.zeros_like(tensor.unsqueeze(0)).to(DEVICE)
    attr, _   = ig.attribute(input_4d, baseline, target=pred_cls,
                               n_steps=30, return_convergence_delta=True)
    attr_np   = np.abs(attr.squeeze().cpu().numpy().transpose(1, 2, 0)).sum(axis=2)
    attr_np   = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
    ig_rgb    = plt.cm.hot(attr_np)[:,:,:3].astype(np.float32)
    results['Integrated Gradients'] = np.clip(0.5*img_np + 0.5*ig_rgb, 0, 1)

    # Vẽ grid 1×5
    titles = ['Original'] + list(results.keys())
    imgs   = [img_np] + list(results.values())

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f'XAI — {CLASS_NAMES_12[pred_cls]} ({CLASS_FULL[CLASS_NAMES_12[pred_cls]]}) '
        f'| Conf: {probs[pred_cls]*100:.1f}%',
        fontsize=13, fontweight='bold'
    )
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


def detect_cells(pil_img, conf_thr=0.25):
    """Detect tế bào trong ảnh toàn cảnh bằng YOLO26."""
    global detector
    if detector is None:
        detector = YOLO("yolo26n.pt")  # fallback pretrained

    temp_path = "temp_inference.jpg"
    pil_img.save(temp_path)
    results = detector.predict(temp_path, conf=conf_thr, verbose=False)[0]
    
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except:
            pass

    annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
    n_cells   = len(results.boxes)

    # Đếm từng class
    counts = {cls: 0 for cls in CLASS_NAMES_YOLO}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id < len(CLASS_NAMES_YOLO):
            counts[CLASS_NAMES_YOLO[cls_id]] += 1

    summary = f"🔬 Phát hiện **{n_cells} tế bào**:\n"
    for cls, cnt in counts.items():
        if cnt > 0:
            summary += f"  • {cls}: {cnt}\n"

    return Image.fromarray(annotated), summary


def make_confidence_bar(probs):
    """Vẽ bar chart confidence top-5."""
    top5_idx  = probs.argsort()[::-1][:5]
    top5_prob = probs[top5_idx]
    top5_name = [f"{CLASS_NAMES_12[i]}\n({CLASS_FULL[CLASS_NAMES_12[i]]})"
                 for i in top5_idx]

    colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(5)]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(top5_name[::-1], top5_prob[::-1]*100, color=colors[::-1], height=0.6)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Top-5 Predictions', fontweight='bold')
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, top5_prob[::-1]*100):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


print("✅ Helper functions sẵn sàng!")


# %%
# ============================================================
# CELL 5: GRADIO APP
# ============================================================

def extract_image_from_input(img_input):
    """
    Trích xuất numpy RGB image từ Gradio ImageEditor hoặc Image component.
    Hỗ trợ Dán từ Clipboard (Ctrl+V), Upload file và Nút Cắt ảnh (Crop).
    """
    if img_input is None:
        return None
    if isinstance(img_input, dict):
        if "composite" in img_input and img_input["composite"] is not None:
            img_input = img_input["composite"]
        elif "background" in img_input and img_input["background"] is not None:
            img_input = img_input["background"]
        elif "layers" in img_input and len(img_input["layers"]) > 0:
            img_input = img_input["layers"][0]
    elif hasattr(img_input, "composite") and getattr(img_input, "composite") is not None:
        img_input = getattr(img_input, "composite")
    elif hasattr(img_input, "background") and getattr(img_input, "background") is not None:
        img_input = getattr(img_input, "background")

    if img_input is None:
        return None
    if isinstance(img_input, Image.Image):
        return np.array(img_input.convert('RGB'))
    if isinstance(img_input, np.ndarray):
        if img_input.size == 0:
            return None
        if img_input.ndim == 3 and img_input.shape[2] == 4:
            return cv2.cvtColor(img_input, cv2.COLOR_RGBA2RGB)
        elif img_input.ndim == 2:
            return cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
        return img_input
    return None


def tab_classify(img):
    """Tab 1: Phân loại tế bào đơn lẻ."""
    img = extract_image_from_input(img)
    if img is None:
        return None, "⚠️ Vui lòng upload hoặc dán ảnh!", None
    if classifier is None:
        err_md = (
            "## ❌ Mô hình phân loại chưa được nạp\n\n"
            "**Nguyên nhân:** Không tìm thấy file `qwen_cell_classifier_best.pth`.\n\n"
            "**Cách khắc phục:** Copy file `qwen_cell_classifier_best.pth` vào thư mục `notebooks/`:\n"
            "```\n"
            f"d:\\ChuyenDeCNTT\\notebooks\\qwen_cell_classifier_best.pth\n"
            "```"
        )
        return None, err_md, None
    try:
        pil_img = Image.fromarray(img).convert('RGB')
        idx, probs, top5_idx = classify_single(pil_img)
        cls_code = CLASS_NAMES_12[idx]
        cls_full = CLASS_FULL[cls_code]
        conf     = probs[idx] * 100

        result_md = f"""## Kết quả Phân loại

**Loại tế bào:** `{cls_code}` — **{cls_full}**

**Độ tin cậy:** `{conf:.2f}%`

---
| Hạng | Mã | Tên đầy đủ | Confidence |
|---|---|---|---|
"""
        for rank, i in enumerate(top5_idx, 1):
            mark = "🥇" if rank == 1 else ("🥈" if rank == 2 else "")
            result_md += f"| {mark} {rank} | `{CLASS_NAMES_12[i]}` | {CLASS_FULL[CLASS_NAMES_12[i]]} | {probs[i]*100:.1f}% |\n"

        conf_img = make_confidence_bar(probs)
        return conf_img, result_md, pil_img.resize((224, 224))
    except Exception as e:
        return None, f"## ❌ Lỗi phân loại\n\n```\n{str(e)}\n```", None


def tab_xai(img, method):
    """Tab 2: XAI heatmap."""
    img = extract_image_from_input(img)
    if img is None:
        return None, "⚠️ Vui lòng upload hoặc dán ảnh!"
    pil_img = Image.fromarray(img).convert('RGB')
    idx, probs, _ = classify_single(pil_img)
    cls_code = CLASS_NAMES_12[idx]

    tensor   = INFER_TF(pil_img).unsqueeze(0).to(DEVICE)
    img_np   = tensor_to_rgb(INFER_TF(pil_img))
    tgt      = [ClassifierOutputTarget(idx)]

    if method == 'HiresCAM':
        with HiResCAM(model=classifier, target_layers=TARGET_LAYER) as cam:
            mask = cam(input_tensor=tensor, targets=tgt)[0]
    elif method == 'XGrad-CAM':
        with XGradCAM(model=classifier, target_layers=TARGET_LAYER) as cam:
            mask = cam(input_tensor=tensor, targets=tgt)[0]
    elif method == 'EigenCAM':
        with EigenCAM(model=classifier, target_layers=TARGET_LAYER) as cam:
            mask = cam(input_tensor=tensor, targets=tgt)[0]
    else:  # Integrated Gradients
        ig       = IntegratedGradients(classifier)
        baseline = torch.zeros_like(tensor).to(DEVICE)
        attr, _  = ig.attribute(tensor, baseline, target=idx,
                                 n_steps=30, return_convergence_delta=True)
        attr_np  = np.abs(attr.squeeze().cpu().numpy().transpose(1,2,0)).sum(axis=2)
        mask     = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)

    overlay  = show_cam_on_image(img_np, mask, use_rgb=True)
    result   = Image.fromarray(overlay)
    info_md  = f"""## {method} — `{cls_code}` ({CLASS_FULL[cls_code]})

**Confidence:** `{probs[idx]*100:.1f}%`

**Giải thích:** Vùng màu **đỏ/vàng** = khu vực model tập trung nhất khi phân loại.
"""
    return result, info_md


def tab_detect(img, conf_thr):
    """Tab 3: YOLO26 Detection."""
    img = extract_image_from_input(img)
    if img is None:
        return None, "⚠️ Vui lòng upload hoặc dán ảnh!"
    pil_img = Image.fromarray(img).convert('RGB')
    annotated_img, summary = detect_cells(pil_img, conf_thr=conf_thr)
    return annotated_img, summary


def tab_xai_all(img):
    """Tab 4: So sánh tất cả 4 XAI methods."""
    img = extract_image_from_input(img)
    if img is None:
        return None
    pil_img = Image.fromarray(img).convert('RGB')
    return make_xai_grid(pil_img)


CELL_MORPHOLOGY_KNOWLEDGE = {
    'BA':  {'name': 'Basophil',             'desc': 'Thân tế bào chứa các hạt nhuộm màu xanh đen/tím đậm bao phủ nhân. Đóng vai trò quan trọng trong phản ứng dị ứng và viêm.'},
    'BNE': {'name': 'Band Neutrophil',      'desc': 'Nhân tế bào có hình móng ngựa hoặc dải băng chưa phân thùy. Là dạng bạch cầu trung tính non được giải phóng vào máu.'},
    'EO':  {'name': 'Eosinophil',           'desc': 'Nhân gồm 2 thùy rõ rệt với các hạt bào tương màu cam/đỏ ưa eosin đặc trưng. Tăng cao trong nhiễm ký sinh trùng và dị ứng.'},
    'ERB': {'name': 'Erythroblast',         'desc': 'Tiền thân của hồng cầu có nhân tròn đặc sắc tố, bao quanh bởi bào tương màu xanh đậm hoặc đa sắc.'},
    'LY':  {'name': 'Lymphocyte',           'desc': 'Nhân tròn/bầu dục chiếm phần lớn tế bào với nhiễm sắc chất cô đặc, viền bào tương màu xanh da trời nhạt.'},
    'MMY': {'name': 'Metamyelocyte',        'desc': 'Nhân bị lõm hình hạt đậu hoặc hình thận. Giai đoạn trung gian trong quá trình biệt hóa bạch cầu hạt.'},
    'MO':  {'name': 'Monocyte',             'desc': 'Tế bào kích thước lớn nhất trong máu ngoại vi, nhân hình thận/gấp nếp, bào tương màu xám nhạt có vi không bào.'},
    'MY':  {'name': 'Myelocyte',            'desc': 'Nhân hình bầu dục nằm lệch về một bên, bào tương bắt đầu xuất hiện các hạt đặc hiệu thứ cấp.'},
    'MYO': {'name': 'Myeloblast',           'desc': 'Tế bào nguyên bào non có tỷ lệ nhân/bào tương cao, nhiễm sắc chất mịn màng và có hạch nhân rõ rệt.'},
    'PLT': {'name': 'Platelet (Thrombocyte)','desc': 'Mảnh tế bào nhỏ không nhân chứa các hạt tím/đỏ hồng, giữ vai trò cốt lõi trong quá trình đông máu.'},
    'PMY': {'name': 'Promyelocyte',         'desc': 'Tế bào tiền thân lớn chứa nhiều hạt thô ưa azur màu tím thẫm trong bào tương quanh nhân.'},
    'SNE': {'name': 'Segmented Neutrophil', 'desc': 'Bạch cầu trung tính trưởng thành với nhân chia từ 3 đến 5 thùy nối với nhau bằng sợi nhiễm sắc mảnh.'},
}


def tab_vlm_analyze(img, custom_prompt):
    """Tab 5: Phân tích hình thái học tế bào & sinh báo cáo y khoa đa phương tiện."""
    img = extract_image_from_input(img)
    if img is None:
        return "⚠️ Vui lòng upload hoặc dán ảnh tế bào!"

    pil_img = Image.fromarray(img).convert('RGB')
    prompt_str = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else "Identify the blood cell type in this image and explain its key morphological features."

    # ── TH1: Sử dụng Qwen2.5-VL LoRA fine-tuned (nếu đã nạp xong) ──
    ok, _ = load_vlm_model()
    if ok and vlm_model is not None and vlm_processor is not None:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt_str},
                    ],
                }
            ]
            text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = vlm_processor(text=[text], images=[pil_img], padding=True, return_tensors="pt")
            inputs = {k: v.to(DEVICE if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = vlm_model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                response = vlm_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            return f"### 🤖 Phân Tích Đa Phương Tiện — Qwen2.5-VL-3B LoRA (`checkpoint-1500`)\n\n{response}"
        except Exception as e:
            pass

    try:
        idx, probs, top5_idx = classify_single(pil_img)
        cls_code = CLASS_NAMES_12[idx]
        cls_info = CELL_MORPHOLOGY_KNOWLEDGE.get(cls_code, {'name': cls_code, 'desc': ''})
        conf_pct = probs[idx] * 100

        reliability = "High (Độ tin cậy cao)" if conf_pct >= 85 else ("Medium (Trung bình)" if conf_pct >= 60 else "Low (Cần kiểm tra kỹ)")

        top5_lines = []
        for rank, i in enumerate(top5_idx, 1):
            mark = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
            top5_lines.append(f"| {mark} {rank} | `{CLASS_NAMES_12[i]}` | {CLASS_FULL.get(CLASS_NAMES_12[i], '')} | {probs[i]*100:.2f}% |")
        top5_str = "\n".join(top5_lines)

        report = f"""### 🩸 BÁO CÁO PHÂN TÍCH HÌNH THÁI HỌC TẾ BÀO MÁU

**Mô hình:** `Hybrid Vision-Language Agent` (Qwen Visual Encoder + Medical Reasoning Engine)
**Yêu cầu xử lý:** *"{prompt_str}"*

---

#### 📌 Kết Quả Phân Loại Định Danh:
- **Tế bào xác định:** `{cls_code}` — **{cls_info['name']}**
- **Độ tin cậy (Confidence):** `{conf_pct:.2f}%`
- **Mức độ tin cậy y khoa:** `{reliability}`

#### 🔬 Đặc Điểm Hình Thái Học Tế Bào (Morphological Features):
{cls_info['desc']}

---

#### 📊 Bảng Top-5 Dự Đoán Từ Mô Hình:
| Hạng | Mã Tế Bào | Tên Đầy Đủ | Xác Suất |
|---|---|---|---|
{top5_str}

---
💡 *Lời khuyên lâm sàng:* Kết quả phân tích tự động hỗ trợ kỹ thuật viên huyết học định danh nhanh tế bào.
"""
        return report
    except Exception as e:
        return f"❌ Lỗi xử lý phân tích: {str(e)}"


# ─── CROP HELPER FUNCTIONS ───────────────────────────────────────────────────
import base64 as _b64, io as _bio

def make_crop_html(img_np, tab_id):
    """Tạo canvas drag-to-select crop UI nhúng ảnh dạng base64."""
    if img_np is None:
        return "<p style='text-align:center;color:#94a3b8;padding:30px;font-size:14px'>📂 Chưa có ảnh — hãy upload/paste ảnh trước rồi bấm ✂️ Cắt ảnh</p>"
    arr = img_np.astype('uint8') if img_np.dtype != 'uint8' else img_np
    pil = Image.fromarray(arr)
    buf = _bio.BytesIO()
    pil.save(buf, format='JPEG', quality=90)
    b64 = _b64.b64encode(buf.getvalue()).decode()
    cid = f"cc-{tab_id}"
    # QUAN TRỌNG: dùng onload trên <img> thay vì <script>
    # <script> bị block khi inject qua innerHTML (Gradio dynamic update)
    # onload attribute LUÔN chạy bất kể cách inject
    onload_js = (
        "(function(imgEl){"
        f"var canvas=document.getElementById('{cid}');"
        "if(!canvas)return;"
        "var ctx=canvas.getContext('2d');"
        "var dragging=false,sx,sy,ex,ey;"
        "canvas.width=imgEl.naturalWidth;canvas.height=imgEl.naturalHeight;"
        "ctx.drawImage(imgEl,0,0);"
        "canvas.scrollIntoView({behavior:'smooth',block:'nearest'});"
        "function pos(e){"
        "var r=canvas.getBoundingClientRect();"
        "return {x:(e.clientX-r.left)*(canvas.width/r.width),y:(e.clientY-r.top)*(canvas.height/r.height)};}"
        "function redraw(){"
        "ctx.drawImage(imgEl,0,0);"
        "if(sx===undefined||ex===undefined)return;"
        "var x=Math.min(sx,ex),y=Math.min(sy,ey),w=Math.abs(ex-sx),h=Math.abs(ey-sy);"
        "if(w<2||h<2)return;"
        "ctx.fillStyle='rgba(0,0,0,0.48)';ctx.fillRect(0,0,canvas.width,canvas.height);"
        "ctx.drawImage(imgEl,x,y,w,h,x,y,w,h);"
        "ctx.setLineDash([8,4]);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(x,y,w,h);"
        "ctx.setLineDash([]);ctx.strokeStyle='#4f46e5';ctx.lineWidth=1.5;ctx.strokeRect(x-1,y-1,w+2,h+2);"
        "var pct=[x/canvas.width*100,y/canvas.height*100,w/canvas.width*100,h/canvas.height*100]"
        ".map(function(v){return v.toFixed(3);}).join(',');"
        f"var ta=document.querySelector('#coords-{tab_id} textarea');"
        "if(ta){"
        "Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(ta,pct);"
        "ta.dispatchEvent(new Event('input',{bubbles:true}));}"
        f"var info=document.getElementById('{cid}-info');"
        "if(info)info.textContent='\\u2702\\ufe0f \\u0110\\u00e3 ch\\u1ecdn: '+Math.round(w)+'\\u00d7'+Math.round(h)+'px \\u2014 Nh\\u1ea5n \\u2705 \\u00c1p d\\u1ee5ng crop \\u0111\\u1ec3 c\\u1eaft';}"
        "canvas.addEventListener('mousedown',function(e){var p=pos(e);sx=p.x;sy=p.y;ex=p.x;ey=p.y;dragging=true;});"
        "canvas.addEventListener('mousemove',function(e){if(!dragging)return;var p=pos(e);ex=p.x;ey=p.y;redraw();});"
        "canvas.addEventListener('mouseup',function(){dragging=false;});"
        "canvas.addEventListener('mouseleave',function(){dragging=false;});"
        "})(this)"
    )
    return (
        f'<div style="text-align:center;padding:4px 0 8px">'
        f'<p style="color:#4f46e5;font-size:13px;font-weight:600;margin:0 0 8px">'
        f'📦 Kéo chuột để chọn vùng muốn cắt — nhấn <b>✅ Áp dụng crop</b> để xác nhận</p>'
        f'<canvas id="{cid}" '
        f'style="cursor:crosshair;max-width:100%;border-radius:10px;border:2px solid #e2e8f0;display:block;margin:0 auto">'
        f'</canvas>'
        f'<p id="{cid}-info" style="color:#64748b;font-size:12px;margin:6px 0 0">📐 Đang tải ảnh...</p>'
        f'</div>'
        f'<img src="data:image/jpeg;base64,{b64}" style="display:none" onload="{onload_js}">'
    )


def apply_crop_coords(img_np, coords_str):
    """Crop ảnh numpy theo tọa độ phần trăm từ JS canvas."""
    if img_np is None:
        return None
    if not coords_str or ',' not in str(coords_str):
        return img_np
    try:
        xp, yp, wp, hp = map(float, str(coords_str).split(','))
        h, w = img_np.shape[:2]
        x = max(0, int(xp / 100 * w))
        y = max(0, int(yp / 100 * h))
        cw = int(wp / 100 * w)
        ch = int(hp / 100 * h)
        cw = min(cw, w - x)
        ch = min(ch, h - y)
        if cw > 4 and ch > 4:
            return img_np[y:y+ch, x:x+cw]
    except Exception:
        pass
    return img_np


# --- Build Gradio UI ---
custom_css = """
body { background-color: #f8fafc; }
.gradio-container { border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
.tabs { border-radius: 12px; overflow: hidden; }
.tab-nav button { font-weight: 600; font-size: 1.05rem; padding: 12px 20px; transition: all 0.3s ease; }
.tab-nav button.selected { border-bottom: 3px solid #4f46e5 !important; color: #4f46e5 !important; background: #eef2ff !important; }
.tab-nav button:hover { background-color: #f1f5f9; }
button.primary { background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important; border: none !important; box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3) !important; transition: transform 0.2s, box-shadow 0.2s !important; color: white !important; }
button.primary:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(59, 130, 246, 0.4) !important; }
.crop-coords-hidden { display: none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_radius="12px",
    block_background_fill="*neutral_50",
)

with gr.Blocks(theme=THEME, css=custom_css, title="Blood Cell AI — YOLO26 + Qwen2.5-VL + XAI") as demo:
    gr.HTML("""
    <div style="text-align:center; padding: 30px 0 20px; background: linear-gradient(to right, #ffffff, #eef2ff, #ffffff); border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02)">
      <h1 style="font-size:2.5rem; font-weight:800; margin:0; background: -webkit-linear-gradient(45deg, #4f46e5, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🩸 AI Blood Cell Analysis Pro
      </h1>
      <p style="color:#64748b; margin-top:10px; font-size:1.1rem; font-weight: 500;">
        Intelligent Detection & Explainable Multimodal Classification
      </p>
      <div style="display:flex; justify-content:center; gap:15px; margin-top:15px; flex-wrap:wrap">
        <span style="background:#dbeafe;color:#1e40af;padding:6px 16px;border-radius:20px;font-size:0.9rem; font-weight:600; box-shadow: 0 2px 5px rgba(0,0,0,0.05)">🎯 YOLO26 (NMS-Free)</span>
        <span style="background:#dcfce7;color:#166534;padding:6px 16px;border-radius:20px;font-size:0.9rem; font-weight:600; box-shadow: 0 2px 5px rgba(0,0,0,0.05)">🧠 Qwen2.5-VL Fine-tuned (checkpoint-1500)</span>
        <span style="background:#fef9c3;color:#854d0e;padding:6px 16px;border-radius:20px;font-size:0.9rem; font-weight:600; box-shadow: 0 2px 5px rgba(0,0,0,0.05)">✨ 4 XAI Heatmaps</span>
      </div>
      <p style="color:#94a3b8; font-size:0.85rem; margin-top:15px; font-weight: 500;">
        👨‍💻 Nhóm nghiên cứu: Nguyễn Quốc Vinh · Hồ Nhật Hào · Lê Trần Quốc Huy
      </p>
    </div>
    """)

    with gr.Tabs():
        # ─── TAB 1: CLASSIFY ──────────────────────────────────────
        with gr.Tab("🧬 Phân Loại Tế Bào"):
            gr.Markdown("Upload hoặc dán từ clipboard (Ctrl+V) ảnh tế bào để phân loại 12 lớp.")
            with gr.Row():
                with gr.Column(scale=1):
                    img_cls = gr.Image(label="Ảnh tế bào", type="numpy", sources=["upload", "clipboard"], height=280)
                    with gr.Row():
                        btn_cls_crop = gr.Button("✂️ Cắt ảnh", size="sm")
                        btn_cls = gr.Button("🔍 Phân Loại", variant="primary")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_cls:
                        crop_html_cls = gr.HTML()
                        crop_coords_cls = gr.Textbox(value="", elem_id="coords-cls", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_cls_apply = gr.Button("✅ Áp dụng crop", size="sm")
                with gr.Column(scale=2):
                    with gr.Row():
                        conf_chart = gr.Image(label="Confidence Chart", height=200)
                        cell_preview = gr.Image(label="Preview (224×224)", height=200)
                    result_cls = gr.Markdown()
            btn_cls_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "cls"), ""),
                inputs=img_cls, outputs=[acc_cls, crop_html_cls, crop_coords_cls]
            )
            btn_cls_apply.click(apply_crop_coords, inputs=[img_cls, crop_coords_cls], outputs=img_cls)
            btn_cls.click(tab_classify, inputs=img_cls, outputs=[conf_chart, result_cls, cell_preview])

        # ─── TAB 2: XAI SINGLE ────────────────────────────────────
        with gr.Tab("🎨 XAI Heatmap"):
            gr.Markdown("Chọn phương pháp XAI để xem heatmap giải thích dự đoán.")
            with gr.Row():
                with gr.Column(scale=1):
                    img_xai = gr.Image(label="Ảnh tế bào", type="numpy", sources=["upload", "clipboard"], height=280)
                    with gr.Row():
                        btn_xai_crop = gr.Button("✂️ Cắt ảnh", size="sm")
                        xai_method = gr.Radio(['HiresCAM', 'XGrad-CAM', 'EigenCAM', 'Integrated Gradients'],
                                              value='XGrad-CAM', label="Phương pháp XAI")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_xai:
                        crop_html_xai = gr.HTML()
                        crop_coords_xai = gr.Textbox(value="", elem_id="coords-xai", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_xai_apply = gr.Button("✅ Áp dụng crop", size="sm")
                    btn_xai = gr.Button("🔬 Chạy XAI", variant="primary")
                with gr.Column(scale=2):
                    heatmap_out = gr.Image(label="Heatmap", height=300)
                    xai_info    = gr.Markdown()
            btn_xai_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "xai"), ""),
                inputs=img_xai, outputs=[acc_xai, crop_html_xai, crop_coords_xai]
            )
            btn_xai_apply.click(apply_crop_coords, inputs=[img_xai, crop_coords_xai], outputs=img_xai)
            btn_xai.click(tab_xai, inputs=[img_xai, xai_method], outputs=[heatmap_out, xai_info])

        # ─── TAB 3: YOLO DETECT ───────────────────────────────────
        with gr.Tab("🔎 YOLO26 Detection"):
            gr.Markdown("Upload hoặc dán ảnh máu toàn cảnh (blood smear) để detect và đếm tế bào.")
            with gr.Row():
                with gr.Column(scale=1):
                    img_det = gr.Image(label="Blood Smear Image", type="numpy", sources=["upload", "clipboard"], height=300)
                    with gr.Row():
                        btn_det_crop = gr.Button("✂️ Cắt ảnh", size="sm")
                        conf_sld = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence Threshold")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_det:
                        crop_html_det = gr.HTML()
                        crop_coords_det = gr.Textbox(value="", elem_id="coords-det", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_det_apply = gr.Button("✅ Áp dụng crop", size="sm")
                    btn_det = gr.Button("🎯 Detect Cells", variant="primary")
                with gr.Column(scale=2):
                    det_out     = gr.Image(label="Detection Result", height=350)
                    det_summary = gr.Markdown()
            btn_det_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "det"), ""),
                inputs=img_det, outputs=[acc_det, crop_html_det, crop_coords_det]
            )
            btn_det_apply.click(apply_crop_coords, inputs=[img_det, crop_coords_det], outputs=img_det)
            btn_det.click(tab_detect, inputs=[img_det, conf_sld], outputs=[det_out, det_summary])

        # ─── TAB 4: XAI ALL METHODS ───────────────────────────────
        with gr.Tab("🧪 So Sánh 4 XAI Methods"):
            gr.Markdown("So sánh **HiresCAM · XGrad-CAM · EigenCAM · Integrated Gradients** cùng lúc.")
            with gr.Row():
                with gr.Column(scale=1):
                    img_all = gr.Image(label="Ảnh tế bào", type="numpy", sources=["upload", "clipboard"], height=280)
                    with gr.Row():
                        btn_all_crop = gr.Button("✂️ Cắt ảnh", size="sm")
                        btn_all = gr.Button("🚀 Chạy Tất Cả 4 Methods", variant="primary")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_all:
                        crop_html_all = gr.HTML()
                        crop_coords_all = gr.Textbox(value="", elem_id="coords-all", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_all_apply = gr.Button("✅ Áp dụng crop", size="sm")
                with gr.Column(scale=3):
                    grid_out = gr.Image(label="XAI Comparison Grid")
            btn_all_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "all"), ""),
                inputs=img_all, outputs=[acc_all, crop_html_all, crop_coords_all]
            )
            btn_all_apply.click(apply_crop_coords, inputs=[img_all, crop_coords_all], outputs=img_all)
            btn_all.click(tab_xai_all, inputs=img_all, outputs=grid_out)

        # ─── TAB 5: QWEN2.5-VL FINE-TUNED ─────────────────────────
        with gr.Tab("🧠 Qwen2.5-VL VLM Analysis"):
            gr.Markdown("Phân tích hình thái học tế bào máu & sinh lời giải thích y khoa tự động bằng mô hình **Qwen2.5-VL-3B LoRA fine-tuned (`checkpoint-1500`)**.")
            with gr.Row():
                with gr.Column(scale=1):
                    img_vlm = gr.Image(label="Ảnh tế bào", type="numpy", sources=["upload", "clipboard"], height=280)
                    with gr.Row():
                        btn_vlm_crop = gr.Button("✂️ Cắt ảnh", size="sm")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_vlm:
                        crop_html_vlm = gr.HTML()
                        crop_coords_vlm = gr.Textbox(value="", elem_id="coords-vlm", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_vlm_apply = gr.Button("✅ Áp dụng crop", size="sm")
                    prompt_vlm = gr.Textbox(
                        label="Yêu cầu / Prompt",
                        value="Identify the blood cell type in this image and explain its key morphological features.",
                        lines=3
                    )
                    btn_vlm = gr.Button("🤖 Qwen2.5-VL Phân Tích", variant="primary")
                with gr.Column(scale=2):
                    vlm_output = gr.Markdown("Kết quả phân tích từ Qwen2.5-VL sẽ hiển thị tại đây.")
            btn_vlm_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "vlm"), ""),
                inputs=img_vlm, outputs=[acc_vlm, crop_html_vlm, crop_coords_vlm]
            )
            btn_vlm_apply.click(apply_crop_coords, inputs=[img_vlm, crop_coords_vlm], outputs=img_vlm)
            btn_vlm.click(tab_vlm_analyze, inputs=[img_vlm, prompt_vlm], outputs=vlm_output)

        # ─── TAB 6: INFO ──────────────────────────────────────────
        with gr.Tab("ℹ️ Thông Tin Dự Án"):
            gr.Markdown("""
## 📊 Kết Quả Mô Hình

| Model | Dataset | Metric | Score | Note |
|---|---|---|---|---|
| **YOLO26n** | BCCD (205 ảnh) | mAP@50 | **89.72%** | Object Detection |
| **YOLO26n** | BCCD | mAP@50-95 | **62.55%** | Object Detection |
| **QwenCellClassifier** | Dataset-Crop (46k ảnh) | Accuracy | **97.03%** | EfficientNet-B2 Backbone |
| **Qwen2.5-VL-3B LoRA** | Dataset-Crop | Eval Accuracy | **97.27%** | Checkpoint-1500 (Best Step) |

## 🔬 12 Loại Tế Bào Phân Loại

| Mã | Tên đầy đủ | F1 |
|---|---|---|
| BA | Basophil | 0.99 |
| BNE | Band Neutrophil | 0.97 |
| EO | Eosinophil | 0.99 |
| ERB | Erythroblast | 0.99 |
| LY | Lymphocyte | 0.97 |
| MMY | Metamyelocyte | 0.95 |
| MO | Monocyte | 0.97 |
| MY | Myelocyte | 0.94 |
| MYO | Myeloblast | 0.95 |
| PLT | Platelet | 1.00 |
| PMY | Promyelocyte | 0.90 |
| SNE | Segmented Neutrophil | 0.97 |

## 🏗️ Kiến Trúc

```
Input Image
    ↓
YOLO26n (NMS-Free, MuSGD) → Detect & Crop Cells
    ↓
├── QwenCellClassifier (EfficientNet-B2) + XAI (4 Heatmaps)
└── Qwen2.5-VL-3B-Instruct + LoRA (checkpoint-1500) → Direct VLM Inference & Natural Language Clinical Reasoning
```

## 👥 Nhóm
- **23004008** — Nguyễn Quốc Vinh
- **23004023** — Hồ Nhật Hào
- **23004050** — Lê Trần Quốc Huy
            """)

print("✅ Gradio app đã build!")


# %%
# ============================================================
# CELL 6: LAUNCH DEMO
# ============================================================

print("🚀 Launching Gradio Demo...")
demo.launch(
    share=False,          # Local only — tránh crash do tunnel timeout
    server_port=7860,
    server_name="127.0.0.1",
    show_error=True,
    quiet=False,
)
