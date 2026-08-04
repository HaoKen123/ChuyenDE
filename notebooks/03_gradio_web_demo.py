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

# Tim YOLO model theo thu tu uu tien (uu tien file da train tren BCCD)
_YOLO_SEARCH = [
    _THIS_DIR / "best.pt",                                          # notebooks/best.pt  (da train)
    _THIS_DIR.parent / "yolo26n.pt",                               # d:/ChuyenDeCNTT/yolo26n.pt
    _THIS_DIR.parent / "best.pt",                                   # d:/ChuyenDeCNTT/best.pt
    WORKING_DIR / "best.pt",                                        # thu muc lam viec
    WORKING_DIR / "runs" / "yolo26_bccd" / "weights" / "best.pt",  # kaggle runs
    Path("best.pt"),
    Path("yolo26n.pt"),
    Path("yolo26_best.pt"),
]

yolo_path = None
for _yp in _YOLO_SEARCH:
    if _yp.exists():
        yolo_path = _yp
        break

if yolo_path:
    detector = YOLO(str(yolo_path))
    print(f"✅ Loaded YOLO26 từ {yolo_path}")
else:
    detector = None
    print("⚠️  YOLO26 không tìm thấy model. Da tim o:")
    for _yp in _YOLO_SEARCH:
        print(f"   - {_yp}")

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ─── ROOT & BODY ─── */
:root {
  --primary:      #1d6fd8;
  --primary-glow: rgba(29,111,216,0.20);
  --accent:       #0ea5e9;
  --accent2:      #06b6d4;
  --rose:         #ef4444;
  --emerald:      #10b981;
  --gold:         #f59e0b;
  --bg-base:      #f0f6ff;
  --bg-card:      #ffffff;
  --bg-card-hov:  #f0f6ff;
  --border:       #dbeafe;
  --border-glow:  rgba(29,111,216,0.30);
  --text-primary: #0f172a;
  --text-muted:   #475569;
  --radius-lg:    16px;
  --radius-xl:    22px;
  --shadow-glow:  0 4px 24px rgba(29,111,216,0.10), 0 2px 8px rgba(0,0,0,0.06);
}

body, .gradio-container {
  background: #f0f6ff !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  color: #0f172a !important;
  min-height: 100vh;
}

/* Soft blue mesh background */
.gradio-container::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 50% at 15% 15%, rgba(29,111,216,0.07) 0%, transparent 70%),
    radial-gradient(ellipse 50% 40% at 85% 75%, rgba(14,165,233,0.06) 0%, transparent 70%),
    radial-gradient(ellipse 40% 60% at 60% 5%,  rgba(6,182,212,0.04)  0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}
@keyframes meshPulse {
  from { opacity: 0.8; }
  to   { opacity: 1.0; }
}

/* ─── GRADIO CONTAINER ─── */
.gradio-container > .main {
  position: relative;
  z-index: 1;
  max-width: 1400px !important;
  margin: 0 auto !important;
  padding: 0 24px 40px !important;
}

/* ─── TABS ─── */
.tabs { border-radius: var(--radius-xl) !important; overflow: visible !important; }

.tab-nav {
  background: #ffffff !important;
  border: 1px solid #dbeafe !important;
  border-radius: var(--radius-lg) !important;
  padding: 6px !important;
  margin-bottom: 20px !important;
  box-shadow: 0 2px 8px rgba(29,111,216,0.07) !important;
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.tab-nav button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  padding: 10px 18px !important;
  border-radius: 10px !important;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
  color: #64748b !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  letter-spacing: 0.01em;
}

.tab-nav button:hover {
  background: #eff6ff !important;
  color: #1d6fd8 !important;
  border-color: #bfdbfe !important;
  transform: translateY(-1px);
}

.tab-nav button.selected {
  background: linear-gradient(135deg, #1d6fd8 0%, #0ea5e9 100%) !important;
  color: #fff !important;
  border-color: transparent !important;
  box-shadow: 0 4px 16px rgba(29,111,216,0.30), 0 0 0 1px rgba(14,165,233,0.2) !important;
  transform: translateY(-1px);
}

/* ─── BLOCKS / PANELS ─── */
.block, .panel, .form, .gap {
  background: #ffffff !important;
  border: 1px solid #dbeafe !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: 0 2px 8px rgba(29,111,216,0.06) !important;
  transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
.block:hover { border-color: #93c5fd !important; box-shadow: 0 4px 16px rgba(29,111,216,0.10) !important; }

/* ─── LABELS & TEXT ─── */
span.svelte-1gfkn6j, label span, .label-wrap span {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: #64748b !important;
}

/* ─── INPUT COMPONENTS ─── */
textarea, input[type=text], input[type=number] {
  background: #f8faff !important;
  border: 1px solid #bfdbfe !important;
  border-radius: 10px !important;
  color: #0f172a !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.92rem !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus, input[type=text]:focus {
  border-color: #1d6fd8 !important;
  box-shadow: 0 0 0 3px rgba(29,111,216,0.15) !important;
  outline: none !important;
}

/* ─── BUTTONS ─── */
button.primary {
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.92rem !important;
  letter-spacing: 0.03em !important;
  background: linear-gradient(135deg, #1d6fd8 0%, #0ea5e9 100%) !important;
  border: none !important;
  border-radius: 12px !important;
  color: #fff !important;
  padding: 12px 28px !important;
  box-shadow: 0 4px 16px rgba(29,111,216,0.30) !important;
  transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
  position: relative;
  overflow: hidden;
}
button.primary::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.15), transparent);
  pointer-events: none;
}
button.primary:hover {
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 8px 24px rgba(29,111,216,0.40) !important;
}
button.primary:active { transform: translateY(0) scale(0.99) !important; }

button.secondary {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  background: #f0f6ff !important;
  border: 1px solid #bfdbfe !important;
  border-radius: 10px !important;
  color: #475569 !important;
  transition: all 0.2s ease !important;
}
button.secondary:hover {
  background: #dbeafe !important;
  border-color: #93c5fd !important;
  color: #1d6fd8 !important;
}

/* ─── SLIDER ─── */
.rangeslider {
  background: #dbeafe !important;
  border-radius: 100px !important;
}
.rangeslider__fill {
  background: linear-gradient(90deg, #1d6fd8, #0ea5e9) !important;
  border-radius: 100px !important;
}
.rangeslider__handle {
  background: #fff !important;
  border: 3px solid #1d6fd8 !important;
  box-shadow: 0 2px 12px rgba(29,111,216,0.25) !important;
}

/* ─── RADIO BUTTONS ─── */
.wrap.svelte-vt7nxi, .wrap {
  gap: 8px !important;
}
.wrap input[type=radio] + span {
  background: #f8faff !important;
  border: 1px solid #bfdbfe !important;
  border-radius: 10px !important;
  color: #475569 !important;
  padding: 7px 14px !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
  cursor: pointer;
}
.wrap input[type=radio]:checked + span {
  background: linear-gradient(135deg, #dbeafe, #e0f2fe) !important;
  border-color: #1d6fd8 !important;
  color: #1d6fd8 !important;
  box-shadow: 0 0 0 3px rgba(29,111,216,0.10) !important;
}

/* ─── IMAGE COMPONENTS ─── */
.image-container, .image-preview {
  background: #f8faff !important;
  border-radius: var(--radius-lg) !important;
  border: 1px dashed #93c5fd !important;
  transition: border-color 0.3s ease !important;
}
.image-container:hover { border-color: #1d6fd8 !important; }

/* ─── ACCORDION ─── */
.accordion > .label-wrap button {
  background: #eff6ff !important;
  border-radius: 10px !important;
  color: #475569 !important;
  font-weight: 600 !important;
}
.accordion > .label-wrap button:hover { color: #1d6fd8 !important; }

/* ─── MARKDOWN ─── */
.prose, .prose p, .md p {
  color: #475569 !important;
  font-size: 0.9rem !important;
  line-height: 1.7 !important;
}
.prose h2, .md h2 {
  color: #0f172a !important;
  font-size: 1.15rem !important;
  font-weight: 700 !important;
  margin-top: 1.5em !important;
  padding-bottom: 0.4em;
  border-bottom: 1px solid #dbeafe;
}
.prose h3, .md h3 {
  color: #1d6fd8 !important;
  font-size: 1rem !important;
  font-weight: 700 !important;
}
.prose h4, .md h4 {
  color: #0f172a !important;
  font-size: 0.9rem !important;
  font-weight: 700 !important;
}
.prose strong, .md strong { color: #0f172a !important; }
.prose code, .md code {
  background: #dbeafe !important;
  color: #1d6fd8 !important;
  padding: 2px 7px !important;
  border-radius: 6px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82em !important;
}
.prose pre, .md pre {
  background: #f0f6ff !important;
  border: 1px solid #bfdbfe !important;
  border-radius: 12px !important;
  color: #0369a1 !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ─── TABLES ─── */
.prose table, .md table {
  border-collapse: separate !important;
  border-spacing: 0 !important;
  width: 100% !important;
  border: 1px solid #dbeafe !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}
.prose th, .md th {
  background: #eff6ff !important;
  color: #1d6fd8 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 10px 14px !important;
  border-bottom: 1px solid #dbeafe !important;
}
.prose td, .md td {
  color: #475569 !important;
  padding: 9px 14px !important;
  border-bottom: 1px solid #f0f6ff !important;
  font-size: 0.88rem !important;
}
.prose tr:hover td, .md tr:hover td {
  background: #f0f6ff !important;
  color: #0f172a !important;
}
.prose tr:last-child td { border-bottom: none !important; }

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f6ff; }
::-webkit-scrollbar-thumb {
  background: #93c5fd;
  border-radius: 100px;
}
::-webkit-scrollbar-thumb:hover { background: #1d6fd8; }

/* ─── HIDDEN COORDS ─── */
.crop-coords-hidden { display: none !important; }

/* ─── STAT CARDS (injected via HTML) ─── */
.stat-card {
  background: #ffffff;
  border: 1px solid #dbeafe;
  border-radius: var(--radius-lg);
  padding: 20px 24px;
  box-shadow: 0 2px 8px rgba(29,111,216,0.07);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}
.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--card-accent, linear-gradient(90deg, #1d6fd8, #0ea5e9));
}
.stat-card:hover {
  border-color: #93c5fd;
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(29,111,216,0.15);
}

/* ─── PULSE ANIMATION ─── */
@keyframes pulse-ring {
  0%   { box-shadow: 0 0 0 0   rgba(29,111,216,0.4); }
  70%  { box-shadow: 0 0 0 12px rgba(29,111,216,0); }
  100% { box-shadow: 0 0 0 0   rgba(29,111,216,0); }
}
@keyframes slideInUp {
  from { opacity:0; transform:translateY(20px); }
  to   { opacity:1; transform:translateY(0);    }
}
.tabs > div > div { animation: slideInUp 0.35s ease forwards; }

/* ─── FOOTER ─── */
.footer-bar {
  text-align: center;
  padding: 28px 0 10px;
  border-top: 1px solid #dbeafe;
  margin-top: 30px;
  color: #94a3b8;
  font-size: 0.82rem;
}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#f0f6ff",
    body_background_fill_dark="#f0f6ff",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#dbeafe",
    block_border_width="1px",
    block_radius="16px",
    block_shadow="0 2px 8px rgba(29,111,216,0.07)",
    button_primary_background_fill="linear-gradient(135deg,#1d6fd8,#0ea5e9)",
    button_primary_background_fill_hover="linear-gradient(135deg,#1a5fbe,#0284c7)",
    button_primary_text_color="#fff",
    button_secondary_background_fill="#f0f6ff",
    button_secondary_border_color="#bfdbfe",
    button_secondary_text_color="#475569",
    input_background_fill="#f8faff",
    input_border_color="#bfdbfe",
    input_placeholder_color="#94a3b8",
    slider_color="#1d6fd8",
    body_text_color="#0f172a",
    body_text_color_subdued="#475569",
)

with gr.Blocks(theme=THEME, css=custom_css, title="Blood Cell AI — YOLO26 + Qwen2.5-VL + XAI") as demo:
    gr.HTML("""
    <!-- ═══════════════════ HERO HEADER ═══════════════════ -->
    <div style="
      position:relative; overflow:hidden;
      background: linear-gradient(135deg, #e0f2fe 0%, #eff6ff 50%, #f0f9ff 100%);
      border: 1px solid #bfdbfe;
      border-radius: 22px;
      padding: 48px 40px 40px;
      margin-bottom: 28px;
      box-shadow: 0 4px 24px rgba(29,111,216,0.12), 0 1px 4px rgba(29,111,216,0.06);
    ">
      <!-- soft orbs -->
      <div style="position:absolute;top:-80px;left:-80px;width:320px;height:320px;background:radial-gradient(circle,rgba(29,111,216,0.10) 0%,transparent 70%);pointer-events:none;"></div>
      <div style="position:absolute;bottom:-60px;right:-60px;width:280px;height:280px;background:radial-gradient(circle,rgba(14,165,233,0.08) 0%,transparent 70%);pointer-events:none;"></div>
      <div style="position:absolute;top:30px;right:200px;width:180px;height:180px;background:radial-gradient(circle,rgba(6,182,212,0.06) 0%,transparent 70%);pointer-events:none;"></div>

      <!-- badge row -->
      <div style="display:flex;justify-content:center;gap:10px;margin-bottom:22px;flex-wrap:wrap;">
        <span style="display:inline-flex;align-items:center;gap:6px;background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;padding:5px 14px;border-radius:100px;font-size:0.75rem;font-weight:700;letter-spacing:0.06em;font-family:'Inter',sans-serif;">⬡ RESEARCH PROJECT</span>
        <span style="display:inline-flex;align-items:center;gap:6px;background:#dcfce7;color:#15803d;border:1px solid #86efac;padding:5px 14px;border-radius:100px;font-size:0.75rem;font-weight:700;letter-spacing:0.06em;font-family:'Inter',sans-serif;">✦ EVAL ACC 97.27%</span>
        <span style="display:inline-flex;align-items:center;gap:6px;background:#fef3c7;color:#b45309;border:1px solid #fde68a;padding:5px 14px;border-radius:100px;font-size:0.75rem;font-weight:700;letter-spacing:0.06em;font-family:'Inter',sans-serif;">★ mAP@50 89.72%</span>
      </div>

      <!-- title -->
      <h1 style="
        text-align:center; margin:0 0 12px;
        font-family:'Inter',sans-serif; font-weight:900; font-size:clamp(2rem,5vw,3.2rem);
        line-height:1.1; letter-spacing:-0.02em;
        background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 50%, #0284c7 100%);
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: shimmer 4s linear infinite;
      ">
        🩸 Blood Cell AI Analysis
      </h1>
      <style>
        @keyframes shimmer { to { background-position: 200% center; } }
        @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        .hero-sub { animation: fadeUp 0.6s 0.2s ease both; }
        .hero-chips { animation: fadeUp 0.6s 0.4s ease both; }
        .hero-stats { animation: fadeUp 0.6s 0.6s ease both; }
      </style>

      <!-- subtitle -->
      <p class="hero-sub" style="text-align:center;color:#64748b;font-family:'Inter',sans-serif;font-size:1.05rem;font-weight:400;margin:0 0 28px;letter-spacing:0.01em;">
        Intelligent Detection &amp; Explainable Multimodal Classification
      </p>

      <!-- tech chips -->
      <div class="hero-chips" style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-bottom:36px;">
        <span style="display:inline-flex;align-items:center;gap:7px;background:#dbeafe;color:#1d4ed8;border:1px solid #93c5fd;padding:8px 18px;border-radius:12px;font-family:'Inter',sans-serif;font-size:0.83rem;font-weight:600;">
          🎯 YOLO26 NMS-Free
        </span>
        <span style="display:inline-flex;align-items:center;gap:7px;background:#e0f2fe;color:#0369a1;border:1px solid #7dd3fc;padding:8px 18px;border-radius:12px;font-family:'Inter',sans-serif;font-size:0.83rem;font-weight:600;">
          🧠 Qwen2.5-VL-3B LoRA
        </span>
        <span style="display:inline-flex;align-items:center;gap:7px;background:#ecfdf5;color:#065f46;border:1px solid #6ee7b7;padding:8px 18px;border-radius:12px;font-family:'Inter',sans-serif;font-size:0.83rem;font-weight:600;">
          ✨ 4× XAI Heatmaps
        </span>
        <span style="display:inline-flex;align-items:center;gap:7px;background:#f0fdf4;color:#166534;border:1px solid #86efac;padding:8px 18px;border-radius:12px;font-family:'Inter',sans-serif;font-size:0.83rem;font-weight:600;">
          🔬 EfficientNet-B2
        </span>
        <span style="display:inline-flex;align-items:center;gap:7px;background:#fffbeb;color:#92400e;border:1px solid #fde68a;padding:8px 18px;border-radius:12px;font-family:'Inter',sans-serif;font-size:0.83rem;font-weight:600;">
          🩺 12 Cell Classes
        </span>
      </div>

      <!-- stat strip -->
      <div class="hero-stats" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;">
        <div style="background:#fff;border:1px solid #bfdbfe;border-radius:14px;padding:16px 18px;text-align:center;position:relative;overflow:hidden;box-shadow:0 2px 8px rgba(29,111,216,0.08);">
          <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#1d6fd8,#0ea5e9);"></div>
          <div style="font-size:1.75rem;font-weight:900;font-family:'Inter',sans-serif;color:#1d6fd8;letter-spacing:-0.02em;">97.27%</div>
          <div style="font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;font-family:'Inter',sans-serif;">Qwen Eval Acc</div>
        </div>
        <div style="background:#fff;border:1px solid #bae6fd;border-radius:14px;padding:16px 18px;text-align:center;position:relative;overflow:hidden;box-shadow:0 2px 8px rgba(14,165,233,0.08);">
          <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#0ea5e9,#38bdf8);"></div>
          <div style="font-size:1.75rem;font-weight:900;font-family:'Inter',sans-serif;color:#0369a1;letter-spacing:-0.02em;">97.03%</div>
          <div style="font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;font-family:'Inter',sans-serif;">Classifier Acc</div>
        </div>
        <div style="background:#fff;border:1px solid #fde68a;border-radius:14px;padding:16px 18px;text-align:center;position:relative;overflow:hidden;box-shadow:0 2px 8px rgba(245,158,11,0.08);">
          <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#f59e0b,#fbbf24);"></div>
          <div style="font-size:1.75rem;font-weight:900;font-family:'Inter',sans-serif;color:#b45309;letter-spacing:-0.02em;">89.72%</div>
          <div style="font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;font-family:'Inter',sans-serif;">mAP@50 YOLO</div>
        </div>
        <div style="background:#fff;border:1px solid #86efac;border-radius:14px;padding:16px 18px;text-align:center;position:relative;overflow:hidden;box-shadow:0 2px 8px rgba(16,185,129,0.08);">
          <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#10b981,#34d399);"></div>
          <div style="font-size:1.75rem;font-weight:900;font-family:'Inter',sans-serif;color:#047857;letter-spacing:-0.02em;">17K+</div>
          <div style="font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;font-family:'Inter',sans-serif;">Training Samples</div>
        </div>
      </div>

      <!-- team -->
      <div style="text-align:center;margin-top:28px;">
        <span style="font-family:'Inter',sans-serif;font-size:0.78rem;color:#94a3b8;font-weight:500;">
          👨‍💻 Nguyễn Quốc Vinh &nbsp;·&nbsp; Hồ Nhật Hào &nbsp;·&nbsp; Lê Trần Quốc Huy
          &emsp;|&emsp; Khoa CNTT — ĐH Sư phạm Kỹ thuật Vĩnh Long
        </span>
      </div>
    </div>
    """)

    with gr.Tabs():
        # ─── TAB 1: CLASSIFY ──────────────────────────────────────
        with gr.Tab("🧬 Phân Loại Tế Bào"):
            gr.HTML("<p style='font-family:Inter,sans-serif;font-size:0.88rem;color:#64748b;margin:0 0 16px;'>Upload hoặc dán (Ctrl+V) ảnh tế bào đơn lẻ để phân loại vào <strong style=\"color:#a78bfa;\">12 lớp hình thái tế bào máu</strong>.</p>")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300):
                    img_cls = gr.Image(label="📷 Ảnh tế bào đầu vào", type="numpy", sources=["upload", "clipboard"], height=290)
                    with gr.Row():
                        btn_cls_crop = gr.Button("✂️ Cắt ảnh", size="sm", variant="secondary")
                        btn_cls = gr.Button("🔍 Phân Loại Ngay", variant="primary")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_cls:
                        crop_html_cls = gr.HTML()
                        crop_coords_cls = gr.Textbox(value="", elem_id="coords-cls", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_cls_apply = gr.Button("✅ Áp dụng crop", size="sm")
                with gr.Column(scale=2):
                    with gr.Row(equal_height=True):
                        conf_chart = gr.Image(label="📊 Confidence Chart", height=210)
                        cell_preview = gr.Image(label="🔎 Preview 224×224", height=210)
                    result_cls = gr.Markdown(label="Kết quả")
            btn_cls_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "cls"), ""),
                inputs=img_cls, outputs=[acc_cls, crop_html_cls, crop_coords_cls]
            )
            btn_cls_apply.click(apply_crop_coords, inputs=[img_cls, crop_coords_cls], outputs=img_cls)
            btn_cls.click(tab_classify, inputs=img_cls, outputs=[conf_chart, result_cls, cell_preview])

        # ─── TAB 2: XAI SINGLE ────────────────────────────────────
        with gr.Tab("🎨 XAI Heatmap"):
            gr.HTML("<p style='font-family:Inter,sans-serif;font-size:0.88rem;color:#64748b;margin:0 0 16px;'>Chọn phương pháp <strong style=\"color:#67e8f9;\">Explainable AI</strong> để trực quan hóa vùng tế bào mà mô hình tập trung khi phân loại.</p>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300):
                    img_xai = gr.Image(label="📷 Ảnh tế bào đầu vào", type="numpy", sources=["upload", "clipboard"], height=290)
                    xai_method = gr.Radio(
                        ['HiresCAM', 'XGrad-CAM', 'EigenCAM', 'Integrated Gradients'],
                        value='XGrad-CAM', label="🔬 Phương pháp XAI"
                    )
                    with gr.Row():
                        btn_xai_crop = gr.Button("✂️ Cắt ảnh", size="sm", variant="secondary")
                        btn_xai = gr.Button("🌡️ Chạy XAI", variant="primary")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_xai:
                        crop_html_xai = gr.HTML()
                        crop_coords_xai = gr.Textbox(value="", elem_id="coords-xai", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_xai_apply = gr.Button("✅ Áp dụng crop", size="sm")
                with gr.Column(scale=2):
                    heatmap_out = gr.Image(label="🗺️ Heatmap Visualization", height=320)
                    xai_info    = gr.Markdown(label="Giải thích")
            btn_xai_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "xai"), ""),
                inputs=img_xai, outputs=[acc_xai, crop_html_xai, crop_coords_xai]
            )
            btn_xai_apply.click(apply_crop_coords, inputs=[img_xai, crop_coords_xai], outputs=img_xai)
            btn_xai.click(tab_xai, inputs=[img_xai, xai_method], outputs=[heatmap_out, xai_info])

        # ─── TAB 3: YOLO DETECT ───────────────────────────────────
        with gr.Tab("🔎 YOLO26 Detection"):
            gr.HTML("<p style='font-family:Inter,sans-serif;font-size:0.88rem;color:#64748b;margin:0 0 16px;'>Upload ảnh phết máu toàn cảnh (blood smear) để <strong style=\"color:#fcd34d;\">YOLO26 tự động phát hiện</strong> và đếm RBC, WBC, Platelets.</p>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300):
                    img_det = gr.Image(label="🩸 Blood Smear — Ảnh phết máu", type="numpy", sources=["upload", "clipboard"], height=300)
                    conf_sld = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="🎚️ Confidence Threshold")
                    with gr.Row():
                        btn_det_crop = gr.Button("✂️ Cắt ảnh", size="sm", variant="secondary")
                        btn_det = gr.Button("🎯 Detect & Đếm Tế Bào", variant="primary")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_det:
                        crop_html_det = gr.HTML()
                        crop_coords_det = gr.Textbox(value="", elem_id="coords-det", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_det_apply = gr.Button("✅ Áp dụng crop", size="sm")
                with gr.Column(scale=2):
                    det_out     = gr.Image(label="📍 Detection Result — Kết quả khoanh vùng", height=360)
                    det_summary = gr.Markdown(label="Thống kê")
            btn_det_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "det"), ""),
                inputs=img_det, outputs=[acc_det, crop_html_det, crop_coords_det]
            )
            btn_det_apply.click(apply_crop_coords, inputs=[img_det, crop_coords_det], outputs=img_det)
            btn_det.click(tab_detect, inputs=[img_det, conf_sld], outputs=[det_out, det_summary])

        # ─── TAB 4: XAI ALL METHODS ───────────────────────────────
        with gr.Tab("🧪 So Sánh 4 XAI"):
            gr.HTML("<p style='font-family:Inter,sans-serif;font-size:0.88rem;color:#64748b;margin:0 0 16px;'>So sánh <strong style=\"color:#a78bfa;\">HiresCAM · XGrad-CAM · EigenCAM · Integrated Gradients</strong> trên cùng một ảnh tế bào.</p>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300):
                    img_all = gr.Image(label="📷 Ảnh tế bào đầu vào", type="numpy", sources=["upload", "clipboard"], height=280)
                    with gr.Row():
                        btn_all_crop = gr.Button("✂️ Cắt ảnh", size="sm", variant="secondary")
                        btn_all = gr.Button("🚀 So Sánh Tất Cả 4", variant="primary")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_all:
                        crop_html_all = gr.HTML()
                        crop_coords_all = gr.Textbox(value="", elem_id="coords-all", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_all_apply = gr.Button("✅ Áp dụng crop", size="sm")
                with gr.Column(scale=3):
                    grid_out = gr.Image(label="📊 XAI Comparison Grid — Lưới so sánh 4 phương pháp")
            btn_all_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "all"), ""),
                inputs=img_all, outputs=[acc_all, crop_html_all, crop_coords_all]
            )
            btn_all_apply.click(apply_crop_coords, inputs=[img_all, crop_coords_all], outputs=img_all)
            btn_all.click(tab_xai_all, inputs=img_all, outputs=grid_out)

        # ─── TAB 5: QWEN2.5-VL FINE-TUNED ─────────────────────────
        with gr.Tab("🧠 Qwen2.5-VL Analysis"):
            gr.HTML("<p style='font-family:Inter,sans-serif;font-size:0.88rem;color:#64748b;margin:0 0 16px;'>Phân tích hình thái học &amp; sinh báo cáo y khoa tự động bằng <strong style=\"color:#6ee7b7;\">Qwen2.5-VL-3B LoRA</strong> (checkpoint-1500).</p>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300):
                    img_vlm = gr.Image(label="📷 Ảnh tế bào đầu vào", type="numpy", sources=["upload", "clipboard"], height=280)
                    with gr.Row():
                        btn_vlm_crop = gr.Button("✂️ Cắt ảnh", size="sm", variant="secondary")
                    with gr.Accordion("✂️ Công cụ cắt ảnh", open=False, visible=False) as acc_vlm:
                        crop_html_vlm = gr.HTML()
                        crop_coords_vlm = gr.Textbox(value="", elem_id="coords-vlm", elem_classes=["crop-coords-hidden"], label="coords")
                        btn_vlm_apply = gr.Button("✅ Áp dụng crop", size="sm")
                    prompt_vlm = gr.Textbox(
                        label="💬 Yêu cầu / Clinical Prompt",
                        value="Identify the blood cell type in this image and explain its key morphological features.",
                        lines=3,
                        placeholder="Nhập câu hỏi hoặc yêu cầu phân tích cho Qwen2.5-VL..."
                    )
                    btn_vlm = gr.Button("🤖 Phân Tích với Qwen2.5-VL", variant="primary")
                with gr.Column(scale=2):
                    vlm_output = gr.Markdown("*Kết quả phân tích từ Qwen2.5-VL LoRA sẽ hiển thị tại đây sau khi nhấn nút phân tích...*")
            btn_vlm_crop.click(
                lambda img: (gr.update(visible=True), make_crop_html(img, "vlm"), ""),
                inputs=img_vlm, outputs=[acc_vlm, crop_html_vlm, crop_coords_vlm]
            )
            btn_vlm_apply.click(apply_crop_coords, inputs=[img_vlm, crop_coords_vlm], outputs=img_vlm)
            btn_vlm.click(tab_vlm_analyze, inputs=[img_vlm, prompt_vlm], outputs=vlm_output)

        # ─── TAB 6: INFO ──────────────────────────────────────────
        with gr.Tab("ℹ️ Về Dự Án"):
            gr.HTML("""
            <div style="font-family:'Inter',sans-serif;max-width:900px;margin:0 auto;padding:8px 0 24px;">

              <!-- Model Results -->
              <h2 style="color:#f0f4ff;font-size:1.05rem;font-weight:700;margin:0 0 14px;display:flex;align-items:center;gap:8px;">
                <span style="background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">📊 Kết Quả Huấn Luyện Mô Hình</span>
              </h2>
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:28px;">
                <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);border-radius:14px;padding:18px;position:relative;overflow:hidden;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6366f1,#a78bfa);"></div>
                  <div style="font-size:0.72rem;color:#8892b0;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">YOLO26 — mAP@50</div>
                  <div style="font-size:2rem;font-weight:900;color:#a78bfa;margin:6px 0 2px;letter-spacing:-0.02em;">89.72%</div>
                  <div style="font-size:0.78rem;color:#64748b;">BCCD Dataset · 205 ảnh phết máu</div>
                </div>
                <div style="background:rgba(34,211,238,0.08);border:1px solid rgba(34,211,238,0.2);border-radius:14px;padding:18px;position:relative;overflow:hidden;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#06b6d4,#22d3ee);"></div>
                  <div style="font-size:0.72rem;color:#8892b0;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">YOLO26 — mAP@50-95</div>
                  <div style="font-size:2rem;font-weight:900;color:#67e8f9;margin:6px 0 2px;letter-spacing:-0.02em;">62.55%</div>
                  <div style="font-size:0.78rem;color:#64748b;">NMS-Free Architecture</div>
                </div>
                <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:14px;padding:18px;position:relative;overflow:hidden;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#10b981,#34d399);"></div>
                  <div style="font-size:0.72rem;color:#8892b0;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">EfficientNet-B2 — Accuracy</div>
                  <div style="font-size:2rem;font-weight:900;color:#6ee7b7;margin:6px 0 2px;letter-spacing:-0.02em;">97.03%</div>
                  <div style="font-size:0.78rem;color:#64748b;">Dataset-Crop · 17K+ ảnh · 12 classes</div>
                </div>
                <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:14px;padding:18px;position:relative;overflow:hidden;">
                  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#f59e0b,#fbbf24);"></div>
                  <div style="font-size:0.72rem;color:#8892b0;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">Qwen2.5-VL LoRA — Eval Acc</div>
                  <div style="font-size:2rem;font-weight:900;color:#fcd34d;margin:6px 0 2px;letter-spacing:-0.02em;">97.27%</div>
                  <div style="font-size:0.78rem;color:#64748b;">Checkpoint-1500 · r=16 α=32</div>
                </div>
              </div>

              <!-- 12 Classes -->
              <h2 style="color:#f0f4ff;font-size:1.05rem;font-weight:700;margin:0 0 14px;">
                <span style="background:linear-gradient(135deg,#22d3ee,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🔬 12 Loại Tế Bào Phân Loại</span>
              </h2>
              <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-bottom:28px;">
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">BA</code>&nbsp; Basophil</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#6ee7b7;">0.99</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">BNE</code>&nbsp; Band Neutrophil</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#6ee7b7;">0.97</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">EO</code>&nbsp; Eosinophil</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#6ee7b7;">0.99</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">ERB</code>&nbsp; Erythroblast</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#6ee7b7;">0.99</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">LY</code>&nbsp; Lymphocyte</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#6ee7b7;">0.97</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">MMY</code>&nbsp; Metamyelocyte</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#fbbf24;">0.95</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">MO</code>&nbsp; Monocyte</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#6ee7b7;">0.97</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">MY</code>&nbsp; Myelocyte</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#fbbf24;">0.94</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">MYO</code>&nbsp; Myeloblast</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#fbbf24;">0.95</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">PLT</code>&nbsp; Platelet</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#6ee7b7;">1.00</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">PMY</code>&nbsp; Promyelocyte</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#f87171;">0.90</span>
                </div>
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-size:0.82rem;color:#8892b0;"><code style="background:rgba(99,102,241,0.2);color:#a78bfa;padding:2px 7px;border-radius:5px;font-size:0.78rem;">SNE</code>&nbsp; Segm. Neutrophil</span>
                  <span style="font-size:0.8rem;font-weight:700;color:#6ee7b7;">0.97</span>
                </div>
              </div>

              <!-- Architecture -->
              <h2 style="color:#f0f4ff;font-size:1.05rem;font-weight:700;margin:0 0 12px;">
                <span style="background:linear-gradient(135deg,#f59e0b,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🏗️ Kiến Trúc Pipeline</span>
              </h2>
              <div style="background:rgba(0,0,0,0.4);border:1px solid rgba(99,102,241,0.2);border-radius:14px;padding:20px 24px;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#67e8f9;line-height:1.8;margin-bottom:24px;overflow-x:auto;">
                <span style="color:#8892b0;">Input Image (Blood Smear)</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                <span style="color:#a78bfa;">YOLO26n</span> <span style="color:#8892b0;">(NMS-Free)</span> <span style="color:#64748b;">→ Detect &amp; Crop Cells</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
                ├── <span style="color:#22d3ee;">QwenCellClassifier</span> <span style="color:#8892b0;">(EfficientNet-B2)</span> + <span style="color:#f59e0b;">XAI ×4</span><br>
                └── <span style="color:#6ee7b7;">Qwen2.5-VL-3B-Instruct</span> + <span style="color:#a78bfa;">LoRA</span> <span style="color:#8892b0;">(checkpoint-1500)</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ <span style="color:#fcd34d;">Clinical Natural Language Reasoning</span>
              </div>

              <!-- Team -->
              <h2 style="color:#f0f4ff;font-size:1.05rem;font-weight:700;margin:0 0 12px;">
                <span style="background:linear-gradient(135deg,#ec4899,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">👥 Nhóm Nghiên Cứu</span>
              </h2>
              <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:14px 20px;">
                  <div style="font-size:0.75rem;color:#6366f1;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">23004008</div>
                  <div style="font-size:0.92rem;color:#e0e7ff;font-weight:600;margin-top:4px;">Nguyễn Quốc Vinh</div>
                </div>
                <div style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.2);border-radius:12px;padding:14px 20px;">
                  <div style="font-size:0.75rem;color:#8b5cf6;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">23004023</div>
                  <div style="font-size:0.92rem;color:#e0e7ff;font-weight:600;margin-top:4px;">Hồ Nhật Hào</div>
                </div>
                <div style="background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.2);border-radius:12px;padding:14px 20px;">
                  <div style="font-size:0.75rem;color:#a78bfa;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">23004050</div>
                  <div style="font-size:0.92rem;color:#e0e7ff;font-weight:600;margin-top:4px;">Lê Trần Quốc Huy</div>
                </div>
              </div>
              <div style="margin-top:14px;font-size:0.78rem;color:#475569;">
                Khoa Công nghệ Thông tin — Trường Đại học Sư phạm Kỹ thuật Vĩnh Long
              </div>

            </div>
            """)

    gr.HTML("""
    <div class="footer-bar">
      <span style="font-family:'Inter',sans-serif;font-size:0.78rem;color:#334155;">
        🩸 Blood Cell AI Analysis Pro &nbsp;·&nbsp; YOLO26 + Qwen2.5-VL + XAI
        &nbsp;·&nbsp; Khoa CNTT — ĐH Sư phạm Kỹ thuật Vĩnh Long &nbsp;·&nbsp; 2025
      </span>
    </div>
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
