# =============================================================================
# NOTEBOOK 01: YOLO26 — PHÁT HIỆN TẾ BÀO MÁU (BCCD DATASET — SUPERVISELY JSON)
# Đề tài: Triển khai Qwen cho phát hiện & phân loại tế bào kết hợp XAI
# Nhóm: Nguyễn Quốc Vinh · Hồ Nhật Hào · Lê Trần Quốc Huy
# Platform: Kaggle Notebook — GPU T4 x2
# Dataset: orvile/bccd-blood-cell-count-and-detection-dataset (Supervisely JSON)
# =============================================================================


# %%
# ============================================================
# CELL 1: CÀI ĐẶT & KIỂM TRA GPU
# ============================================================
import subprocess

subprocess.run(['pip', 'install', '-q', 'ultralytics>=8.4.0'], check=True)
subprocess.run(['pip', 'install', '-q', 'opencv-python-headless',
                'matplotlib', 'seaborn', 'pandas', 'Pillow', 'tqdm'], check=True)

import torch
print(f"✅ CUDA: {torch.cuda.is_available()} | GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


# %%
# ============================================================
# CELL 2: IMPORT
# ============================================================
import os, json, shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch
from ultralytics import YOLO

print("✅ Import xong!")


# %%
# ============================================================
# CELL 3: ĐƯỜNG DẪN DATASET
# ============================================================

# BCCD Detection dataset (Supervisely JSON format)
BCCD_ROOT   = Path("/kaggle/input/datasets/orvile/bccd-blood-cell-count-and-detection-dataset")
WORKING_DIR = Path("/kaggle/working")

# Dataset đã chia sẵn train/val/test
SPLITS = {
    'train': BCCD_ROOT / 'train',
    'val':   BCCD_ROOT / 'val',
    'test':  BCCD_ROOT / 'test',
}

print("✅ Kiểm tra dataset:")
for split, path in SPLITS.items():
    img_count = len(list((path / 'img').glob('*'))) if (path / 'img').exists() else 0
    ann_count = len(list((path / 'ann').glob('*.json'))) if (path / 'ann').exists() else 0
    print(f"  {split:5s}: {img_count:3d} ảnh | {ann_count:3d} annotations")

# Classes
CLASS_NAMES = ['RBC', 'WBC', 'Platelets']
CLASS_MAP   = {'RBC': 0, 'WBC': 1, 'Platelets': 2}
print(f"\n📌 Classes: {CLASS_NAMES}")


# %%
# ============================================================
# CELL 4: EDA — PHÂN TÍCH DỮ LIỆU
# ============================================================

def read_supervisely_json(json_path):
    """Đọc annotation từ Supervisely JSON, trả về list dict."""
    with open(json_path) as f:
        data = json.load(f)
    img_h = data['size']['height']
    img_w = data['size']['width']
    records = []
    for obj in data.get('objects', []):
        cls = obj.get('classTitle', '').strip()
        if cls not in CLASS_MAP:
            continue
        ext = obj['points']['exterior']
        x1, y1 = ext[0]
        x2, y2 = ext[1]
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        records.append({
            'class': cls,
            'width': w, 'height': h,
            'area': w * h,
            'img_w': img_w, 'img_h': img_h,
        })
    return records

# Thu thập thống kê toàn bộ splits
all_records = []
for split, path in SPLITS.items():
    ann_dir = path / 'ann'
    if not ann_dir.exists():
        continue
    for json_path in tqdm(sorted(ann_dir.glob('*.json')), desc=f"Đọc {split}"):
        recs = read_supervisely_json(json_path)
        for r in recs:
            r['split'] = split
        all_records.extend(recs)

df = pd.DataFrame(all_records)
print(f"\n📊 Tổng bounding boxes: {len(df)}")
print(f"\n🔢 Phân bố class:")
print(df['class'].value_counts())
print(f"\n📂 Phân bố theo split:")
print(df.groupby(['split','class']).size().unstack(fill_value=0))

# Visualize EDA
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('BCCD Dataset — EDA (Supervisely JSON)', fontsize=16, fontweight='bold')
colors = ['#e74c3c', '#3498db', '#f39c12']

counts = df['class'].value_counts().reindex(CLASS_NAMES, fill_value=0)
axes[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Phân bố Class', fontweight='bold')

for cls, c in zip(CLASS_NAMES, colors):
    sub = df[df['class'] == cls]['area']
    if not sub.empty:
        axes[1].hist(sub, bins=30, alpha=0.6, color=c, label=cls)
axes[1].set_xlabel('Diện tích BB (px²)')
axes[1].set_ylabel('Số lượng')
axes[1].set_title('Phân bố Kích Thước Tế Bào', fontweight='bold')
axes[1].legend()

for cls, c in zip(CLASS_NAMES, colors):
    sub = df[df['class'] == cls]
    axes[2].scatter(sub['width'], sub['height'], alpha=0.4, color=c, label=cls, s=10)
axes[2].set_xlabel('Width (px)')
axes[2].set_ylabel('Height (px)')
axes[2].set_title('Width vs Height theo Class', fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig(WORKING_DIR / 'eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA hoàn tất!")


# %%
# ============================================================
# CELL 5: VISUALIZE MẪU ẢNH VỚI BOUNDING BOX
# ============================================================

def draw_sample_json(json_path, img_dir, ax):
    """Vẽ bounding box từ Supervisely JSON lên ảnh."""
    with open(json_path) as f:
        data = json.load(f)
    # Tên ảnh = tên json bỏ đuôi .json
    img_name = json_path.name.replace('.json', '')
    img_path = img_dir / img_name
    if not img_path.exists():
        # thử tìm theo stem
        cands = list(img_dir.glob(f"{json_path.stem.replace('.jpeg','').replace('.jpg','')}*"))
        if not cands:
            ax.axis('off')
            return
        img_path = cands[0]

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    color_map = {'RBC': 'red', 'WBC': 'cyan', 'Platelets': 'yellow'}
    for obj in data.get('objects', []):
        cls = obj.get('classTitle', '').strip()
        if cls not in CLASS_MAP:
            continue
        ext = obj['points']['exterior']
        x1, y1 = ext[0]
        x2, y2 = ext[1]
        rect = patches.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1),
                                   linewidth=1.5, edgecolor=color_map.get(cls, 'white'),
                                   facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-3, cls, color=color_map.get(cls, 'white'),
                fontsize=6, fontweight='bold')
    ax.set_title(img_path.name, fontsize=7)
    ax.axis('off')

# Lấy 6 ảnh mẫu từ train
train_jsons = sorted((SPLITS['train'] / 'ann').glob('*.json'))[:6]
train_img   = SPLITS['train'] / 'img'

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Mẫu Ảnh BCCD với Bounding Box', fontsize=14, fontweight='bold')
for json_path, ax in zip(train_jsons, axes.flatten()):
    draw_sample_json(json_path, train_img, ax)
plt.tight_layout()
plt.savefig(WORKING_DIR / 'sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Visualize mẫu ảnh xong!")


# %%
# ============================================================
# CELL 6: CONVERT SUPERVISELY JSON → YOLO FORMAT
# ============================================================

YOLO_DIR = WORKING_DIR / "bccd_yolo"

def convert_split_to_yolo(split_name, split_path, yolo_dir):
    """Convert 1 split (train/val/test) từ Supervisely JSON sang YOLO."""
    img_dir = split_path / 'img'
    ann_dir = split_path / 'ann'
    out_img = yolo_dir / split_name / 'images'
    out_lbl = yolo_dir / split_name / 'labels'
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    json_files = sorted(ann_dir.glob('*.json'))
    ok = 0
    for json_path in tqdm(json_files, desc=f"Convert {split_name}"):
        with open(json_path) as f:
            data = json.load(f)

        img_h = data['size']['height']
        img_w = data['size']['width']

        # Copy ảnh
        img_name = json_path.name.replace('.json', '')
        img_src  = img_dir / img_name
        if not img_src.exists():
            continue
        shutil.copy(img_src, out_img / img_src.name)

        # Tạo label YOLO
        lines = []
        for obj in data.get('objects', []):
            cls = obj.get('classTitle', '').strip()
            if cls not in CLASS_MAP:
                continue
            ext = obj['points']['exterior']
            x1, y1 = ext[0]
            x2, y2 = ext[1]
            # Normalize
            cx = max(0, min(1, (x1 + x2) / 2 / img_w))
            cy = max(0, min(1, (y1 + y2) / 2 / img_h))
            bw = max(0, min(1, abs(x2 - x1) / img_w))
            bh = max(0, min(1, abs(y2 - y1) / img_h))
            lines.append(f"{CLASS_MAP[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            lbl_name = img_src.stem + '.txt'
            (out_lbl / lbl_name).write_text("\n".join(lines))
            ok += 1

    print(f"  ✅ {split_name}: {ok}/{len(json_files)} ảnh converted")
    return ok

# Convert tất cả splits
for split_name, split_path in SPLITS.items():
    convert_split_to_yolo(split_name, split_path, YOLO_DIR)

print(f"\n📁 YOLO dataset: {YOLO_DIR}")


# %%
# ============================================================
# CELL 7: TẠO bccd.yaml
# ============================================================

yaml_content = f"""# BCCD Dataset — YOLO26
# Đề tài: Phát hiện & phân loại tế bào kết hợp XAI
# Nhóm: Nguyễn Quốc Vinh · Hồ Nhật Hào · Lê Trần Quốc Huy

path: {YOLO_DIR}
train: train/images
val:   val/images
test:  test/images

nc: 3
names:
  0: RBC
  1: WBC
  2: Platelets
"""

yaml_path = WORKING_DIR / "bccd.yaml"
yaml_path.write_text(yaml_content)
print("✅ bccd.yaml đã tạo:")
print(yaml_content)


# %%
# ============================================================
# CELL 8: TRAIN YOLO26 — 2x T4 GPU (DDP)
# ============================================================

model = YOLO("yolo26n.pt")

results = model.train(
    data         = str(yaml_path),
    epochs       = 100,
    imgsz        = 640,
    batch        = 32,       # 16/GPU × 2 GPU
    device       = "0,1",   # 2x T4 GPU — DDP
    workers      = 4,
    patience     = 20,       # EarlyStopping
    optimizer    = "MuSGD",  # Optimizer mới của YOLO26
    lr0          = 0.01,
    lrf          = 0.01,
    momentum     = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3,
    cos_lr       = True,
    mosaic       = 1.0,
    flipud       = 0.5,
    fliplr       = 0.5,
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    degrees      = 10.0,
    translate    = 0.1,
    scale        = 0.5,
    project      = str(WORKING_DIR / "runs"),
    name         = "yolo26_bccd",
    exist_ok     = True,
    save         = True,
    save_period  = 10,
    plots        = True,
    verbose      = True,
)
print("✅ Training hoàn tất!")


# %%
# ============================================================
# CELL 9: ĐÁNH GIÁ — mAP, Precision, Recall
# ============================================================

best_weights = WORKING_DIR / "runs" / "yolo26_bccd" / "weights" / "best.pt"
model_eval   = YOLO(str(best_weights))

metrics = model_eval.val(
    data    = str(yaml_path),
    imgsz   = 640,
    device  = "0",
    plots   = True,
    verbose = True,
)

print("\n" + "="*60)
print("📊 KẾT QUẢ YOLO26 — BCCD DATASET")
print("="*60)
print(f"  mAP@50       : {metrics.box.map50:.4f}")
print(f"  mAP@50-95    : {metrics.box.map:.4f}")
print(f"  Precision    : {metrics.box.mp:.4f}")
print(f"  Recall       : {metrics.box.mr:.4f}")
print("-"*60)
for i, cls_name in enumerate(CLASS_NAMES):
    if i < len(metrics.box.ap50):
        print(f"  {cls_name:12s} AP@50: {metrics.box.ap50[i]:.4f}")
print("="*60)


# %%
# ============================================================
# CELL 10: VISUALIZE DETECTION RESULTS
# ============================================================

val_images = sorted((YOLO_DIR / 'val' / 'images').glob("*"))[:8]
fig, axes  = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('YOLO26 Detection Results — BCCD', fontsize=14, fontweight='bold')

for i, (img_path, ax) in enumerate(zip(val_images, axes.flatten())):
    res     = model_eval.predict(str(img_path), conf=0.25, verbose=False)[0]
    img_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    n_det   = len(res.boxes)
    ax.set_title(f"Sample {i+1} — {n_det} cells", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig(WORKING_DIR / 'detection_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Detection results saved!")


# %%
# ============================================================
# CELL 11: CROP CELL PATCHES → INPUT CHO QWEN3.5
# ============================================================

CROP_DIR = WORKING_DIR / "cell_patches_yolo"
for cls in CLASS_NAMES:
    (CROP_DIR / cls).mkdir(parents=True, exist_ok=True)

CONF_THR = 0.4
PADDING  = 5
crop_cnt = {c: 0 for c in CLASS_NAMES}

all_val_imgs = sorted((YOLO_DIR / 'val' / 'images').glob("*"))
for img_path in tqdm(all_val_imgs, desc="Cropping cells"):
    results = model_eval.predict(str(img_path), conf=CONF_THR, verbose=False)
    if not results:
        continue
    res  = results[0]
    img  = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for j, box in enumerate(res.boxes):
        cls_id   = int(box.cls[0])
        if cls_id >= len(CLASS_NAMES):
            continue
        cls_name = CLASS_NAMES[cls_id]
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1 = max(0, x1-PADDING), max(0, y1-PADDING)
        x2, y2 = min(w, x2+PADDING), min(h, y2+PADDING)
        crop    = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        out = cv2.resize(crop, (224, 224))
        fname = f"{img_path.stem}_b{j}_{conf:.2f}.jpg"
        cv2.imwrite(str(CROP_DIR / cls_name / fname), out)
        crop_cnt[cls_name] += 1

total = sum(crop_cnt.values())
print("✅ Cell patches hoàn tất!")
for cls, cnt in crop_cnt.items():
    print(f"  {cls:12s}: {cnt:5d} ảnh")
print(f"  {'TOTAL':12s}: {total:5d} ảnh")


# %%
# ============================================================
# CELL 12: TỔNG KẾT
# ============================================================

print("=" * 65)
print("🎯 HOÀN THÀNH — YOLO26 BCCD DETECTION")
print("=" * 65)
print(f"  Model    : YOLO26n (NMS-Free · DFL-Free · MuSGD)")
print(f"  Dataset  : BCCD — Supervisely JSON")
print(f"  mAP@50   : {metrics.box.map50:.4f}")
print(f"  mAP@50-95: {metrics.box.map:.4f}")
print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall   : {metrics.box.mr:.4f}")
print(f"  Patches  : {total} ảnh (224×224) → sẵn sàng cho Qwen3.5")
print(f"  Best wts : {best_weights}")
print("=" * 65)
print("⏭️  BƯỚC TIẾP: Notebook 02 — Qwen3.5 Fine-tune + XAI")
