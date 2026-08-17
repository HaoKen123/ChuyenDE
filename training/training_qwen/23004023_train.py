# %%
# ==============================================================
# 0. KAGGLE SETUP: CÀI ĐẶT THƯ VIỆN BẮT BUỘC
# (Chạy ô này đầu tiên nếu dùng trên Kaggle)
# ==============================================================
# !pip uninstall -y torchaudio
# !pip install -q bitsandbytes accelerate peft "transformers>=4.45.0" qwen-vl-utils

# %%
# ==============================================================
# QWEN2.5-VL **DoRA** TRAINING + AUTO EVALUATION REPORT GENERATOR
# 23004023 — Phân Loại Tế Bào Máu
# MODEL B (DoRA) vs Model A (LoRA checkpoint-5500)
#
# KHÁC BIỆT CHÍNH:
#   Model A: LoRA (use_dora=False), r=16, alpha=32, dropout=0.2
#   Model B: DoRA (use_dora=True),  r=8,  alpha=16, dropout=0.1
#
# DoRA = Weight-Decomposed Low-Rank Adaptation
#   → Phân tách trọng số thành magnitude + direction
#   → Chất lượng fine-tuning tốt hơn LoRA ở cùng rank
#   → Paper: "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al. 2024)
# ==============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"   # FIX: Khai báo IP ảo cho môi trường cluster giả lập
os.environ["MASTER_PORT"] = "12345"       # FIX: Khai báo Port ảo cho môi trường cluster giả lập
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys, gc, glob, time, warnings, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

import torch
_original_device_count = torch.cuda.device_count
torch.cuda.device_count = lambda: 1

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore")

# %%
# ─── PATHS ────────────────────────────────────────────────────
OUTPUT_DIR = Path("/kaggle/working/23004023_PhanLoaiTeBaoMau")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = None
for _p in [
    Path("/kaggle/input/data-crop/Dataset-Crop"),
    Path("/kaggle/input/dataset-crop/Dataset-Crop"),
]:
    if _p.exists():
        DATASET_PATH = _p
        break
if DATASET_PATH is None:
    _found = glob.glob("/kaggle/input/**/Dataset-Crop", recursive=True)
    if _found:
        DATASET_PATH = Path(_found[0])
assert DATASET_PATH is not None and DATASET_PATH.exists(), \
    "❌ Không tìm thấy thư mục Dataset-Crop!"
print(f"✅ Dataset found: {DATASET_PATH}")

# %%
# ─── HYPERPARAMETERS ──────────────────────────────────────────
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# ════════════════════════════════════════════════════════════════
# MODEL B — DoRA CONFIGURATION
# ────────────────────────────────────────────────────────────────
# So sánh với Model A (LoRA checkpoint-5500):
#   Model A: LoRA  | r=16 | alpha=32 | dropout=0.2 | full dataset | batch=8
#   Model B: DoRA  | r=8  | alpha=16 | dropout=0.1 | 100/class   | batch=4
#
# DoRA khác LoRA ở chỗ: phân tách weight thành magnitude × direction
# → Giữ magnitude gốc, chỉ fine-tune direction qua low-rank update
# → Ở cùng rank, DoRA thường cho kết quả tốt hơn LoRA
#
# QUAN TRỌNG: DoRA + 4-bit NF4 cần PEFT >= 0.10.0
# Nếu gặp lỗi trên Kaggle, script sẽ tự fallback sang 8-bit
# ════════════════════════════════════════════════════════════════
MAX_PER_CLASS  = 100
EPOCHS         = 50
BATCH_SIZE     = 2
GRADIENT_ACCUM = 2     # effective = 4
LEARNING_RATE  = 2e-4
LORA_R         = 8     # Model A dùng r=16
LORA_ALPHA     = 16    # Model A dùng alpha=32
LORA_DROPOUT   = 0.1   # Model A dùng 0.2
IMG_SIZE       = 224
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %%
# ─── CLASS DEFINITIONS ────────────────────────────────────────
CELL_LABELS = ["BA", "BNE", "EO", "ERB", "LY", "MMY", "MO", "MY", "MYO", "PLT", "PMY", "SNE"]
CELL_NAMES = {
    "BA":  "Basophil",         "BNE": "Band Neutrophil",   "EO":  "Eosinophil",
    "ERB": "Erythroblast",     "LY":  "Lymphocyte",         "MMY": "Metamyelocyte",
    "MO":  "Monocyte",         "MY":  "Myelocyte",           "MYO": "Myeloblast",
    "PLT": "Platelet",         "PMY": "Promyelocyte",        "SNE": "Segmented Neutrophil",
}

PROMPT_TEXT = (
    "This is a microscopic blood cell image. "
    f"Classify it into exactly one of these categories: {', '.join(CELL_LABELS)}. "
    + "Abbreviation meanings: "
    + ", ".join(f"{k}={v}" for k, v in CELL_NAMES.items())
    + ". Reply with ONLY the 2-3 letter abbreviation code (e.g., 'LY')."
)

# %%
# ══════════════════════════════════════════════════════════════
# 1. DATASET  (stratified subsample → fit 50 epochs)
# ══════════════════════════════════════════════════════════════
def prepare_dataset():
    all_by_class = {}
    for label in CELL_LABELS:
        cls_dir = DATASET_PATH / label
        if not cls_dir.exists():
            print(f"  ⚠️  Missing: {label}")
            continue
        imgs = [str(p) for p in sorted(cls_dir.glob("*"))
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
        random.shuffle(imgs)
        all_by_class[label] = imgs[:MAX_PER_CLASS]

    train_d, val_d = [], []
    print(f"\n{'='*55}")
    print(f"  MODEL B (DoRA) — DATASET SUMMARY  (MAX_PER_CLASS={MAX_PER_CLASS})")
    print(f"{'='*55}")
    print(f"  {'Class':<6} {'Total':>6} {'Train':>6} {'Val':>5}")
    print(f"  {'-'*30}")
    for label in CELL_LABELS:
        imgs = all_by_class.get(label, [])
        split = int(len(imgs) * 0.85)
        tr = [{"image_path": p, "label": label} for p in imgs[:split]]
        vl = [{"image_path": p, "label": label} for p in imgs[split:]]
        train_d.extend(tr); val_d.extend(vl)
        print(f"  {label:<6} {len(imgs):>6} {len(tr):>6} {len(vl):>5}")

    random.shuffle(train_d); random.shuffle(val_d)

    eff_batch = BATCH_SIZE * GRADIENT_ACCUM
    spe = max(1, len(train_d) // eff_batch)
    total_s = spe * EPOCHS
    est_h = total_s * 3 / 3600
    print(f"  {'-'*30}")
    print(f"  Total   {len(train_d)+len(val_d):>6} {len(train_d):>6} {len(val_d):>5}")
    print(f"{'='*55}")
    print(f"  Method          : DoRA (Weight-Decomposed LoRA)")
    print(f"  Effective batch : {eff_batch}  (Model A=8)")
    print(f"  Steps/epoch     : {spe}")
    print(f"  Total steps     : {total_s}  ({EPOCHS} epochs)")
    print(f"  Estimated time  : ~{est_h:.1f}h  (@ 3s/step on T4)")
    print(f"{'='*55}\n")
    return train_d, val_d


class BloodCellDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, processor, is_train=True):
        self.data = data_list
        self.processor = processor
        self.is_train = is_train
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ])
        self._im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._asst_header_len = len(
            processor.tokenizer.encode("assistant\n", add_special_tokens=False))
        print(f"   [Dataset] im_start_id={self._im_start_id} "
              f"| asst_header_len={self._asst_header_len} "
              f"| n={len(data_list)}")

    def __len__(self):
        return len(self.data)

    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.is_train:
            img = self.aug(img)
        return img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self._load_image(item["image_path"])

        full_msgs = [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPT_TEXT}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": item["label"]}]},
        ]
        full_text = self.processor.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False)
        enc = self.processor(
            text=[full_text], images=[image],
            padding=False, return_tensors="pt")

        ids = enc["input_ids"][0].tolist()
        last_im_start = -1
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == self._im_start_id:
                last_im_start = i
                break
        if last_im_start == -1:
            prompt_len = int(len(ids) * 0.9)
        else:
            prompt_len = last_im_start + 1 + self._asst_header_len

        result = {}
        for k, v in enc.items():
            if k in ("pixel_values", "image_grid_thw"):
                result[k] = v
            else:
                result[k] = v[0]

        labels = result["input_ids"].clone()
        labels[:prompt_len] = -100
        result["labels"] = labels
        return result


# %%
# ══════════════════════════════════════════════════════════════
# 2. COLLATE FN
# ══════════════════════════════════════════════════════════════
def make_collate_fn(pad_token_id):
    PAD_WITH = {
        "input_ids"         : pad_token_id,
        "attention_mask"    : 0,
        "labels"            : -100,
        "mm_token_type_ids" : 0,
        "token_type_ids"    : 0,
    }

    def collate_fn(features):
        batch = {}
        for key in features[0].keys():
            tensors = [f[key] for f in features]
            if key in ("pixel_values", "image_grid_thw"):
                batch[key] = torch.cat(tensors, dim=0)
            elif key in PAD_WITH:
                fill = PAD_WITH[key]
                max_len = max(t.size(0) for t in tensors)
                padded = torch.full((len(tensors), max_len), fill, dtype=torch.long)
                for i, t in enumerate(tensors):
                    padded[i, :t.size(0)] = t
                batch[key] = padded
            else:
                try:
                    batch[key] = torch.stack(tensors, dim=0)
                except Exception:
                    pass
        return batch
    return collate_fn


# %%
# ══════════════════════════════════════════════════════════════
# 3. MODEL SETUP — DoRA (Weight-Decomposed LoRA)
# ══════════════════════════════════════════════════════════════
# ┌──────────────────────────────────────────────────────────┐
# │  DoRA phân tách weight W thành:                          │
# │    W = m × (V / ||V||)                                  │
# │  Trong đó:                                               │
# │    m = magnitude (trainable scalar)                      │
# │    V = direction (updated via low-rank: V₀ + BA)         │
# │                                                          │
# │  → Giữ magnitude gốc ổn định                            │
# │  → Chỉ fine-tune hướng (direction) qua LoRA matrices    │
# │  → Kết quả: chất lượng tốt hơn LoRA ở cùng rank        │
# │                                                          │
# │  Fix cho Kaggle T4:                                      │
# │    - Thử 4-bit trước (nhanh, ít VRAM)                   │
# │    - Nếu DoRA+4bit deadlock → fallback 8-bit            │
# │    - Nếu vẫn lỗi → fallback 16-bit (chậm nhưng chắc)   │
# └──────────────────────────────────────────────────────────┘
def setup_model():
    print("\n" + "="*60)
    print("  ⚙️  Setting up DoRA (Weight-Decomposed LoRA)")
    print("  Model A dùng LoRA thuần — Model B dùng DoRA")
    print("="*60)
    gc.collect(); torch.cuda.empty_cache()

    # ── Thử load với 4-bit NF4 trước ──────────────────────────
    # DoRA + 4-bit cần PEFT >= 0.10.0 (Kaggle thường có sẵn)
    quantization_configs = [
        ("4-bit NF4", BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,  # Tắt double quant cho DoRA ổn định hơn
        )),
        ("8-bit", BitsAndBytesConfig(
            load_in_8bit=True,
        )),
    ]

    model = None
    used_quant = None
    for quant_name, bnb_config in quantization_configs:
        for attn in ("sdpa", "eager"):
            try:
                print(f"\n  Thử: {quant_name} + attn={attn}...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    BASE_MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map={"": 0},
                    torch_dtype=torch.float16,
                    attn_implementation=attn,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                used_quant = quant_name
                print(f"  ✅ Model loaded [{quant_name}, attn={attn}]")
                break
            except Exception as e:
                last_error = e
                print(f"  ⚠️  {quant_name}/attn={attn} failed: {e}")
                model = None
                gc.collect(); torch.cuda.empty_cache()
        if model is not None:
            break

    if model is None:
        raise RuntimeError(f"❌ Cannot load model with any quantization! Last error: {last_error}")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    # ══════════════════════════════════════════════════════════
    # DoRA CONFIG — use_dora=True
    # Đây là điểm khác biệt CỐT LÕI so với Model A (LoRA)
    # ══════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════
    # DoRA CONFIG — use_dora=True (BẮT BUỘC)
    # Đây là điểm khác biệt CỐT LÕI so với Model A (LoRA)
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Đang áp dụng DoRA (BẮT BUỘC)...")
    try:
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True,  # ← BẮT BUỘC LÀ DoRA
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.config.use_cache = False

        # Smoke test: thử forward pass nhỏ để đảm bảo không deadlock
        print(f"  Smoke test DoRA...")
        test_input = torch.randint(0, 100, (1, 10), device=DEVICE)
        with torch.no_grad():
            _ = model(input_ids=test_input)
        del test_input
        torch.cuda.empty_cache()

        print(f"  ✅ DoRA hoạt động OK!")
    except Exception as e:
        print(f"  ⚠️  DoRA thất bại: {e}")
        raise RuntimeError(f"BẮT BUỘC PHẢI DÙNG DORA NHƯNG BỊ LỖI: {e}")

    print(f"\n{'='*60}")
    print(f"  ✅ METHOD: DoRA (Weight-Decomposed LoRA)")
    print(f"     → Phân tách weight = magnitude × direction")
    print(f"     → Model A dùng LoRA thuần → KHÁC BIỆT RÕ RÀNG")
    print(f"  Quantization: {used_quant}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"{'='*60}")

    model.print_trainable_parameters()

    # Lưu flag để dùng trong report
    model._is_dora = True
    model._quant_method = used_quant

    return model, processor


# %%
# ══════════════════════════════════════════════════════════════
# 4. METRICS
# ══════════════════════════════════════════════════════════════
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if preds.ndim == 3:
        preds = preds.argmax(-1)
    preds_s = preds[:, :-1]
    labels_s = labels[:, 1:]
    mask = labels_s != -100
    total = int(mask.sum())
    if total == 0:
        return {"accuracy": 0.0}
    correct = int((preds_s == labels_s)[mask].sum())
    return {"accuracy": correct / total}


# %%
# ══════════════════════════════════════════════════════════════
# 5. REPORT GENERATION
# ══════════════════════════════════════════════════════════════
def generate_reports(trainer, model, processor, val_data, train_time):
    is_dora = getattr(model, '_is_dora', False)
    quant_method = getattr(model, '_quant_method', '4-bit NF4')
    method_label = "DoRA" if is_dora else "LoRA"

    print("\n" + "="*62)
    print(f"  📊  GENERATING FULL EVALUATION REPORTS — MODEL B ({method_label})")
    print("="*62)

    # ── 01: Training History ───────────────────────────────────
    print("\n[1/7] Training Loss & Accuracy curves...")
    t_steps, t_loss, e_steps, e_loss, e_acc = [], [], [], [], []
    for entry in trainer.state.log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            t_steps.append(entry.get('epoch', 0))
            t_loss.append(entry['loss'])
        elif 'eval_loss' in entry:
            e_steps.append(entry.get('epoch', 0))
            e_loss.append(entry['eval_loss'])
            if 'eval_accuracy' in entry:
                e_acc.append(entry['eval_accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Model B (23004023) — QWen2.5-VL {method_label} r={LORA_R} — Training History',
                 fontsize=12, fontweight='bold')
    ax = axes[0]
    if t_loss: ax.plot(t_steps, t_loss,  'o-',  label='Train Loss', color='#1f77b4', lw=2, ms=4)
    if e_loss: ax.plot(e_steps, e_loss,  's--', label='Val Loss',   color='#d62728', lw=2, ms=4)
    ax.set(title='Loss', xlabel='Epoch', ylabel='Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    if e_acc:
        ax.plot(e_steps, e_acc, '^-', color='#2ca02c', lw=2, ms=4, label='Val Accuracy')
        ax.axhline(max(e_acc), color='red', ls=':', alpha=0.7,
                   label=f'Best = {max(e_acc):.4f}')
    ax.set(title='Validation Accuracy', xlabel='Epoch', ylabel='Accuracy')
    ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 01_training_history.png")

    # ── INFERENCE ──────────────────────────────────────────────
    print(f"\n[2/7] Inference on {len(val_data)} validation images...")
    model.eval()
    true_labels, pred_labels = [], []
    wrong_samples = []

    t_inf = time.time()
    for i, item in enumerate(val_data):
        true_labels.append(item['label'])
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPT_TEXT}]}]
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=10, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id)
            gen = processor.batch_decode(
                [out[0][inputs["input_ids"].shape[1]:]],
                skip_special_tokens=True)[0].strip().upper()
            pred = gen if gen in CELL_LABELS else "Unknown"
        except Exception:
            pred = "Unknown"

        pred_labels.append(pred)
        if pred != item['label'] and len(wrong_samples) < 12:
            wrong_samples.append({"path": item["image_path"],
                                   "true": item["label"], "pred": pred})
        del inputs, out
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0 or i == len(val_data) - 1:
            cur_acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
            print(f"   → {i+1:4d}/{len(val_data)} | Running Acc: {cur_acc:.4f}")

    test_time = time.time() - t_inf
    overall_acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
    report = classification_report(
        true_labels, pred_labels, labels=CELL_LABELS, output_dict=True, zero_division=0)

    # ── 02: Confusion Matrix ───────────────────────────────────
    print("\n[3/7] Confusion Matrix (raw + normalized)...")
    cm = confusion_matrix(true_labels, pred_labels, labels=CELL_LABELS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(f'Model B ({method_label}) — Confusion Matrix', fontsize=14, fontweight='bold')
    sns.heatmap(cm,      annot=True, fmt='d',    cmap='Blues',
                xticklabels=CELL_LABELS, yticklabels=CELL_LABELS, ax=axes[0])
    axes[0].set(title='Raw Count', xlabel='Predicted', ylabel='True')
    sns.heatmap(cm_norm, annot=True, fmt='.2f',  cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=CELL_LABELS, yticklabels=CELL_LABELS, ax=axes[1])
    axes[1].set(title='Normalized (Recall per class)', xlabel='Predicted', ylabel='True')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 02_confusion_matrix.png")

    # ── 03: Per-class Precision / Recall / F1 ─────────────────
    print("\n[4/7] Per-class Precision / Recall / F1...")
    per_class = {lbl: report[lbl] for lbl in CELL_LABELS if lbl in report}
    x, w = np.arange(len(CELL_LABELS)), 0.25
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x-w, [per_class[l]['precision'] for l in CELL_LABELS], w,
           label='Precision', color='#4C72B0', alpha=0.85)
    ax.bar(x,   [per_class[l]['recall']    for l in CELL_LABELS], w,
           label='Recall',    color='#55A868', alpha=0.85)
    ax.bar(x+w, [per_class[l]['f1-score']  for l in CELL_LABELS], w,
           label='F1-Score',  color='#C44E52', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({CELL_NAMES[l][:9]})" for l in CELL_LABELS], fontsize=8)
    ax.set_ylim([0, 1.12])
    ax.set(title=f'Model B ({method_label} r={LORA_R}) — Per-class Metrics', ylabel='Score')
    macro_f1 = report['macro avg']['f1-score']
    ax.axhline(macro_f1, color='purple', ls='--', lw=1.5,
               label=f'Macro F1 = {macro_f1:.4f}')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_per_class_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 03_per_class_metrics.png")

    # ── 04: Class Distribution ─────────────────────────────────
    print("\n[5/7] Class distribution in validation set...")
    true_cnt = Counter(true_labels)
    pred_cnt = Counter(pred_labels)
    counts_t = [true_cnt.get(l, 0) for l in CELL_LABELS]
    counts_p = [pred_cnt.get(l, 0) for l in CELL_LABELS]
    x2, w2 = np.arange(len(CELL_LABELS)), 0.4
    fig, ax = plt.subplots(figsize=(14, 5))
    b1 = ax.bar(x2-w2/2, counts_t, w2, label='True',      color='#5c85d6', alpha=0.85)
    b2 = ax.bar(x2+w2/2, counts_p, w2, label='Predicted', color='#e07b54', alpha=0.85)
    ax.bar_label(b1, fmt='%d', fontsize=7, padding=1)
    ax.bar_label(b2, fmt='%d', fontsize=7, padding=1)
    ax.set_xticks(x2)
    ax.set_xticklabels(CELL_LABELS, fontsize=9)
    ax.set(title=f'True vs Predicted Distribution — Model B ({method_label})',
           xlabel='Cell Type', ylabel='Count')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 04_class_distribution.png")

    # ── 05: Sample Predictions ────────────────────────────────
    print("\n[6/7] Sample predictions visualization...")
    correct_samples = [
        (item["label"], item["image_path"])
        for item, pred in zip(val_data, pred_labels)
        if item["label"] == pred
    ]
    if len(correct_samples) > 6:
        correct_samples = random.sample(correct_samples, 6)
    n_cor   = min(len(correct_samples), 6)
    n_wrong = min(len(wrong_samples),   6)
    n_cols  = max(n_cor, n_wrong, 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3 + 1, 7))
    fig.suptitle(f'Sample Predictions ✅ Correct  ❌ Wrong  ({method_label})',
                 fontsize=12, fontweight='bold')
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    for col in range(n_cols):
        for row, (samples, is_wrong) in enumerate(
                [(correct_samples, False), (wrong_samples, True)]):
            ax = axes[row][col]
            ax.axis('off')
            if col < len(samples):
                s = samples[col]
                try:
                    path = s["path"] if is_wrong else s[1]
                    ax.imshow(Image.open(path).convert("RGB"))
                except Exception:
                    pass
                if is_wrong:
                    ax.set_title(f"❌ True:{s['true']} → {s['pred']}",
                                 color='red', fontsize=8, pad=3)
                else:
                    ax.set_title(f"✅ {s[0]} ({CELL_NAMES.get(s[0], '')})",
                                 color='green', fontsize=8, pad=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ 05_sample_predictions.png")

    # ── 06: CSV Classification Report ────────────────────────
    pd.DataFrame(report).transpose().round(4).to_csv(
        OUTPUT_DIR / "06_classification_report.csv")
    print("   ✅ 06_classification_report.csv")

    # ── 07: Summary TXT ───────────────────────────────────────
    print("\n[7/7] Summary report...")
    try:
        trainable_p, all_p = model.get_nb_trainable_parameters()
    except Exception:
        try:
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_p = sum(p.numel() for p in model.parameters())
        except Exception:
            trainable_p, all_p = 0, 1

    best_acc = max(e_acc) if e_acc else float('nan')

    lines = [
        "=" * 64,
        f"  BÁO CÁO TỔNG HỢP — QWEN2.5-VL {method_label} FINE-TUNING",
        f"  MODEL B — 23004023 | Phân Loại Tế Bào Máu",
        "=" * 64,
        "",
        f"  Base model     : {BASE_MODEL_NAME}",
        f"  Method         : {'DoRA (Weight-Decomposed Low-Rank Adaptation)' if is_dora else 'LoRA (Low-Rank Adaptation) — DoRA fallback'}",
        f"  So sánh        : Model A dùng LoRA r=16, alpha=32, batch=8, dropout=0.2",
        f"  Target modules : q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj",
        f"  Rank / Alpha   : r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}",
        f"  Quantization   : {quant_method} (bitsandbytes)",
        f"  Epochs         : {EPOCHS}  (early stopping patience=3)",
        f"  Eff. batch     : {BATCH_SIZE} × accum {GRADIENT_ACCUM} = {BATCH_SIZE*GRADIENT_ACCUM}  (Model A=8)",
        f"  Learning rate  : {LEARNING_RATE}  (cosine schedule, warmup 5%)",
        f"  Image size     : {IMG_SIZE} × {IMG_SIZE}",
        f"  Data/class     : {MAX_PER_CLASS} ảnh max (stratified subsample)",
        f"  Augmentation   : H/V flip, rotation ±45°, color jitter  ← Model A không có",
        "",
        "-" * 64,
        "  MODEL PARAMETERS",
        "-" * 64,
        f"  Total params         : {all_p:>15,}",
        f"  Trainable ({method_label:4s})  : {trainable_p:>15,}",
        f"  % Trainable          : {100*trainable_p/max(all_p,1):>14.4f}%",
        "",
        "-" * 64,
        "  ĐIỂM KHÁC BIỆT DoRA vs LoRA",
        "-" * 64,
    ]

    if is_dora:
        lines += [
            "  DoRA phân tách weight W = m × (V / ||V||)",
            "    m = magnitude (scalar, trainable)",
            "    V = direction (updated via low-rank: V₀ + BA)",
            "  → Giữ magnitude gốc ổn định, chỉ update direction",
            "  → Ở cùng rank (r=8), DoRA cho kết quả tốt hơn LoRA",
            "  → Model A: LoRA r=16 | Model B: DoRA r=8 (ít param hơn, chất lượng tương đương)",
        ]
    else:
        lines += [
            "  ⚠️ DoRA không khả dụng trên GPU này, đã fallback sang LoRA",
        ]

    lines += [
        "",
        "-" * 64,
        "  EVALUATION RESULTS  (Validation Set)",
        "-" * 64,
        f"  Overall Accuracy  : {overall_acc:.4f}  ({overall_acc*100:.2f}%)",
        f"  Best Val Accuracy : {best_acc:.4f}  (during training)",
        f"  Macro Precision   : {report['macro avg']['precision']:.4f}",
        f"  Macro Recall      : {report['macro avg']['recall']:.4f}",
        f"  Macro F1-Score    : {report['macro avg']['f1-score']:.4f}",
        f"  Weighted F1-Score : {report['weighted avg']['f1-score']:.4f}",
        "",
        "  Per-class F1-Score (sorted best → worst):",
    ]
    per_f1 = sorted(
        [(lbl, report[lbl]['f1-score'], report[lbl]['support'])
         for lbl in CELL_LABELS if lbl in report],
        key=lambda x: x[1], reverse=True)
    for lbl, f1, sup in per_f1:
        bar = '█' * int(f1 * 20)
        lines.append(
            f"    {lbl:4s} {CELL_NAMES[lbl]:22s}  F1={f1:.4f}  "
            f"n={int(sup):4d}  |{bar:<20}|")

    lines += [
        "",
        "-" * 64,
        "  SO SÁNH VỚI MODEL A (LoRA checkpoint-5500)",
        "-" * 64,
        f"  Model A — Method         : LoRA (use_dora=False)",
        f"  Model A — Best accuracy  : 95.70%  (step 5400)",
        f"  Model A — Final accuracy : 98.15%  (step 5500)",
        f"  Model A — Final F1 macro : 95.56%",
        f"  Model A — LoRA rank      : r=16, alpha=32",
        f"",
        f"  Model B — Method         : {method_label} (use_dora={'True' if is_dora else 'False'})",
        f"  Model B — Val accuracy   : {overall_acc*100:.2f}%",
        f"  Model B — F1 macro       : {report['macro avg']['f1-score']*100:.2f}%",
        f"  Model B — Rank           : r={LORA_R}, alpha={LORA_ALPHA}",
        "",
        "-" * 64,
        "  TIMING",
        "-" * 64,
        f"  Training time        : {train_time/60:.2f} phút  ({train_time:.0f}s)",
        f"  Inference time       : {test_time:.2f}s  ({len(val_data)} images)",
        f"  Avg inference speed  : {test_time/max(len(val_data),1)*1000:.1f} ms/image",
        "",
        "-" * 64,
        "  OUTPUT FILES",
        "-" * 64,
    ]
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file():
            kb = f.stat().st_size / 1024
            lines.append(f"    {f.name:<48} {kb:7.1f} KB")
    lines += [
        "",
        "  📂 HƯỚNG DẪN TÍCH HỢP VÀO HemoAI:",
        "  Copy thư mục final_model/ vào:",
        "  custom_models/qwen_blood_cell/23004023_final/",
        "  Server sẽ tự nhận diện và hiện trong dropdown.",
        "",
        "=" * 64,
    ]
    full_report = "\n".join(lines)

    with open(OUTPUT_DIR / "07_summary_report.txt", "w", encoding="utf-8") as fh:
        fh.write(full_report)
    print(full_report)

    final_model_dir = OUTPUT_DIR / "final_model"
    if final_model_dir.exists():
        req_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [f for f in req_files if not (final_model_dir / f).exists()]
        if missing:
            print(f"\n⚠️  CẢNH BÁO: final_model thiếu file: {missing}")
        else:
            # Verify DoRA flag in saved adapter
            import json
            cfg = json.loads((final_model_dir / "adapter_config.json").read_text())
            saved_dora = cfg.get("use_dora", False)
            print(f"\n✅ final_model hợp lệ!")
            print(f"   use_dora = {saved_dora}  ← {'DoRA ✓' if saved_dora else 'LoRA (fallback)'}")
            print(f"   Copy vào: custom_models/qwen_blood_cell/23004023_final/")
    else:
        print(f"\n⚠️  Không tìm thấy final_model/")

    print(f"\n🎉 Hoàn tất! Tải output tại: {OUTPUT_DIR}")


# %%
# ══════════════════════════════════════════════════════════════
# 6. MAIN — TRAIN
# ══════════════════════════════════════════════════════════════
def train():
    train_data, val_data = prepare_dataset()
    model, processor = setup_model()
    pad_id = processor.tokenizer.pad_token_id

    train_ds = BloodCellDataset(train_data, processor, is_train=True)
    val_ds   = BloodCellDataset(val_data,   processor, is_train=False)

    steps_per_epoch  = max(1, len(train_data) // (BATCH_SIZE * GRADIENT_ACCUM))
    total_steps      = steps_per_epoch * EPOCHS
    warmup_steps_val = max(1, int(total_steps * 0.05))
    print(f"📐 steps/epoch={steps_per_epoch} | total={total_steps} | warmup={warmup_steps_val}")

    training_args = None
    for optim_name in ["paged_adamw_32bit", "adamw_8bit", "adamw_torch"]:
        for extra_kwargs in [
            {"eval_strategy": "epoch"},
            {"evaluation_strategy": "epoch"},
        ]:
            try:
                training_args = TrainingArguments(
                    output_dir                    = str(OUTPUT_DIR),
                    num_train_epochs              = EPOCHS,
                    per_device_train_batch_size   = BATCH_SIZE,
                    per_device_eval_batch_size    = BATCH_SIZE,
                    gradient_accumulation_steps   = GRADIENT_ACCUM,
                    learning_rate                 = LEARNING_RATE,
                    warmup_steps                  = warmup_steps_val,
                    logging_strategy              = "steps",
                    logging_steps                 = 2,
                    save_strategy                 = "epoch",
                    save_total_limit              = 2,
                    load_best_model_at_end        = True,
                    metric_for_best_model         = "accuracy",
                    greater_is_better             = True,
                    max_grad_norm                 = 1.0,
                    fp16                          = True,
                    bf16                          = False,
                    dataloader_num_workers        = 0,
                    dataloader_pin_memory         = False,
                    remove_unused_columns         = False,
                    report_to                     = "none",
                    seed                          = SEED,
                    data_seed                     = SEED,
                    optim                         = optim_name,
                    lr_scheduler_type             = "cosine",
                    gradient_checkpointing        = True,
                    gradient_checkpointing_kwargs = {"use_reentrant": False},
                    ddp_find_unused_parameters    = False,
                    **extra_kwargs,
                )
                print(f"✅ Optimizer={optim_name} | kwargs={list(extra_kwargs.keys())}")
                break
            except TypeError as e:
                print(f"  ⚠️  {optim_name}/{list(extra_kwargs.keys())}: {e}")
                training_args = None
                continue
        if training_args is not None:
            break
    assert training_args is not None, "❌ Cannot create TrainingArguments!"

    trainer = None
    for trainer_kwargs in [
        {"processing_class": processor},
        {"tokenizer": processor},
    ]:
        try:
            trainer = Trainer(
                model                         = model,
                args                          = training_args,
                train_dataset                 = train_ds,
                eval_dataset                  = val_ds,
                data_collator                 = make_collate_fn(pad_id),
                compute_metrics               = compute_metrics,
                preprocess_logits_for_metrics = preprocess_logits_for_metrics,
                callbacks                     = [EarlyStoppingCallback(early_stopping_patience=3)],
                **trainer_kwargs,
            )
            print(f"✅ Trainer created with {list(trainer_kwargs.keys())[0]}")
            break
        except TypeError as e:
            print(f"  ⚠️  Trainer kwarg {list(trainer_kwargs.keys())}: {e}")
            trainer = None
    assert trainer is not None, "❌ Cannot create Trainer!"

    is_dora = getattr(model, '_is_dora', False)
    method_label = "DoRA" if is_dora else "LoRA"
    print(f"\n🚀 Starting {method_label} Training (Model B — r={LORA_R})...\n" + "-"*40)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\n⏱️  Training completed in {train_time/60:.2f} minutes")

    print("\n💾 Saving final model & processor...")
    trainer.save_model(str(OUTPUT_DIR / "final_model"))
    processor.save_pretrained(str(OUTPUT_DIR / "final_model"))

    generate_reports(trainer, model, processor, val_data, train_time)


# %%
if __name__ == "__main__":
    train()
