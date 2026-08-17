"""
FastAPI Server - Blood Cell Detection & Classification Demo
Main API server with all endpoints for the web demo.
"""
import os
import sys
import io
import base64
import random
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolo_detector import YOLODetector, numpy_to_base64
from src.models.qwen_classifier import QWenClassifier, CELL_TYPES, CELL_LABELS
from src.models.xai_engine import XAIEngine

# ───────────────────────────────────────────────────────────
# App Setup
# ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Blood Cell Detection & Classification Demo",
    description="YOLO26 + QWen2.5-VL + XAI for blood cell analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────────────────
# Global Models (lazy loaded)
# ───────────────────────────────────────────────────────────
detector: Optional[YOLODetector] = None
classifier: Optional[QWenClassifier] = None
xai_engine: Optional[XAIEngine] = None

DATASET_PATH = PROJECT_ROOT / "Dataset-Crop"


def get_detector(model_path=None):
    global detector
    
    # If no model requested and we have one loaded, return it
    if model_path is None and detector is not None:
        return detector
        
    if not model_path:
        # Check for trained model
        trained_path = PROJECT_ROOT / "outputs" / "yolo26_bccd" / "weights" / "best.pt"
        model_path = str(trained_path) if trained_path.exists() else "yolo26n.pt"
        
    if detector is None or getattr(detector, 'model_path', None) != model_path:
        detector = YOLODetector(model_path=model_path)
    return detector


def _normalize_model_id(model_id: Optional[str]) -> str:
    if not model_id:
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    p = Path(model_id)
    if p.exists():
        return str(p.resolve())
    return model_id


def get_classifier(model_name=None, device=None, compression=None):
    global classifier, xai_engine
    
    # If classifier is already loaded and no specific model/device/compression is requested, return it
    if classifier is not None and model_name is None and device is None and compression is None:
        return classifier
        
    # If no model requested but we have one loaded, use the current one
    if model_name is None and classifier is not None:
        model_name = getattr(classifier, 'model_name', None)
        
    if not device:
        # Use current device if available, otherwise default to cpu
        device = getattr(classifier, 'device', 'cpu') if classifier else "cpu"
        
    if not compression:
        compression = getattr(classifier, 'compression', '4bit') if classifier else "4bit"

    target_model_id = _normalize_model_id(model_name)
    current_model_id = _normalize_model_id(getattr(classifier, 'model_name', None)) if classifier else None
    current_device = getattr(classifier, 'device', None)
    current_compression = getattr(classifier, 'compression', None)

    if (
        classifier is None
        or current_model_id != target_model_id
        or current_device != device
        or current_compression != compression
    ):
        print(f"[API] Loading QWen classifier: {model_name} (norm: {target_model_id}) on {device.upper()} with {compression} compression")
        classifier = QWenClassifier(model_name=model_name, device=device, compression=compression)
        # Reset XAI engine model reference when classifier changes
        if xai_engine is not None and classifier.model is not None:
            vision_encoder = classifier.get_vision_encoder()
            if vision_encoder is not None:
                xai_engine.set_model(vision_encoder, classifier.device)
        print(f"[API] QWen classifier loaded. Model: {classifier.model is not None}, Device: {classifier.device}")
    return classifier


def get_xai_engine():
    global xai_engine
    if xai_engine is None:
        xai_engine = XAIEngine()
        # Try to get vision encoder from classifier
        clf = get_classifier()
        if clf.model is not None:
            vision_encoder = clf.get_vision_encoder()
            if vision_encoder is not None:
                xai_engine.set_model(vision_encoder, clf.device)
    return xai_engine


# ───────────────────────────────────────────────────────────
# Static Files
# ───────────────────────────────────────────────────────────
WEB_DIR = PROJECT_ROOT / "web"

# Mount static files
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def root():
    """Serve main page."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Blood Cell Analysis API", "status": "running"}


# ───────────────────────────────────────────────────────────
# API Endpoints
# ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models": {
            "yolo": detector is not None,
            "qwen": classifier is not None,
            "xai": xai_engine is not None
        }
    }


@app.get("/api/models")
async def get_models():
    """Get available YOLO and QWen models."""
    trained_yolo = PROJECT_ROOT / "outputs" / "yolo26_bccd" / "weights" / "best.pt"
    yolo_models = [{"id": "yolo26n.pt", "name": "YOLO26 Nano (Pretrained Base)"}]
    if trained_yolo.exists():
        yolo_models.insert(0, {"id": str(trained_yolo), "name": "YOLO26 Custom (BCCD Fine-tuned)"})
        
    # Scan custom YOLO models
    custom_yolo_dir = PROJECT_ROOT / "custom_models" / "yolo"
    if custom_yolo_dir.exists():
        for pt_file in custom_yolo_dir.glob("*.pt"):
            yolo_models.append({"id": str(pt_file), "name": f"[Custom] {pt_file.name}"})

    # Collect QWen checkpoints from outputs and custom_models
    qwen_models = []
    ckpt_dirs = []

    search_dirs = [
        PROJECT_ROOT / "outputs" / "qwen_blood_cell",
        PROJECT_ROOT / "custom_models" / "qwen_blood_cell",
        PROJECT_ROOT / "custom_models" / "qwen",
        # Tự động scan thêm các thư mục 23004023_* (Model B output từ Kaggle)
        *list(PROJECT_ROOT.glob("custom_models/23004023_*")),
        *list(PROJECT_ROOT.glob("custom_models/qwen_blood_cell/23004023_*")),
    ]

    for s_dir in search_dirs:
        if not s_dir or not s_dir.exists():
            continue
        # Kiểm tra xem thư mục con là adapter trực tiếp hay chứa nhiều checkpoint
        if (s_dir / "adapter_config.json").exists():
            # Thư mục chính là adapter (e.g. 23004023_final/)
            label = s_dir.name
            is_23004023 = "23004023" in label or "model_b" in label.lower()
            display = f"[Model B — 23004023] QWen DoRA r=8 ({label})" if is_23004023 else f"QWen LoRA ({label})"
            ckpt_dirs.append((9999 if is_23004023 else 1, str(s_dir), display))
            continue
        for item in s_dir.iterdir():
            if item.is_dir() and (item / "adapter_config.json").exists():
                # Extract step number if available
                step = 0
                if item.name.startswith("checkpoint-"):
                    try:
                        step = int(item.name.split("-")[-1])
                    except ValueError:
                        step = 0
                label = item.name
                is_23004023 = "23004023" in str(item) or "23004023" in label
                display = f"[Model B — 23004023] QWen DoRA r=8 ({label})" if is_23004023 else f"QWen LoRA ({label})"
                ckpt_dirs.append((step, str(item), display))

    # Sort checkpoints by step descending
    ckpt_dirs.sort(key=lambda x: x[0], reverse=True)

    if ckpt_dirs:
        # Mark highest step as Best (Model B luôn hiện đầu nếu có)
        top_step, top_id, top_display = ckpt_dirs[0]
        if "Model B" in top_display:
            qwen_models.append({"id": top_id, "name": top_display})
        else:
            qwen_models.append({"id": top_id, "name": f"[Best Fine-tuned] QWen LoRA ({Path(top_id).name})"})
        for step, c_id, c_display in ckpt_dirs[1:]:
            if "Model B" in c_display:
                qwen_models.append({"id": c_id, "name": c_display})
            else:
                qwen_models.append({"id": c_id, "name": f"QWen LoRA ({Path(c_id).name})"})

    # Always include base model option
    qwen_models.append({"id": "Qwen/Qwen2.5-VL-3B-Instruct", "name": "QWen2.5-VL-3B-Instruct (Base Model)"})

    return {
        "yolo_models": yolo_models,
        "qwen_models": qwen_models
    }


@app.get("/api/cell-types")
async def get_cell_types():
    """Get all 12 cell type information."""
    result = []
    for code, info in CELL_TYPES.items():
        # Count samples in dataset
        cell_dir = DATASET_PATH / code
        sample_count = 0
        if cell_dir.exists():
            sample_count = len(list(cell_dir.glob("*.jpg"))) + \
                          len(list(cell_dir.glob("*.jpeg"))) + \
                          len(list(cell_dir.glob("*.png")))

        result.append({
            "code": code,
            "name": info["name"],
            "full_name": info["full_name"],
            "description": info["description"],
            "color": info["color"],
            "sample_count": sample_count
        })
    return {"cell_types": result, "total_types": len(result)}


@app.get("/api/samples")
async def get_samples(cell_type: Optional[str] = None, limit: int = 8):
    """Get sample images from dataset."""
    samples = []

    if cell_type:
        types_to_scan = [cell_type] if cell_type in CELL_LABELS else []
    else:
        types_to_scan = CELL_LABELS

    for ct in types_to_scan:
        cell_dir = DATASET_PATH / ct
        if not cell_dir.exists():
            continue

        images = list(cell_dir.glob("*.jpg")) + list(cell_dir.glob("*.jpeg")) + list(cell_dir.glob("*.png"))
        selected = random.sample(images, min(limit, len(images)))

        for img_path in selected:
            # Read and encode image
            try:
                with open(img_path, "rb") as f:
                    img_data = f.read()
                img_b64 = base64.b64encode(img_data).decode('utf-8')
                samples.append({
                    "cell_type": ct,
                    "cell_name": CELL_TYPES[ct]["name"],
                    "filename": img_path.name,
                    "image_base64": img_b64,
                    "color": CELL_TYPES[ct]["color"]
                })
            except Exception:
                continue

    # Shuffle if mixed types
    if not cell_type:
        random.shuffle(samples)
        samples = samples[:limit * 3]  # Limit total

    return {"samples": samples, "total": len(samples)}


@app.get("/api/sample-image/{cell_type}/{filename}")
async def get_sample_image(cell_type: str, filename: str):
    """Serve a specific sample image."""
    img_path = DATASET_PATH / cell_type / filename
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(img_path))


@app.post("/api/detect")
async def detect_cells(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    yolo_model: Optional[str] = Form(None)
):
    """
    Detect blood cells in an uploaded image using YOLO26.
    Also returns crop images of each detected cell.
    """
    try:
        contents = await file.read()
        det = get_detector(yolo_model)
        # Use detect_and_crop to get both detection + crop images
        result = det.detect_and_crop(contents, conf=confidence)

        # Convert annotated image to base64
        annotated_b64 = None
        if result["annotated_image"] is not None:
            annotated_b64 = numpy_to_base64(result["annotated_image"])

        # Convert each crop image to base64
        crops = []
        for crop_info in result.get("crops", []):
            crop_b64 = numpy_to_base64(crop_info["image"])
            crops.append({
                "index": crop_info["index"],
                "box": crop_info["box"],
                "detect_label": crop_info["detect_label"],
                "detect_score": crop_info["detect_score"],
                "crop_image_base64": crop_b64
            })

        return {
            "success": True,
            "boxes": result["boxes"],
            "labels": result["labels"],
            "scores": result["scores"],
            "annotated_image_base64": annotated_b64,
            "crops": crops,
            "summary": result["summary"],
            "total_cells": result["total_cells"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/crop-classify")
async def crop_and_classify(
    file: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    qwen_model: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    qwen_compression: Optional[str] = Form(None)
):
    """
    Crop a region from an image and classify it using QWen2.5-VL.
    Used when user clicks on a detected cell in the Detection tab.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image

        contents = await file.read()
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            return {"success": False, "error": "Could not decode image"}

        h, w = img_np.shape[:2]
        # Clamp box coordinates to image boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # Crop with small padding
        padding = 5
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        crop = img_np[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            return {"success": False, "error": "Empty crop region"}

        # Convert crop to PIL RGB for classifier
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        # Classify
        clf = get_classifier(qwen_model, device=device, compression=qwen_compression)
        cls_result = clf.classify(pil_crop, top_k=5)

        # Encode crop image for display
        crop_b64 = numpy_to_base64(crop)

        return {
            "success": True,
            "crop_image_base64": crop_b64,
            "box": [x1, y1, x2, y2],
            "classification": cls_result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/classify")
async def classify_cell(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    qwen_model: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    qwen_compression: Optional[str] = Form(None)
):
    """
    Classify a cropped cell image using QWen2.5-VL.
    """
    try:
        contents = await file.read()
        clf = get_classifier(qwen_model, device=device, compression=qwen_compression)
        result = clf.classify(contents, top_k=top_k)

        return {
            "success": True,
            **result
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/classify-sample")
async def classify_sample(
    cell_type: str = Form(...),
    filename: str = Form(...),
    qwen_model: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    qwen_compression: Optional[str] = Form(None)
):
    """Classify a sample image from the dataset."""
    try:
        img_path = DATASET_PATH / cell_type / filename
        if not img_path.exists():
            raise HTTPException(404, "Sample not found")

        clf = get_classifier(qwen_model, device=device, compression=qwen_compression)
        result = clf.classify(str(img_path), top_k=5)

        # Add ground truth
        result["ground_truth"] = cell_type
        result["ground_truth_name"] = CELL_TYPES[cell_type]["name"]
        result["is_correct"] = result["predicted_class"] == cell_type

        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/xai")
async def generate_xai(
    file: UploadFile = File(...),
    method: str = Form("HiResCAM"),
    alpha: float = Form(0.5)
):
    """
    Generate XAI explanation heatmap for a cell image.
    """
    try:
        contents = await file.read()
        engine = get_xai_engine()
        result = engine.generate_explanation(contents, method=method, alpha=alpha)

        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/xai-all")
async def generate_all_xai(
    file: UploadFile = File(...),
    alpha: float = Form(0.5)
):
    """
    Generate XAI explanations using all 4 methods.
    """
    try:
        contents = await file.read()
        engine = get_xai_engine()
        results = engine.generate_all_explanations(contents, alpha=alpha)

        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/xai-sample")
async def xai_sample(
    cell_type: str = Form(...),
    filename: str = Form(...),
    method: str = Form("HiResCAM"),
    alpha: float = Form(0.5)
):
    """Generate XAI for a sample image from dataset."""
    try:
        img_path = DATASET_PATH / cell_type / filename
        if not img_path.exists():
            raise HTTPException(404, "Sample not found")

        engine = get_xai_engine()
        result = engine.generate_explanation(str(img_path), method=method,
                                              alpha=alpha)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    xai_method: str = Form("HiResCAM"),
    xai_alpha: float = Form(0.5),
    yolo_model: Optional[str] = Form(None),
    qwen_model: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    qwen_compression: Optional[str] = Form(None)
):
    """
    Full pipeline: Detect → Crop → Classify → XAI
    """
    try:
        contents = await file.read()

        # Step 1: Detection
        det = get_detector(yolo_model)
        detect_result = det.detect_and_crop(contents, conf=confidence)

        annotated_b64 = None
        if detect_result["annotated_image"] is not None:
            annotated_b64 = numpy_to_base64(detect_result["annotated_image"])

        # Step 2 & 3: Classify each crop + XAI
        clf = get_classifier(qwen_model, device=device, compression=qwen_compression)
        engine = get_xai_engine()
        classifications = []

        crops = detect_result.get("crops", [])
        for crop_info in crops[:10]:  # Limit to 10 cells
            crop_img = crop_info["image"]

            # Classify
            cls_result = clf.classify(crop_img, top_k=3)

            # XAI on crop
            from PIL import Image
            import cv2
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            xai_result = engine.generate_explanation(pil_crop,
                                                       method=xai_method,
                                                       alpha=xai_alpha)

            # Encode crop image
            crop_b64 = numpy_to_base64(crop_img)

            classifications.append({
                "index": crop_info["index"],
                "box": crop_info["box"],
                "detect_label": crop_info["detect_label"],
                "detect_score": crop_info["detect_score"],
                "crop_image_base64": crop_b64,
                "classification": cls_result,
                "xai": {
                    "method": xai_result["method"],
                    "overlay_base64": xai_result.get("overlay_base64"),
                    "heatmap_base64": xai_result.get("heatmap_base64")
                }
            })

        return {
            "success": True,
            "detection": {
                "annotated_image_base64": annotated_b64,
                "boxes": detect_result["boxes"],
                "labels": detect_result["labels"],
                "scores": detect_result["scores"],
                "summary": detect_result["summary"],
                "total_cells": detect_result["total_cells"]
            },
            "classifications": classifications,
            "total_classified": len(classifications)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/api/xai-methods")
async def get_xai_methods():
    """Get available XAI methods info."""
    return {
        "methods": XAIEngine.METHOD_INFO,
        "available": XAIEngine.METHODS
    }


@app.post("/api/compare")
async def compare_models(
    file: UploadFile = File(...),
    model_a: Optional[str] = Form(None),
    model_b: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    compression: Optional[str] = Form(None),
    top_k: int = Form(5),
):
    """
    So sánh kết quả phân loại của 2 model QWen trên cùng 1 ảnh.
    Trả về top_k predictions, confidence, thời gian inference của mỗi model.
    """
    try:
        import time as _time
        from PIL import Image as _PIL
        import io as _io

        contents = await file.read()
        pil_image = _PIL.open(_io.BytesIO(contents)).convert("RGB")

        results = {}
        for model_tag, model_id in [("model_a", model_a), ("model_b", model_b)]:
            if not model_id:
                results[model_tag] = {
                    "error": "Chưa chọn model",
                    "model_id": None,
                    "success": False
                }
                continue
            t_start = _time.time()
            try:
                # Tạo classifier riêng cho mỗi lần gọi để đảm bảo đúng model
                clf_instance = QWenClassifier(
                    model_name=model_id,
                    device=device or "cpu",
                    compression=compression or "4bit"
                )
                cls_result = clf_instance.classify(pil_image, top_k=top_k)
                inference_ms = int((_time.time() - t_start) * 1000)
                results[model_tag] = {
                    "success": True,
                    "model_id": model_id,
                    "model_name": Path(model_id).name if "/" in model_id or "\\" in model_id else model_id,
                    "inference_ms": inference_ms,
                    **cls_result,
                }
            except Exception as e:
                results[model_tag] = {
                    "success": False,
                    "model_id": model_id,
                    "error": str(e),
                }

        return {
            "success": True,
            "results": results,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"success": False, "error": str(e)}


# ───────────────────────────────────────────────────────────
# Startup Event
# ───────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup."""
    print("=" * 60)
    print("  Blood Cell Detection & Classification Demo")
    print("  YOLO26 + QWen2.5-VL + XAI")
    print("=" * 60)
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Web UI:  {WEB_DIR}")

    # Create custom model directories if they don't exist
    (PROJECT_ROOT / "custom_models" / "yolo").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "custom_models" / "qwen").mkdir(parents=True, exist_ok=True)
    print(f"  Custom Models: {PROJECT_ROOT / 'custom_models'}")

    # Lazy load - models will be loaded on first request
    print("\n  Models will be loaded on first request.")
    print("  Open http://localhost:8000 in your browser.")
    print("=" * 60)
