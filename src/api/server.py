"""
FastAPI Server - Blood Cell Detection & Classification Demo
Main API server with all endpoints for the web demo.
"""
import os
import sys
import base64
import random
import threading
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
from src.models.model_registry import (
    DEFAULT_MODEL_ID,
    ModelRegistryError,
    public_model_list,
    resolve_model,
)
from src.models.xai_engine import XAIEngine

# ───────────────────────────────────────────────────────────
# App Setup
# ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Hệ thống phân tích tế bào máu",
    description="YOLO26 + Qwen2-VL/Qwen2.5-VL + XAI",
    version="1.1.0"
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
classifier_key: Optional[tuple[str, str, str]] = None
classifier_lock = threading.RLock()

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


def unload_classifier() -> None:
    """Unload the active Qwen model before any registry/model setting switch."""
    global classifier, classifier_key, xai_engine
    old_classifier = classifier
    classifier = None
    classifier_key = None
    if xai_engine is not None:
        xai_engine.set_model(None)
    if old_classifier is not None:
        old_classifier.unload()


def get_classifier(model_id=None, device=None, compression=None):
    """Resolve a stable model ID and keep at most one Qwen model in memory."""
    global classifier, classifier_key, xai_engine
    with classifier_lock:
        if classifier is not None and model_id is None and device is None and compression is None:
            return classifier
        if model_id is None and classifier is not None:
            requested_id = classifier.model_id
        else:
            requested_id = model_id or DEFAULT_MODEL_ID
        resolved = resolve_model(requested_id)
        requested_device = device or "auto"
        if compression:
            requested_compression = compression
        elif requested_device == "cpu":
            requested_compression = "full"
        else:
            try:
                import torch

                requested_compression = "4bit" if torch.cuda.is_available() else "full"
            except ImportError:
                requested_compression = "full"
        target_key = (resolved.model_id, requested_device, requested_compression)

        if classifier is not None and classifier_key == target_key:
            return classifier

        unload_classifier()
        print(
            f"[API] Loading {resolved.display_name} on {requested_device.upper()} "
            f"with {requested_compression} compression"
        )
        try:
            loaded = QWenClassifier(
                model_name=resolved.adapter_path,
                model_id=resolved.model_id,
                display_name=resolved.display_name,
                expected_base_model=resolved.base_model,
                device=requested_device,
                compression=requested_compression,
            )
        except Exception:
            # The previous model is already gone. Never preserve or reuse partial state.
            unload_classifier()
            raise

        classifier = loaded
        classifier_key = target_key
        if xai_engine is not None:
            xai_engine.set_model(classifier.get_vision_encoder(), classifier.device)
        print(
            f"[API] Loaded {resolved.architecture}; adapter={resolved.adapter_type}, "
            f"r={resolved.lora_r}, alpha={resolved.lora_alpha}, dropout={resolved.lora_dropout}"
        )
        return classifier


def classify_with_model(image_input, model_id=None, device=None, compression=None, top_k=3):
    """Shared resolver/load/inference path used by every classification endpoint."""
    with classifier_lock:
        clf = get_classifier(model_id, device=device, compression=compression)
        return clf.classify(image_input, top_k=top_k)


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
    return {"message": "Hệ thống phân tích tế bào máu", "status": "running"}


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
    yolo_models = []
    seen_ids = set()

    # Priority 1: Check trained YOLO weights from runs / outputs
    candidate_paths = [
        PROJECT_ROOT / "runs" / "detect" / "outputs" / "yolo26_bccd" / "weights" / "best.pt",
        PROJECT_ROOT / "outputs" / "yolo26_bccd" / "weights" / "best.pt",
        PROJECT_ROOT / "custom_models" / "yolo" / "best.pt",
    ]
    for cand in candidate_paths:
        if cand.exists() and str(cand) not in seen_ids:
            yolo_models.append({"id": str(cand), "name": "YOLO26 Custom (BCCD Fine-tuned)"})
            seen_ids.add(str(cand))
            break

    # Priority 2: Other custom YOLO models in custom_models/yolo
    custom_yolo_dir = PROJECT_ROOT / "custom_models" / "yolo"
    if custom_yolo_dir.exists():
        for pt_file in custom_yolo_dir.glob("*.pt"):
            if str(pt_file) not in seen_ids and pt_file.name != "best.pt":
                yolo_models.append({"id": str(pt_file), "name": f"[Custom] {pt_file.name}"})
                seen_ids.add(str(pt_file))

    # Priority 3: Pretrained base nano model
    base_yolo = PROJECT_ROOT / "yolo26n.pt"
    yolo_models.append({"id": "yolo26n.pt", "name": "YOLO26 Nano (Pretrained Base)"})

    # Fixed registry: no duplicates, clean metadata
    qwen_models = public_model_list()

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
        cls_result = classify_with_model(
            pil_crop,
            qwen_model,
            device=device,
            compression=qwen_compression,
            top_k=5,
        )

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
        result = classify_with_model(
            contents,
            qwen_model,
            device=device,
            compression=qwen_compression,
            top_k=top_k,
        )

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

        result = classify_with_model(
            str(img_path),
            qwen_model,
            device=device,
            compression=qwen_compression,
            top_k=5,
        )

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
        get_classifier(qwen_model, device=device, compression=qwen_compression)
        engine = get_xai_engine()
        classifications = []

        crops = detect_result.get("crops", [])
        for crop_info in crops[:10]:  # Limit to 10 cells
            crop_img = crop_info["image"]

            # Classify
            cls_result = classify_with_model(
                crop_img,
                qwen_model,
                device=device,
                compression=qwen_compression,
                top_k=3,
            )

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
    Mỗi phía dùng cùng registry/resolver và chỉ trả kết quả inference thật.
    """
    if not model_a or not model_b:
        raise HTTPException(status_code=400, detail="Phải chọn đủ hai model để so sánh.")
    if model_a == model_b:
        raise HTTPException(status_code=400, detail="Hai phía phải chọn hai model khác nhau.")
    try:
        resolved_models = {
            "model_a": resolve_model(model_a),
            "model_b": resolve_model(model_b),
        }
    except ModelRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        from PIL import Image as _PIL
        import io as _io

        contents = await file.read()
        pil_image = _PIL.open(_io.BytesIO(contents)).convert("RGB")

        results = {}
        for model_tag, resolved in resolved_models.items():
            try:
                cls_result = classify_with_model(
                    pil_image,
                    resolved.model_id,
                    device=device or "auto",
                    compression=compression,
                    top_k=top_k,
                )
                results[model_tag] = {
                    "success": True,
                    "model_id": resolved.model_id,
                    "model_name": resolved.display_name,
                    **cls_result,
                }
            except Exception as e:
                results[model_tag] = {
                    "success": False,
                    "model_id": resolved.model_id,
                    "model_name": resolved.display_name,
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
