"""
Demo Script: Quick test pipeline
YOLO26 Detection → QWen Classification → XAI Explanation
Chạy thử nghiệm nhanh pipeline phát hiện và phân loại tế bào máu

Usage:
    python demo.py                          # Chạy với ảnh mẫu có sẵn
    python demo.py --image path/to/img.jpg  # Chạy với ảnh tự chọn
    python demo.py --web                    # Khởi động web server
    python demo.py --train-yolo             # Train YOLO26
    python demo.py --train-qwen             # Fine-tune QWen
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment():
    """Kiểm tra môi trường và các thư viện cần thiết."""
    print("=" * 60)
    print("  HemoAI - Blood Cell Analysis System")
    print("  YOLO26 + QWen2.5-VL + XAI")
    print("=" * 60)

    # Check Python
    print(f"\n[CHECK] Python: {sys.version.split()[0]}")

    # Check PyTorch
    try:
        import torch
        print(f"[CHECK] PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[CHECK] CUDA: {torch.cuda.get_device_name(0)}")
            print(f"[CHECK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("[CHECK] CUDA: NOT AVAILABLE (using CPU)")
    except ImportError:
        print("[CHECK] PyTorch: NOT INSTALLED (pip install torch torchvision)")

    # Check Ultralytics
    try:
        import ultralytics
        print(f"[CHECK] Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("[CHECK] Ultralytics: NOT INSTALLED (pip install ultralytics)")

    # Check Transformers
    try:
        import transformers
        print(f"[CHECK] Transformers: {transformers.__version__}")
    except ImportError:
        print("[CHECK] Transformers: NOT INSTALLED")

    # Check PEFT
    try:
        import peft
        print(f"[CHECK] PEFT: {peft.__version__}")
    except ImportError:
        print("[CHECK] PEFT: NOT INSTALLED")

    # Check QWen VL utils
    try:
        import qwen_vl_utils
        print(f"[CHECK] qwen-vl-utils: INSTALLED")
    except ImportError:
        print("[CHECK] qwen-vl-utils: NOT INSTALLED")

    # Check GradCAM
    try:
        import pytorch_grad_cam
        print(f"[CHECK] grad-cam: INSTALLED")
    except ImportError:
        print("[CHECK] grad-cam: NOT INSTALLED")

    # Check Captum
    try:
        import captum
        print(f"[CHECK] captum: INSTALLED")
    except ImportError:
        print("[CHECK] captum: NOT INSTALLED")

    # Check Datasets
    try:
        import datasets
        print(f"[CHECK] Datasets: {datasets.__version__}")
    except ImportError:
        print("[CHECK] Datasets: NOT INSTALLED")

    # Check datasets exist
    bccd_path = PROJECT_ROOT.parent / "data" / "BCCD"
    dataset_crop_path = PROJECT_ROOT.parent / "data" / "Dataset-Crop"

    print(f"\n[CHECK] BCCD dataset: {'FOUND' if bccd_path.exists() else 'NOT FOUND'} at {bccd_path}")
    print(f"[CHECK] Dataset-Crop: {'FOUND' if dataset_crop_path.exists() else 'NOT FOUND'} at {dataset_crop_path}")

    # Check trained models
    yolo_trained = PROJECT_ROOT / "outputs" / "yolo26_bccd" / "weights" / "best.pt"
    qwen_lora = PROJECT_ROOT / "outputs" / "qwen_blood_cell" / "final_lora_adapter"
    qwen_final = PROJECT_ROOT / "outputs" / "qwen_blood_cell" / "final"

    print(f"[CHECK] YOLO trained: {'FOUND' if yolo_trained.exists() else 'NOT FOUND'}")
    print(f"[CHECK] QWen LoRA: {'FOUND' if qwen_lora.exists() else 'NOT FOUND'}")
    print(f"[CHECK] QWen final: {'FOUND' if qwen_final.exists() else 'NOT FOUND'}")


def run_demo(image_path=None):
    """Run end-to-end demo on a single image."""
    print("\n[DATA] Loading models...")

    # Load YOLO detector
    from src.models.yolo_detector import YOLODetector
    yolo_trained = PROJECT_ROOT / "outputs" / "yolo26_bccd" / "weights" / "best.pt"
    model_path = str(yolo_trained) if yolo_trained.exists() else None

    detector = YOLODetector(model_path=model_path)
    print(f"[DATA] YOLO model loaded: {model_path or 'yolo26n.pt (pretrained)'}")

    # Load QWen classifier
    from src.models.qwen_classifier import QWenClassifier
    qwen_lora = PROJECT_ROOT / "outputs" / "qwen_blood_cell" / "final_lora_adapter"
    qwen_final = PROJECT_ROOT / "outputs" / "qwen_blood_cell" / "final"

    if qwen_lora.exists() and (qwen_lora / "adapter_config.json").exists():
        qwen_model_path = str(qwen_lora)
    elif qwen_final.exists() and (qwen_final / "adapter_config.json").exists():
        qwen_model_path = str(qwen_final)
    else:
        qwen_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

    classifier = QWenClassifier(model_name=qwen_model_path, device="cpu")
    print(f"[DATA] QWen model: {qwen_model_path}")

    # Load XAI engine
    from src.models.xai_engine import XAIEngine
    xai = XAIEngine()
    if classifier.model is not None:
        vision_encoder = classifier.get_vision_encoder()
        if vision_encoder is not None:
            xai.set_model(vision_encoder, classifier.device)
    print("[DATA] XAI engine loaded")

    # Find test image
    if image_path and os.path.exists(image_path):
        test_img = image_path
    else:
        # Try to find a sample from BCCD test set
        bccd_test = PROJECT_ROOT.parent / "data" / "BCCD" / "images" / "test"
        if bccd_test.exists():
            images = list(bccd_test.glob("*"))
            if images:
                test_img = str(images[0])
            else:
                print("[DATA] No test images found. Using mock detection.")
                test_img = None
        else:
            print("[DATA] BCCD test set not found. Using mock detection.")
            test_img = None

    if not test_img:
        # Create a dummy image for demo
        import cv2
        import numpy as np
        dummy = np.ones((640, 640, 3), dtype=np.uint8) * 200
        test_img_path = PROJECT_ROOT / "_demo_test.jpg"
        cv2.imwrite(str(test_img_path), dummy)
        test_img = str(test_img_path)

    print(f"\n[DATA] Using test image: {test_img}")

    # ─── Step 1: YOLO Detection ───
    print("\n" + "=" * 60)
    print("  STEP 1: YOLO26 Detection")
    print("=" * 60)
    detect_result = detector.detect_and_crop(test_img, conf=0.25)
    print(f"  Total cells detected: {detect_result['total_cells']}")
    print(f"  Summary: {detect_result['summary']}")
    for box, label, score in zip(detect_result['boxes'], detect_result['labels'], detect_result['scores']):
        print(f"    [{label}] confidence={score:.3f} box={box}")

    # Save annotated image
    if detect_result['annotated_image'] is not None:
        import cv2
        output_path = PROJECT_ROOT / "_demo_yolo_result.jpg"
        cv2.imwrite(str(output_path), detect_result['annotated_image'])
        print(f"\n  Annotated image saved to: {output_path}")

    # ─── Step 2 & 3: QWen Classification + XAI ───
    print("\n" + "=" * 60)
    print("  STEP 2: QWen Classification + XAI")
    print("=" * 60)

    crops = detect_result.get('crops', [])
    if not crops:
        print("  No cells detected to classify.")
    else:
        for i, crop_info in enumerate(crops[:5]):  # Limit to first 5 crops
            crop_img = crop_info['image']
            print(f"\n  --- Cell {i+1} (YOLO: {crop_info['detect_label']}) ---")

            # Classify
            cls_result = classifier.classify(crop_img, top_k=3)
            print(f"  QWen prediction: {cls_result['predicted_class']} ({cls_result['class_name']})")
            print(f"  Confidence: {cls_result['confidence']:.4f}")
            print(f"  Top predictions:")
            for pred in cls_result['top_predictions']:
                print(f"    - {pred['class']} ({pred['name']}): {pred['confidence']:.4f}")

            # XAI
            from PIL import Image
            import cv2
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            xai_results = xai.generate_all_explanations(pil_crop, alpha=0.5)
            print(f"  XAI heatmaps generated: {list(xai_results.keys())}")

            # Save crop + heatmap
            crop_out = PROJECT_ROOT / f"_demo_cell_{i+1}_{cls_result['predicted_class']}.jpg"
            cv2.imwrite(str(crop_out), crop_img)
            print(f"  Crop saved to: {crop_out}")

    print("\n" + "=" * 60)
    print("  Demo completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="HemoAI - Blood Cell Detection & Classification Demo"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image for testing")
    parser.add_argument("--web", action="store_true",
                        help="Start web server")
    parser.add_argument("--train-yolo", action="store_true",
                        help="Train YOLO26 on BCCD dataset")
    parser.add_argument("--train-qwen", action="store_true",
                        help="Fine-tune QWen on Dataset-Crop")
    parser.add_argument("--check", action="store_true",
                        help="Check environment")
    parser.add_argument("--demo", action="store_true", default=True,
                        help="Run quick demo (default)")

    args = parser.parse_args()

    if args.check:
        check_environment()
    elif args.web:
        print("[SYSTEM] Starting web server...")
        os.chdir(PROJECT_ROOT)
        import uvicorn
        uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)
    elif args.train_yolo:
        print("[SYSTEM] Starting YOLO26 training...")
        exec(open(str(PROJECT_ROOT / "train_yolo.py")).read())
    elif args.train_qwen:
        print("[SYSTEM] Starting QWen fine-tuning...")
        exec(open(str(PROJECT_ROOT / "train_qwen.py")).read())
    else:
        run_demo(args.image)


if __name__ == "__main__":
    main()