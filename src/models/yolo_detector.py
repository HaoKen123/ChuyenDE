"""
YOLO26 Blood Cell Detector
Phát hiện tế bào máu từ ảnh blood smear sử dụng YOLO26
"""
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64


class YOLODetector:
    """Wrapper for YOLO26 blood cell detection model."""

    # BCCD classes
    CLASS_NAMES = {0: "RBC", 1: "WBC", 2: "Platelets"}
    CLASS_COLORS = {
        "RBC": (255, 82, 82),      # Red
        "WBC": (68, 138, 255),     # Blue
        "Platelets": (76, 175, 80) # Green
    }

    def __init__(self, model_path=None, conf_threshold=0.25):
        """
        Initialize YOLO26 detector.
        Args:
            model_path: Path to trained YOLO26 weights (.pt file)
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_path = model_path
        self._load_model(model_path)

    def _load_model(self, model_path):
        """Load YOLO26 model."""
        try:
            from ultralytics import YOLO

            if model_path and os.path.exists(model_path) and "yolo26n.pt" not in model_path:
                print(f"[YOLO] Loading trained model from {model_path}")
                self.model = YOLO(model_path)
                print("[YOLO] Model loaded successfully")
            else:
                # Sử dụng mock cho demo nếu chưa có model YOLO custom
                print(f"[YOLO] No trained custom model found (path={model_path}). Detection will use mock results for demo.")
                self.model = None
        except Exception as e:
            print(f"[YOLO] Warning: Could not load model: {e}")
            print("[YOLO] Detection will use mock results for demo")
            self.model = None

    def detect(self, image_input, conf=None):
        """
        Detect blood cells in an image.
        Args:
            image_input: Can be file path (str), PIL Image, numpy array, or bytes
            conf: Override confidence threshold
        Returns:
            dict with keys: boxes, labels, scores, annotated_image, summary
        """
        conf = conf or self.conf_threshold

        # Convert input to numpy array
        img_np = self._to_numpy(image_input)
        if img_np is None:
            return self._empty_result(image_input)

        if self.model is None:
            return self._mock_detect(img_np)

        # Run YOLO inference
        results = self.model(img_np, conf=conf, verbose=False)
        result = results[0]

        boxes = []
        labels = []
        scores = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().numpy())
                score = float(box.conf[0].cpu().numpy())
                cls_name = self.CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                labels.append(cls_name)
                scores.append(round(score, 4))

        # Draw annotations
        annotated = self._draw_boxes(img_np.copy(), boxes, labels, scores)

        # Summary statistics
        summary = {}
        for label in labels:
            summary[label] = summary.get(label, 0) + 1

        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "annotated_image": annotated,
            "summary": summary,
            "total_cells": len(boxes)
        }

    def detect_and_crop(self, image_input, conf=None, padding=5):
        """
        Detect cells and crop each detected cell.
        Returns detection result + list of cropped cell images.
        """
        result = self.detect(image_input, conf)
        img_np = self._to_numpy(image_input)

        crops = []
        h, w = img_np.shape[:2]

        for i, (box, label, score) in enumerate(
            zip(result["boxes"], result["labels"], result["scores"])
        ):
            x1, y1, x2, y2 = box
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            crop = img_np[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append({
                    "image": crop,
                    "box": [x1, y1, x2, y2],
                    "detect_label": label,
                    "detect_score": score,
                    "index": i
                })

        result["crops"] = crops
        return result

    def _draw_boxes(self, image, boxes, labels, scores):
        """Draw bounding boxes on image."""
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            color = self.CLASS_COLORS.get(label, (255, 255, 255))

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text = f"{label} {score:.2f}"
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale, thickness)
            cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                         color, -1)
            cv2.putText(image, text, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 255, 255), thickness)

        return image

    def _to_numpy(self, image_input):
        """Convert various image inputs to numpy array (BGR)."""
        try:
            if isinstance(image_input, np.ndarray):
                return image_input
            elif isinstance(image_input, str):
                if os.path.exists(image_input):
                    img = cv2.imread(image_input)
                    return img
            elif isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img
            elif isinstance(image_input, Image.Image):
                img_np = np.array(image_input)
                if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                return img_np
        except Exception as e:
            print(f"[YOLO] Error converting image: {e}")
        return None

    def _empty_result(self, image_input=None):
        """Return empty result."""
        return {
            "boxes": [], "labels": [], "scores": [],
            "annotated_image": None, "summary": {}, "total_cells": 0
        }

    def _mock_detect(self, img_np):
        """Mock detection for demo when model is not available."""
        h, w = img_np.shape[:2]
        # Generate some plausible mock detections
        np.random.seed(42)
        n_cells = np.random.randint(5, 15)
        boxes = []
        labels = []
        scores = []

        for _ in range(n_cells):
            cell_type = np.random.choice(["RBC", "WBC", "Platelets"],
                                          p=[0.7, 0.2, 0.1])
            size = np.random.randint(20, 60)
            cx = np.random.randint(size, w - size)
            cy = np.random.randint(size, h - size)
            boxes.append([cx - size, cy - size, cx + size, cy + size])
            labels.append(cell_type)
            scores.append(round(np.random.uniform(0.5, 0.95), 4))

        annotated = self._draw_boxes(img_np.copy(), boxes, labels, scores)
        summary = {}
        for label in labels:
            summary[label] = summary.get(label, 0) + 1

        return {
            "boxes": boxes, "labels": labels, "scores": scores,
            "annotated_image": annotated, "summary": summary,
            "total_cells": len(boxes)
        }


def numpy_to_base64(img_np, format="jpg"):
    """Convert numpy image to base64 string."""
    if img_np is None:
        return None
    if format == "jpg":
        _, buffer = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        _, buffer = cv2.imencode('.png', img_np)
    return base64.b64encode(buffer).decode('utf-8')
