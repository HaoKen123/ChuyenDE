"""Real Qwen-VL inference for the 12-class blood-cell task."""

from __future__ import annotations

import gc
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image


CELL_TYPES = {
    "BA": {"name": "Basophil", "full_name": "Basophil", "description": "Bạch cầu ưa kiềm - Tham gia phản ứng dị ứng và viêm. Chứa hạt lớn bắt màu xanh tím đậm.", "color": "#6C5CE7"},
    "BNE": {"name": "Band Neutrophil", "full_name": "Band Neutrophil", "description": "Bạch cầu trung tính dạng băng - Dạng chưa trưởng thành của neutrophil. Nhân hình chữ C hoặc chữ S.", "color": "#0984E3"},
    "EO": {"name": "Eosinophil", "full_name": "Eosinophil", "description": "Bạch cầu ưa axit - Chống ký sinh trùng và dị ứng. Chứa hạt lớn bắt màu cam/đỏ.", "color": "#E17055"},
    "ERB": {"name": "Erythroblast", "full_name": "Erythroblast", "description": "Nguyên hồng cầu - Tiền thân của hồng cầu trong tủy xương. Có nhân tròn đậm.", "color": "#D63031"},
    "LY": {"name": "Lymphocyte", "full_name": "Lymphocyte", "description": "Tế bào lympho - Quan trọng trong miễn dịch thu được (T cells, B cells). Nhân tròn lớn, ít bào tương.", "color": "#00B894"},
    "MMY": {"name": "Metamyelocyte", "full_name": "Metamyelocyte", "description": "Hậu tủy bào - Giai đoạn trung gian trong biệt hóa neutrophil. Nhân hình thận.", "color": "#FDCB6E"},
    "MO": {"name": "Monocyte", "full_name": "Monocyte", "description": "Bạch cầu đơn nhân - Tế bào lớn nhất trong máu ngoại vi. Nhân hình thận hoặc gấp lại.", "color": "#E84393"},
    "MY": {"name": "Myelocyte", "full_name": "Myelocyte", "description": "Tủy bào - Giai đoạn sớm trong biệt hóa granulocyte. Nhân tròn lệch tâm.", "color": "#00CEC9"},
    "MYO": {"name": "Myeloblast", "full_name": "Myeloblast", "description": "Nguyên tủy bào - Tế bào gốc dòng tủy. Nhân lớn, tỷ lệ nhân/bào tương cao.", "color": "#A29BFE"},
    "PLT": {"name": "Platelet", "full_name": "Platelet", "description": "Tiểu cầu - Mảnh tế bào nhỏ, tham gia đông máu. Kích thước rất nhỏ, không có nhân.", "color": "#FD79A8"},
    "PMY": {"name": "Promyelocyte", "full_name": "Promyelocyte", "description": "Tiền tủy bào - Giai đoạn sau myeloblast. Chứa hạt azurophilic lớn đặc trưng.", "color": "#FAB1A0"},
    "SNE": {"name": "Segmented Neutrophil", "full_name": "Segmented Neutrophil", "description": "Bạch cầu trung tính phân thùy - Dạng trưởng thành. Nhân chia 2-5 thùy nối bằng sợi mảnh.", "color": "#74B9FF"},
}

CELL_LABELS = list(CELL_TYPES)


class QWenClassifier:
    """Load one PEFT adapter and run genuine Qwen-VL generation."""

    BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        model_name: str | Path | None = None,
        device: str | None = None,
        compression: str = "4bit",
        *,
        model_id: str | None = None,
        display_name: str | None = None,
        expected_base_model: str | None = None,
    ):
        self.model_name = str(model_name or self.BASE_MODEL_NAME)
        self.model_id = model_id or self.model_name
        self.display_name = display_name or self.model_id
        self.device = device
        self.compression = compression
        self.expected_base_model = expected_base_model
        self.model = None
        self.processor = None
        self.adapter_config: dict | None = None
        self.base_model_name: str | None = None
        self.architecture: str | None = None
        try:
            self._load_model()
        except Exception:
            # Ensure a failed/partial load cannot leave stale CUDA allocations.
            self.unload()
            raise

    @staticmethod
    def _model_class_for_base(base_model: str):
        """Select the exact Transformers architecture recorded by the adapter."""
        normalized = base_model.lower()
        if "qwen2.5-vl" in normalized:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        if "qwen2-vl" in normalized:
            from transformers import Qwen2VLForConditionalGeneration

            return Qwen2VLForConditionalGeneration
        raise ValueError(
            f"Base model không được hỗ trợ: {base_model}. Cần Qwen2-VL hoặc Qwen2.5-VL."
        )

    def _get_quantization_kwargs(self) -> dict:
        import torch
        from transformers import BitsAndBytesConfig

        allowed = {"4bit", "8bit", "16bit", "full"}
        if self.compression not in allowed:
            raise ValueError(
                f"Chế độ nén {self.compression!r} không được hỗ trợ. Chọn: {', '.join(sorted(allowed))}."
            )
        if self.device != "cuda" and self.compression in {"4bit", "8bit"}:
            raise RuntimeError(
                f"{self.compression} cần CUDA/bitsandbytes. Hãy chọn GPU (CUDA), "
                "hoặc chọn Full 32-bit khi chạy CPU."
            )
        if self.compression == "4bit":
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                "torch_dtype": torch.float16,
            }
        if self.compression == "8bit":
            return {
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                "torch_dtype": torch.float16,
            }
        if self.compression == "16bit":
            return {"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        return {"torch_dtype": torch.float32}

    def _load_model(self) -> None:
        import torch

        if self.device in (None, "auto"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Thiết bị phải là 'cpu', 'cuda' hoặc 'auto'.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Đã chọn CUDA nhưng PyTorch không phát hiện GPU CUDA khả dụng.")

        model_path = Path(self.model_name)
        if model_path.is_dir() and (model_path / "adapter_config.json").is_file():
            self._load_lora_adapter(model_path)
            return
        if model_path.is_file() and model_path.suffix.lower() in {".pth", ".pt", ".bin"}:
            raise ValueError(
                "Không thể nạp state_dict rời từ model lượng tử hóa. Hãy dùng thư mục PEFT "
                "chứa adapter_config.json và adapter_model.safetensors."
            )

        # Direct programmatic use is supported; API traffic only reaches registered adapters.
        self.base_model_name = self.model_name
        model_class = self._model_class_for_base(self.base_model_name)
        self.architecture = model_class.__name__
        kwargs = self._get_quantization_kwargs()
        self.model = model_class.from_pretrained(
            self.base_model_name,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.base_model_name)
        self.model.eval()

    def _load_lora_adapter(self, adapter_dir: Path) -> None:
        config_path = adapter_dir / "adapter_config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Không đọc được adapter_config.json: {exc}") from exc

        base_model = config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("adapter_config.json thiếu base_model_name_or_path.")
        if self.expected_base_model and base_model != self.expected_base_model:
            raise ValueError(
                f"Base model trong adapter_config.json là {base_model}, "
                f"không khớp registry ({self.expected_base_model})."
            )

        weights_path = adapter_dir / "adapter_model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {weights_path}")
        if weights_path.stat().st_size < 1024:
            raise RuntimeError(
                f"{weights_path.name} chỉ có {weights_path.stat().st_size} byte; đây có thể là Git LFS pointer. "
                "Hãy chạy `git lfs install` và `git lfs pull`."
            )

        self.adapter_config = config
        self.base_model_name = str(base_model)
        model_class = self._model_class_for_base(self.base_model_name)
        self.architecture = model_class.__name__
        kwargs = self._get_quantization_kwargs()
        load_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "low_cpu_mem_usage": True,
            **kwargs,
        }
        if self.device == "cuda":
            load_kwargs["offload_folder"] = str(adapter_dir.parent / ".offload")

        base = model_class.from_pretrained(self.base_model_name, **load_kwargs)
        from peft import PeftModel
        from transformers import AutoProcessor

        self.model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.base_model_name)

    def classify(self, image_input, top_k: int = 5) -> dict:
        """Generate a real prediction and calculate mathematical softmax confidence."""
        pil_image = self._to_pil(image_input)
        if pil_image is None:
            raise ValueError("Không thể đọc ảnh đầu vào.")
        if self.model is None or self.processor is None:
            raise RuntimeError("Model chưa được nạp; không thể chạy inference.")

        started = time.perf_counter()
        raw_output, predicted_class, confidence, top_probs = self._run_classification(pil_image, top_k=top_k)
        self._synchronize_cuda()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percent": f"{confidence * 100:.1f}%",
            "top_probabilities": top_probs,
            "raw_output": raw_output,
            "model_used": self.display_name,
            "inference_time_ms": elapsed_ms,
        }

    def _run_classification(self, pil_image: Image.Image, top_k: int = 5) -> tuple[str, str | None, float, list[dict]]:
        import torch

        cell_names = ", ".join(f"{code}={CELL_TYPES[code]['name']}" for code in CELL_LABELS)
        prompt = (
            "This is a microscopic image of a blood cell. "
            f"Classify it into exactly one of these categories: {', '.join(CELL_LABELS)}. "
            f"The abbreviations mean: {cell_names}. Reply with ONLY the abbreviation code."
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_image], padding=False, return_tensors="pt")
        target_device = self._input_device()
        inputs = {
            key: value.to(target_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        self._synchronize_cuda()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], outputs.sequences)
        ]
        raw_output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        predicted_class = self._parse_prediction(raw_output)

        # Calculate exact softmax probability over the 12 candidate classes from output scores
        class_scores = []
        if outputs.scores and len(outputs.scores) > 0:
            first_step_logits = outputs.scores[0][0]  # shape: (vocab_size,)
            log_probs = torch.log_softmax(first_step_logits, dim=-1)

            for code in CELL_LABELS:
                token_ids = self.processor.tokenizer.encode(code, add_special_tokens=False)
                if not token_ids:
                    token_ids = self.processor.tokenizer.encode(" " + code, add_special_tokens=False)

                # Accumulate step log-probabilities
                total_log_prob = 0.0
                for step_idx, tid in enumerate(token_ids):
                    if step_idx < len(outputs.scores):
                        step_logits = outputs.scores[step_idx][0]
                        step_log_probs = torch.log_softmax(step_logits, dim=-1)
                        total_log_prob += float(step_log_probs[tid].item())
                    else:
                        break
                class_scores.append(total_log_prob)

            tensor_scores = torch.tensor(class_scores, dtype=torch.float32)
            normalized_probs = torch.softmax(tensor_scores, dim=-1).tolist()
        else:
            normalized_probs = [1.0 / len(CELL_LABELS)] * len(CELL_LABELS)

        prob_list = []
        for code, prob in zip(CELL_LABELS, normalized_probs):
            prob_list.append({
                "class": code,
                "name": CELL_TYPES[code]["name"],
                "full_name": CELL_TYPES[code]["full_name"],
                "color": CELL_TYPES[code]["color"],
                "probability": round(float(prob), 4),
                "percentage": f"{prob * 100:.1f}%",
            })
        prob_list.sort(key=lambda x: x["probability"], reverse=True)

        confidence = 0.0
        if predicted_class:
            for item in prob_list:
                if item["class"] == predicted_class:
                    confidence = item["probability"]
                    break
        elif prob_list:
            predicted_class = prob_list[0]["class"]
            confidence = prob_list[0]["probability"]

        return raw_output, predicted_class, confidence, prob_list[:top_k]

    @staticmethod
    def _parse_prediction(raw_output: str) -> str | None:
        normalized = raw_output.strip().upper()
        if normalized in CELL_LABELS:
            return normalized
        matches = {
            label
            for label in CELL_LABELS
            if re.search(rf"(?<![A-Z]){re.escape(label)}(?![A-Z])", normalized)
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def _input_device(self):
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return self.device

    def _synchronize_cuda(self) -> None:
        if self.device == "cuda":
            import torch

            torch.cuda.synchronize()

    def unload(self) -> None:
        """Release this model before another registry entry is loaded."""
        model = self.model
        self.model = None
        self.processor = None
        self.adapter_config = None
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass

    def classify_batch(self, images, top_k: int = 3) -> list[dict]:
        return [self.classify(image, top_k=top_k) for image in images]

    def get_vision_encoder(self):
        if self.model is None:
            return None
        accessors = (
            lambda model: model.visual,
            lambda model: model.model.visual,
            lambda model: model.base_model.model.visual,
            lambda model: model.model.model.visual,
        )
        for accessor in accessors:
            try:
                encoder = accessor(self.model)
                if encoder is not None:
                    return encoder
            except (AttributeError, TypeError):
                continue
        return None

    @staticmethod
    def _to_pil(image_input) -> Image.Image | None:
        try:
            if isinstance(image_input, Image.Image):
                return image_input.convert("RGB")
            if isinstance(image_input, np.ndarray):
                import cv2

                if image_input.ndim == 3 and image_input.shape[2] == 3:
                    return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
                return Image.fromarray(image_input).convert("RGB")
            if isinstance(image_input, (str, os.PathLike)) and os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
            if isinstance(image_input, bytes):
                return Image.open(io.BytesIO(image_input)).convert("RGB")
        except (OSError, ValueError):
            return None
        return None
