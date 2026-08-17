"""Fixed registry and resolver for the trained Qwen blood-cell adapters.

Only stable public IDs leave the API. Local adapter paths are kept server-side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "qwen2.5-vl-3b-dora-r8-checkpoint-3315"


class ModelRegistryError(RuntimeError):
    """Raised when a registered model cannot be resolved safely."""


@dataclass(frozen=True)
class RegistryEntry:
    model_id: str
    display_name: str
    adapter_relative_path: str
    training_label: str

    @property
    def adapter_path(self) -> Path:
        return (PROJECT_ROOT / self.adapter_relative_path).resolve()


@dataclass(frozen=True)
class ResolvedModel:
    model_id: str
    display_name: str
    adapter_path: Path
    base_model: str
    architecture: str
    adapter_type: str
    training_label: str
    lora_r: int
    lora_alpha: float
    lora_dropout: float
    use_dora: bool
    adapter_config: dict[str, Any]

    def to_public_dict(self) -> dict[str, Any]:
        """Return model metadata without leaking a machine-local path."""
        return {
            "id": self.model_id,
            "name": self.display_name,
            "base_model": self.base_model,
            "architecture": self.architecture,
            "adapter_type": self.adapter_type,
            "training_method": self.training_label,
            "r": self.lora_r,
            "alpha": self.lora_alpha,
            "dropout": self.lora_dropout,
            "available": True,
        }


MODEL_REGISTRY: dict[str, RegistryEntry] = {
    "quoc-huy-qwen2-vl-2b-qlora-r8-final": RegistryEntry(
        model_id="quoc-huy-qwen2-vl-2b-qlora-r8-final",
        display_name="Quốc Huy — Qwen2-VL-2B + QLoRA r=8 (Final)",
        adapter_relative_path="FileTrainByQuocHuy_QLoRa_Qwen_2_2B/qwen_qlora/final",
        training_label="QLoRA",
    ),
    "quoc-huy-qwen2-vl-2b-qlora-r8-checkpoint-2000": RegistryEntry(
        model_id="quoc-huy-qwen2-vl-2b-qlora-r8-checkpoint-2000",
        display_name="Quốc Huy — Qwen2-VL-2B + QLoRA r=8 (Checkpoint 2000)",
        adapter_relative_path="FileTrainByQuocHuy_QLoRa_Qwen_2_2B/qwen_qlora/checkpoint-2000",
        training_label="QLoRA",
    ),
    "qwen2.5-vl-3b-lora-r16-checkpoint-5500": RegistryEntry(
        model_id="qwen2.5-vl-3b-lora-r16-checkpoint-5500",
        display_name="Qwen2.5-VL-3B + LoRA r=16 (Checkpoint 5500)",
        adapter_relative_path="custom_models/qwen_blood_cell/checkpoint-5500",
        training_label="LoRA",
    ),
    "qwen2.5-vl-3b-dora-r8-checkpoint-3315": RegistryEntry(
        model_id="qwen2.5-vl-3b-dora-r8-checkpoint-3315",
        display_name="Nhật Hào — Qwen2.5-VL-3B + DoRA r=8 (Checkpoint 3315 Mới Nhất)",
        adapter_relative_path="KetQuaMoiNhat_DoRa/23004023_PhanLoaiTeBaoMau/checkpoint-3315",
        training_label="DoRA",
    ),
    "qwen2.5-vl-3b-dora-r8-final": RegistryEntry(
        model_id="qwen2.5-vl-3b-dora-r8-final",
        display_name="Nhật Hào — Qwen2.5-VL-3B + DoRA r=8 (Final Model)",
        adapter_relative_path="KetQuaMoiNhat_DoRa/23004023_PhanLoaiTeBaoMau/final_model",
        training_label="DoRA",
    ),
}

MODEL_ALIASES: dict[str, str] = {
    "quoc-huy-qwen2-vl-2b-qlora-r8": "quoc-huy-qwen2-vl-2b-qlora-r8-final",
    "qwen2.5-vl-3b-dora-r8-23004023": "qwen2.5-vl-3b-dora-r8-final",
}


def _architecture_for_base(base_model: str) -> str:
    normalized = base_model.lower()
    if "qwen2.5-vl" in normalized:
        return "Qwen2_5_VLForConditionalGeneration"
    if "qwen2-vl" in normalized:
        return "Qwen2VLForConditionalGeneration"
    raise ModelRegistryError(
        f"Base model không được hỗ trợ: {base_model}. "
        "Registry chỉ chấp nhận Qwen2-VL hoặc Qwen2.5-VL."
    )


def _read_adapter_config(adapter_path: Path) -> dict[str, Any]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise ModelRegistryError(f"Không tìm thấy adapter_config.json cho model: {adapter_path.name}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(f"Không đọc được {config_path.name}: {exc}") from exc

    required = ("base_model_name_or_path", "r", "lora_alpha", "lora_dropout")
    missing = [key for key in required if key not in config]
    if missing:
        raise ModelRegistryError(
            f"adapter_config.json thiếu trường bắt buộc: {', '.join(missing)}"
        )
    return config


def _validate_adapter_weights(adapter_path: Path) -> None:
    weights_path = adapter_path / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise ModelRegistryError(f"Không tìm thấy adapter_model.safetensors: {adapter_path.name}")

    size = weights_path.stat().st_size
    header = b""
    if size <= 1024:
        try:
            header = weights_path.read_bytes()
        except OSError as exc:
            raise ModelRegistryError(f"Không đọc được adapter weights: {exc}") from exc

    is_lfs_pointer = header.startswith(b"version https://git-lfs.github.com/spec/v1")
    if is_lfs_pointer or size < 1024:
        relative = weights_path.relative_to(PROJECT_ROOT)
        raise ModelRegistryError(
            f"{relative} chỉ có {size} byte và là Git LFS pointer, không phải model weights. "
            "Hãy chạy `git lfs install` và `git lfs pull` trong thư mục dự án."
        )


def resolve_model(model_id: str | None) -> ResolvedModel:
    """Resolve a stable ID to validated server-local adapter metadata."""
    requested_id = model_id or DEFAULT_MODEL_ID
    requested_id = MODEL_ALIASES.get(requested_id, requested_id)
    entry = MODEL_REGISTRY.get(requested_id)
    if entry is None:
        allowed = ", ".join(MODEL_REGISTRY)
        raise ModelRegistryError(
            f"Model ID không hợp lệ: {requested_id!r}. Chỉ chấp nhận các ID: {allowed}. "
            "Đường dẫn máy không được phép gửi qua API."
        )

    adapter_path = entry.adapter_path
    if not adapter_path.is_dir():
        raise ModelRegistryError(f"Không tìm thấy thư mục adapter cho model {entry.display_name}")

    config = _read_adapter_config(adapter_path)
    _validate_adapter_weights(adapter_path)

    base_model = str(config["base_model_name_or_path"])
    use_dora = bool(config.get("use_dora", False))
    return ResolvedModel(
        model_id=entry.model_id,
        display_name=entry.display_name,
        adapter_path=adapter_path,
        base_model=base_model,
        architecture=_architecture_for_base(base_model),
        adapter_type="DoRA" if use_dora else "LoRA",
        training_label=entry.training_label,
        lora_r=int(config["r"]),
        lora_alpha=float(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        use_dora=use_dora,
        adapter_config=config,
    )


def public_model_list() -> list[dict[str, Any]]:
    """Return exactly the three registered models and their adapter metadata."""
    models: list[dict[str, Any]] = []
    for model_id, entry in MODEL_REGISTRY.items():
        try:
            models.append(resolve_model(model_id).to_public_dict())
        except ModelRegistryError as exc:
            models.append(
                {
                    "id": entry.model_id,
                    "name": entry.display_name,
                    "available": False,
                    "error": str(exc),
                }
            )
    return models
