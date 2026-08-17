"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class DetectionResult(BaseModel):
    boxes: List[List[int]]
    labels: List[str]
    scores: List[float]
    annotated_image_base64: Optional[str] = None
    summary: Dict[str, int]
    total_cells: int


class ClassificationResult(BaseModel):
    predicted_class: Optional[str]
    raw_output: str
    model_used: str
    inference_time_ms: float


class XAIResult(BaseModel):
    method: str
    method_info: Dict
    heatmap_base64: Optional[str]
    overlay_base64: Optional[str]
    original_base64: Optional[str]
    success: bool


class PipelineResult(BaseModel):
    detection: DetectionResult
    classifications: List[Dict]
    xai_results: Optional[Dict] = None


class CellTypeInfo(BaseModel):
    code: str
    name: str
    full_name: str
    description: str
    color: str
    sample_count: Optional[int] = 0
