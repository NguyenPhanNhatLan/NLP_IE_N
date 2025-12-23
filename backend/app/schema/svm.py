# app/api/train_re_svm/schemas.py
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class SVMTrainParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_type: str = Field(default="LinearSVC", examples=["LinearSVC"])
    phobert_name: str = Field(default="vinai/phobert-base-v2", examples=["vinai/phobert-base-v2"])
    device: str = Field(default="cpu", examples=["cpu", "cuda"])

    max_len: int = Field(default=256, ge=8, le=1024, examples=[256])
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0, examples=[0.2])
    seed: int = Field(default=42, ge=0, examples=[42])

    # sklearn: "balanced" hoặc None hoặc dict class->weight (nếu bạn hỗ trợ)
    class_weight: Optional[Any] = Field(default="balanced", examples=["balanced", None])

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        return v


class TrainReSvmRequest(BaseModel):
    experiment_name: str = Field(..., examples=["svm.re"])
    run_name: str = Field(..., examples=["svm_re"])

    params: SVMTrainParams

    tags: Optional[Dict[str, str]] = Field(
        default=None,
        examples=[{"task": "relation_extraction", "framework": "sklearn", "embedding": "phobert"}],
    )

    registry_name: Optional[str] = Field(default=None, examples=["re_svm_phobert"])
    do_register: bool = Field(default=True, examples=[True])

    # Nếu bạn muốn override required keys (ít dùng)
    required_param_keys: Optional[List[str]] = Field(
        default=None,
        examples=[["model_type", "phobert_name", "device"]],
    )
class TrainReSvmResponse(BaseModel):
    status: Literal["completed", "failed"]
    report: Dict[str, Any]
