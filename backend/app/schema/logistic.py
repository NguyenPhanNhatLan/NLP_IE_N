from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

class LogisticTrainParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_type: str = Field(
        default="logistic_re",
        examples=["logistic_re"],
        description="Model identifier for RE logistic regression",
    )

    vectorizer: str = Field(
        default="PhoBERT",
        examples=["PhoBERT"],
        description="Vectorizer used to extract features",
    )

    phobert_name: str = Field(
        default="vinai/phobert-base-v2",
        examples=["vinai/phobert-base-v2"],
    )

    device: str = Field(
        default="cpu",
        examples=["cpu", "cuda"],
    )

    max_len: int = Field(
        default=256,
        ge=8,
        le=1024,
        examples=[256],
    )

    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        examples=[0.2],
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        examples=[42],
        description="Random seed for train/test split",
    )
    solver: str = Field(
        default="lbfgs",
        examples=["lbfgs", "liblinear", "saga"],
    )

    max_iter: int = Field(
        default=2000,
        ge=100,
        examples=[2000],
    )

    class_weight: Optional[Any] = Field(
        default="balanced",
        examples=["balanced", None],
    )

    # ===== validators =====
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        return v

    
class TrainReLogisticRequest(BaseModel):
    experiment_name: str = Field(..., examples=["re_benchmark_phobert_v1"])
    run_name: str = Field(..., examples=["logistic"])

    params: LogisticTrainParams

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
class TrainReLogisticResponse(BaseModel):
    status: Literal["completed", "failed"]
    report: Dict[str, Any]
