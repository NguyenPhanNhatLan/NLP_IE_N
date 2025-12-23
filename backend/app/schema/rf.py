from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


class RfTrainParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_type: str = Field(default="rf_re", examples=["rf_re"])
    phobert_name: str = Field(default="vinai/phobert-base-v2", examples=["vinai/phobert-base", "vinai/phobert-base-v2"])
    device: str = Field(default="cpu", examples=["cpu", "cuda", "mps"])

    rf_n_estimators: int = Field(default=500, ge=1, examples=[500], alias="rf.n_estimators")
    rf_max_depth: Optional[int] = Field(default=25, ge=1, examples=[25, None], alias="rf.max_depth")
    rf_min_samples_leaf: int = Field(default=16, ge=1, examples=[16], alias="rf.min_samples_leaf")
    rf_max_features: Any = Field(default="sqrt", examples=["sqrt", "log2", None, 0.5], alias="rf.max_features")
    rf_class_weight: Any = Field(default="balanced", examples=["balanced", None], alias="rf.class_weight")
    rf_seed: int = Field(default=81, ge=0, examples=[81], alias="rf.seed")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        return v


class TrainReRfRequest(BaseModel):
    experiment_name: str = Field(default="rf_re", examples=["re_benchmark_phobert_v1", "rf_re"])
    run_name: str = Field(default="rf_v1", examples=["rf_v1"])

    params: RfTrainParams = Field(default_factory=RfTrainParams)

    tags: Optional[Dict[str, str]] = Field(default=None, examples=[{"stage": "benchmark"}])
    registry_name: Optional[str] = Field(default="re_rf_phobert", examples=["re_rf_phobert"])
    do_register: bool = Field(default=True, examples=[True])

    required_param_keys: Optional[list[str]] = Field(
        default=None,
        examples=[["model_type", "phobert_name", "device"]],
    )


class TrainReRfResponse(BaseModel):
    status: str = Field(examples=["completed"])
    report: Dict[str, Any]