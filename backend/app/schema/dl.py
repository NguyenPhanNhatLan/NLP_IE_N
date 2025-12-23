from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field, ConfigDict, field_validator


class MlpTrainParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_type: str = Field(default="mlp_re", examples=["mlp_re"])

    phobert_name: str = Field(
        default="vinai/phobert-base",
        examples=["vinai/phobert-base", "vinai/phobert-base-v2"],
    )

    device: str = Field(default="cuda", examples=["cpu", "cuda", "mps"])

    max_len: int = Field(default=256, ge=8, le=1024, examples=[256])
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0, examples=[0.2])
    val_size: float = Field(default=0.2, gt=0.0, lt=1.0, examples=[0.2])
    seed: int = Field(default=42, ge=0, examples=[42])

    lr: float = Field(default=1e-3, gt=0.0, examples=[1e-3])
    batch_size: int = Field(default=64, ge=1, le=4096, examples=[64])
    epochs: int = Field(default=30, ge=1, le=1000, examples=[30])
    hidden_dim: int = Field(default=256, ge=1, le=8192, examples=[256])
    dropout: float = Field(default=0.3, ge=0.0, le=0.9, examples=[0.3])

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        return v

    @field_validator("val_size")
    @classmethod
    def validate_val_size(cls, v: float) -> float:
        # val_size là tỷ lệ trên train split (như code train_mlp hiện tại)
        if v <= 0.0 or v >= 1.0:
            raise ValueError("val_size must be in (0,1)")
        return v

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        if v <= 0.0 or v >= 1.0:
            raise ValueError("test_size must be in (0,1)")
        return v


class TrainReMlpRequest(BaseModel):
    experiment_name: str = Field(default="mlp_re", examples=["re_benchmark_phobert_v1", "mlp_re"])
    run_name: str = Field(default="mlp_v1", examples=["mlp_v1"])

    params: MlpTrainParams = Field(default_factory=MlpTrainParams)

    tags: Optional[Dict[str, str]] = Field(default=None, examples=[{"stage": "benchmark"}])
    registry_name: Optional[str] = Field(default="re_mlp_phobert", examples=["re_mlp_phobert"])
    do_register: bool = Field(default=True, examples=[True])

    required_param_keys: Optional[List[str]] = Field(
        default=None,
        examples=[["model_type", "phobert_name", "device"]],
    )


class TrainReMlpResponse(BaseModel):
    status: str = Field(examples=["completed"])
    report: Dict[str, Any]
