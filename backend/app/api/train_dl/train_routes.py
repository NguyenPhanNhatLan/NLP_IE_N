from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from app.schema.dl import TrainReMlpRequest, TrainReMlpResponse
from training.models.mlp.train_re import MLPTrainConfig, train_re_mlp_and_report

router = APIRouter(prefix="/train/re/mlp", tags=["train-re"])


@router.post(
    "",
    response_model=TrainReMlpResponse,
    summary="Train RE MLP (DL) + log MLflow (pyfunc) + (optional) register model",
)
def train_re_mlp_endpoint(payload: TrainReMlpRequest) -> TrainReMlpResponse:
    try:
        p = payload.params

        # build cfg dataclass từ pydantic params + request-level fields
        cfg = MLPTrainConfig(
            max_len=p.max_len,
            test_size=p.test_size,
            val_size=p.val_size,
            seed=p.seed,
            lr=p.lr,
            batch_size=p.batch_size,
            epochs=p.epochs,
            hidden_dim=p.hidden_dim,
            dropout=p.dropout,
            phobert_name=p.phobert_name,
            device=p.device,
            experiment_name=payload.experiment_name,
            run_name=payload.run_name,
            registry_name=payload.registry_name or "",
            do_register=payload.do_register,
            model_type=p.model_type,
        )

        report: Dict[str, Any] = train_re_mlp_and_report(
            cfg=cfg,
            tags=payload.tags,
            required_param_keys=payload.required_param_keys,
        )

        return TrainReMlpResponse(status="completed", report=report)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "error": str(e),
                "hint": "Check MLFLOW_TRACKING_URI, artifact store (MinIO), GPU/CPU device, and dataset builder/vectorizer.",
            },
        )
