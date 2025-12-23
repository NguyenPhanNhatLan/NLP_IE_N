from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from app.schema.rf import TrainReRfRequest, TrainReRfResponse
from training.models.rf.train_re import train_re_rf_and_report

router = APIRouter(prefix="/train/re/rf", tags=["train-re"])


@router.post(
    "",
    response_model=TrainReRfResponse,
    summary="Train RE RandomForest + log MLflow + (optional) register model",
)
def train_re_rf_endpoint(payload: TrainReRfRequest) -> TrainReRfResponse:
    try:
        report: Dict[str, Any] = train_re_rf_and_report(
            experiment_name=payload.experiment_name,
            run_name=payload.run_name,
            params=payload.params.model_dump(by_alias=True),  # ✅ giữ keys rf.* đúng như code train đang đọc
            tags=payload.tags,
            registry_name=payload.registry_name,
            do_register=payload.do_register,
            required_param_keys=payload.required_param_keys,
        )
        return TrainReRfResponse(status="completed", report=report)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "error": str(e),
                "hint": "Check MLFLOW_TRACKING_URI, artifact store (MinIO), and dataset builder/vectorizer.",
            },
        )
