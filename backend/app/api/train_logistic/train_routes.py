from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from app.schema.logistic import TrainReLogisticRequest, TrainReLogisticResponse
from training.models.logistic.train_re import train_re_logistic_and_report

router = APIRouter(prefix="/train/re/logistic", tags=["train-re"])


@router.post(
    "",
    response_model=TrainReLogisticResponse,
    summary="Train RE Logistic + log MLflow + (optional) register model",
)
def train_re_logistic_endpoint(payload: TrainReLogisticRequest) -> TrainReLogisticResponse:
    try:
        report: Dict[str, Any] = train_re_logistic_and_report(
            experiment_name=payload.experiment_name,
            run_name=payload.run_name,
            params=payload.params.model_dump(),  # Pydantic -> dict
            tags=payload.tags,
            registry_name=payload.registry_name,
            do_register=payload.do_register,
            required_param_keys=payload.required_param_keys,
        )
        return TrainReLogisticResponse(status="completed", report=report)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "error": str(e),
                "hint": "Check MLFLOW_TRACKING_URI, artifact store (MinIO), and dataset builder/vectorizer.",
            },
        )
