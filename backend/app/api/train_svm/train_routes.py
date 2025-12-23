# app/api/train_re_svm/routes.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Any, Dict

from app.schema.svm import TrainReSvmRequest, TrainReSvmResponse
from training.models.svm.train_re import train_re_svm_and_report

router = APIRouter(prefix="/train/re/svm", tags=["train-re"])



@router.post(
    "",
    response_model=TrainReSvmResponse,
    summary="Train RE SVM + log MLflow + (optional) register model",
)
def train_re_svm_endpoint(payload: TrainReSvmRequest) -> TrainReSvmResponse:
    try:
        report: Dict[str, Any] = train_re_svm_and_report(
            experiment_name=payload.experiment_name,
            run_name=payload.run_name,
            params=payload.params.model_dump(),   # convert Pydantic -> dict
            tags=payload.tags,
            registry_name=payload.registry_name,
            do_register=payload.do_register,
            required_param_keys=payload.required_param_keys,
        )
        return TrainReSvmResponse(status="completed", report=report)

    except Exception as e:
        # Bạn có thể log thêm server-side ở đây
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "error": str(e),
                "hint": "Check MLflow env (MLFLOW_TRACKING_URI), MinIO/Artifact store, and dataset builder.",
            },
        )
