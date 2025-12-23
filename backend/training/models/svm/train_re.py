# Run: python -m training.models.svm.train_re
import os
import time
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import torch

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, AutoTokenizer


from training.mlflow.schema import init_report
from training.mlflow.run_utils import (
    end_active_run_safely,
    setup_experiment,
    start_run_strict,
)
from training.mlflow.utils_log import (
    log_params_required,
    log_label_maps,
    log_eval_results,
    pick_core_eval_metrics,
    print_run_summary,
)
from training.mlflow.registry import register_and_promote

from training.features.build_data.build_re_dataset import build_re_datasets
from training.features.vectorize.utils import build_label_maps
from training.features.vectorize.re_vectorize import build_xy_ml
from training.evaluation.svm_metrics import full_evaluation


PHOBERT_NAME = os.getenv("PHOBERT_NAME", "vinai/phobert-base-v2")

def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 42,
):
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {c: w for c, w in zip(classes, cw)}

    clf = LinearSVC(class_weight=class_weight)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return clf, X_test, y_test, y_pred, idx_test


def run_train_svm(
    samples: list[dict],
    *,
    max_len: int = 256,
    seed: int = 42,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME, use_fast=False)
    model = AutoModel.from_pretrained(PHOBERT_NAME).to(device)
    model.eval()  # freeze PhoBERT

    id2label, label2id = build_label_maps(samples)

    X, y, idx = build_xy_ml(
        samples,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        label2id=label2id,
    )

    clf, X_test, y_test, y_pred, idx_test = train_svm(
        X, y, idx, seed=seed
    )

    return clf, tokenizer, model, (X_test, y_test, y_pred, id2label, label2id, idx_test)


# =========================================================
# Full pipeline + MLflow
# =========================================================

def train_re_svm_and_report(
    *,
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    tags: Optional[Dict[str, str]] = None,
    registry_name: Optional[str] = None,
    do_register: bool = True,
    required_param_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:

    required_param_keys = required_param_keys or [
        "model_type",
        "phobert_name",
        "device",
    ]

    report = init_report(
        experiment_name=experiment_name,
        run_name=run_name,
        params=params,
        tags=tags,
        registry_name=registry_name,
        do_register=do_register,
    )

    t0 = time.time()

    try:
        # 1) Cleanup
        end_active_run_safely()

        exp_id, tracking_uri = setup_experiment(experiment_name)

        report["run"]["experiment_id"] = exp_id
        report["run"]["tracking_uri"] = tracking_uri

        # 3) Start run
        with start_run_strict(run_name=run_name, tags=tags) as run:
            report["run"]["run_id"] = run.info.run_id
            report["run"]["artifact_uri"] = mlflow.get_artifact_uri()

            # 4) Params
            log_params_required(params, required_keys=required_param_keys)

            # 5) Data
            samples = build_re_datasets()
            report["data"]["dataset_size"] = len(samples)
            mlflow.log_metric("data.dataset_size", float(len(samples)))

            # 6) Train
            clf, tokenizer, model, pack = run_train_svm(samples)
            X_test, y_test, y_pred, id2label, label2id, idx_test = pack

            report["train"]["num_classes"] = len(id2label)
            mlflow.log_metric("train.num_classes", len(id2label))

            # 7) Eval
            results = full_evaluation(
                y_true=y_test,
                y_pred=y_pred,
                id2label=id2label,
                label2id=label2id,
                samples=samples,
                idx_test=idx_test,
            )

            log_eval_results(results, metric_prefix="test", artifact_dir="evaluation")

            core = pick_core_eval_metrics(results, metric_prefix="test")
            report["eval"].update({
                "test_accuracy": core["accuracy"],
                "macro_f1": core["macro_f1"],
                "weighted_f1": core["weighted_f1"],
            })

            # 8) Artifacts
            mlflow.sklearn.log_model(clf, artifact_path="model")
            labels_paths = log_label_maps(
                id2label=id2label,
                label2id=label2id,
                artifact_dir="model/labels",
            )
            print(id2label)

            report["artifacts"] = {
                "model": "model",
                "labels": labels_paths,
                "evaluation": "evaluation",
            }

            # 9) Registry
            if do_register and registry_name:
                report["registry"]["attempted"] = True
                try:
                    reg_info = register_and_promote(
                        run_id=run.info.run_id,
                        name=registry_name,
                        artifact_path="model",
                    )
                    report["registry"]["registered"] = True
                    report["registry"]["details"] = reg_info
                except Exception as e:
                    report["registry"]["error"] = str(e)

            # 10) Finish
            report["status"] = "completed"
            report["train"]["duration_sec"] = round(time.time() - t0, 3)
            mlflow.log_metric("train.duration_sec", report["train"]["duration_sec"])

            print_run_summary(report)
            return report

    except Exception as e:
        report["status"] = "failed"
        report["error"] = str(e)

        if mlflow.active_run():
            mlflow.log_param("error", str(e))
            end_active_run_safely(status="FAILED")

        raise

    finally:
        end_active_run_safely()
