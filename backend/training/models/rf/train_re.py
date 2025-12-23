# training/models/rf/train_re.py
# Run: python -u -m training.models.rf.train_re

import os
import mlflow
import numpy as np
import torch
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import mlflow.sklearn
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
from training.rules.post_rules import apply_post_rules
from training.evaluation.svm_metrics import full_evaluation

print(">>> START IMPORT train_re.py", flush=True)


PHOBERT_NAME = os.getenv("PHOBERT_NAME", "vinai/phobert-base-v2")

def extract_rf_params(params: Dict[str, Any]):
    return {k: v for k, v in params.items() if k.startswith("rf.")}

def train_rf(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    *,
    test_size: float = 0.2,
    rf_params: Dict[str, Any],
):
    seed = int(rf_params.get("rf.seed", 42))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=int(rf_params.get("rf.n_estimators", 500)),
        max_depth=rf_params.get("rf.max_depth", None),
        min_samples_leaf=int(rf_params.get("rf.min_samples_leaf", 1)),
        max_features=rf_params.get("rf.max_features", "sqrt"),
        class_weight=rf_params.get("rf.class_weight", "balanced"),
        random_state=seed,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, X_test, y_test, y_pred, idx_test

def run_train_rf(
    samples: list[dict],
    *,
    max_len: int = 256,
    rf_params: Dict[str, Any],
):
    device = "cpu"  # ✅ benchmark fair (same as LR/SVM/RF/MLP feature extraction)

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME, use_fast=False)
    model = AutoModel.from_pretrained(PHOBERT_NAME).to(device)
    model.eval()

    id2label, label2id = build_label_maps(samples)

    print("vectorizing", flush=True)
    with torch.no_grad():  # ✅
        X, y, idx = build_xy_ml(
            samples,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_len=max_len,
            label2id=label2id,
        )

    print("training", flush=True)
    clf, X_test, y_test, y_pred, idx_test = train_rf(
        X, y, idx, rf_params=rf_params,
    )
    return clf, tokenizer, model, (X_test, y_test, y_pred, id2label, label2id, idx_test)


def train_re_rf_and_report(
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
    print('started')
    t0 = time.time()

    try:
        end_active_run_safely()

        exp_id, tracking_uri = setup_experiment(experiment_name)

        report["run"]["experiment_id"] = exp_id
        report["run"]["tracking_uri"] = tracking_uri

        with start_run_strict(run_name=run_name, tags=tags) as run:
            report["run"]["run_id"] = run.info.run_id
            report["run"]["artifact_uri"] = mlflow.get_artifact_uri()

            rf_params = extract_rf_params(params)        
            mlflow.log_params(params)
            log_params_required(params, required_keys=required_param_keys)


            samples = build_re_datasets()
            report["data"]["dataset_size"] = len(samples)
            mlflow.log_metric("data.dataset_size", float(len(samples)))

            clf, tokenizer, model, pack = run_train_rf(samples, rf_params=rf_params)
            X_test, y_test, y_pred, id2label, label2id, idx_test = pack

            report["train"]["num_classes"] = len(id2label)
            mlflow.log_metric("train.num_classes", len(id2label))

            print('evaluating')
            test_samples = [samples[int(i)] for i in idx_test]

            results = full_evaluation(
                y_true=y_test,
                y_pred=y_pred,
                id2label=id2label,
                label2id=label2id,
                samples=test_samples,
                idx_test=None,
            )


            try:
                log_eval_results(results, metric_prefix="test", artifact_dir="evaluation")
            except Exception as e:
                print("Skip logging evaluation artifacts:", e)

            core = pick_core_eval_metrics(results, metric_prefix="test")
            report["eval"].update({
                "test_accuracy": core["accuracy"],
                "macro_f1": core["macro_f1"],
                "weighted_f1": core["weighted_f1"],
            })

            mlflow.sklearn.log_model(clf, artifact_path="model")
            labels_paths = log_label_maps(
                id2label=id2label,
                label2id=label2id,
                artifact_dir="model/labels",
            )

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



if __name__ == "__main__":

    params = {
        "model_type": "rf_re",
        "phobert_name": PHOBERT_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "rf.n_estimators": 500,
        "rf.max_depth": 25,
        "rf.min_samples_leaf": 16,
        "rf.max_features": "sqrt",
        "rf.class_weight": "balanced",
        "rf.seed": 81,
    }

    train_re_rf_and_report(
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "rf_re"),
        run_name="rf_v1",
        params=params,
        registry_name="re_rf_phobert",
    )

