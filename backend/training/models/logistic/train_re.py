# Run: py -m training.models.logistic.train_re

from dotenv import load_dotenv
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline
load_dotenv()

import os
import time
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
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


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    max_iter: int = 2000,
    solver: str = "lbfgs",
):
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # giống SVM: compute class_weight từ y_train
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {c: w for c, w in zip(classes, cw)}

    # ✅ giống mẫu SVM nhưng thêm scaler để logistic ổn định hơn
    clf = make_pipeline(
        StandardScaler(with_mean=False),  # with_mean=False an toàn nếu X là sparse
        LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=None,         # lbfgs không dùng n_jobs, giữ None
        )
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, X_test, y_test, y_pred, idx_test

def run_train_logistic(
    samples: list[dict],
    *,
    phobert_name: str,
    max_len: int = 256,
    seed: int = 42,
    test_size: float = 0.2,
    max_iter: int = 2000,
    solver: str = "lbfgs",
):
    device = "cpu"  # bạn muốn CPU thì giữ

    tokenizer = AutoTokenizer.from_pretrained(phobert_name, use_fast=False)
    model = AutoModel.from_pretrained(phobert_name).to(device)
    model.eval()

    id2label, label2id = build_label_maps(samples)

    X, y, idx = build_xy_ml(
        samples,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        label2id=label2id,
    )

    clf, X_test, y_test, y_pred, idx_test = train_logistic(
        X, y, idx,
        test_size=test_size,
        seed=seed,
        max_iter=max_iter,
        solver=solver,
    )

    return clf, tokenizer, model, (X_test, y_test, y_pred, id2label, label2id, idx_test)
def train_re_logistic_and_report(
    
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

        # 2) MLflow env + experiment
        exp_id, tracking_uri = setup_experiment(experiment_name)
        report["run"]["experiment_id"] = exp_id
        report["run"]["tracking_uri"] = tracking_uri


        with start_run_strict(run_name=run_name, tags=tags) as run:
            report["run"]["run_id"] = run.info.run_id
            report["run"]["artifact_uri"] = mlflow.get_artifact_uri()


            mlflow.log_params(params)
            log_params_required(params, required_keys=required_param_keys)

            samples = build_re_datasets()
            report["data"]["dataset_size"] = len(samples)
            mlflow.log_metric("data.dataset_size", float(len(samples)))

            clf, tokenizer, model, pack = run_train_logistic(
                samples,
                phobert_name=params["phobert_name"],
                max_len=params["max_len"],
                test_size=params["test_size"],
                seed=params["random_seed"],
            )

            X_test, y_test, y_pred, id2label, label2id, idx_test = pack

            report["train"]["num_classes"] = len(id2label)
            mlflow.log_metric("train.num_classes", len(id2label))


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
                log_eval_results(
                    results,
                    metric_prefix="test",
                    artifact_dir="evaluation",
                )
            except Exception as e:
                print("⚠️ Skip logging evaluation artifacts:", e)
                report["artifacts"]["evaluation_error"] = str(e)

            core = pick_core_eval_metrics(
                results,
                metric_prefix="test",
            )

            report["eval"].update({
                "test_accuracy": core["accuracy"],
                "macro_f1": core["macro_f1"],
                "weighted_f1": core["weighted_f1"],
            })

            try:
                mlflow.sklearn.log_model(clf, artifact_path="model")
                report["artifacts"]["model"] = "model"
            except Exception as e:
                print("⚠️ Skip logging model artifact:", e)
                report["artifacts"]["model_error"] = str(e)

            try:
                labels_paths = log_label_maps(
                    id2label=id2label,
                    label2id=label2id,
                    artifact_dir="model/labels",
                )
                report["artifacts"]["labels"] = labels_paths
            except Exception as e:
                print("⚠️ Skip logging label maps:", e)
                report["artifacts"]["labels_error"] = str(e)

            # 9) Registry (🔥 FIX QUAN TRỌNG)
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
                    print("⚠️ Register model failed:", e)
                    report["registry"]["error"] = str(e)

            # 10) Finish
            report["status"] = "completed"
            report["train"]["duration_sec"] = round(
                time.time() - t0, 3
            )
            mlflow.log_metric(
                "train.duration_sec",
                report["train"]["duration_sec"],
            )

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
        "model_type": "logistic_re",
        "vectorizer": "PhoBERT",
        "phobert_name": PHOBERT_NAME,
        "device": "cpu",
        "max_len": 256,
        "random_seed": 42,
        "test_size": 0.2,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": "balanced",
    }

    train_re_logistic_and_report(
        experiment_name="re_logistic_phobert",
        run_name="logistic_run_1",
        params=params,
        registry_name="re_logistic",
        do_register=True,
    )
