from dataclasses import asdict
import json
import os
import time
from typing import Any, Dict, List, Optional
import mlflow
from sklearn.metrics import classification_report
from training.features.build_data.build_ner_dataset import build_ner_dataset
from training.mlflow.registry import register_and_promote
from training.mlflow.run_utils import end_active_run_safely, setup_experiment, start_run_strict
from training.mlflow.schema import init_report
from training.mlflow.utils_log import log_eval_results, log_params_required, print_run_summary
from training.models.ner.config import CRFConfig, NerCrfPyFunc
from training.models.ner.data_io import to_xy
from training.models.ner.features import build_features
from training.models.ner.model import train_crf, save_model
from training.models.ner.evaluate import evaluate_seqeval, get_report_str
from seqeval.metrics import classification_report

def to_jsonable(x):
    """Convert numpy/scalar types into pure Python types recursively."""
    # numpy scalar?
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            pass

    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x

def train_ner_crf_and_report(
    *,
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    tags: Optional[Dict[str, str]] = None,
    registry_name: Optional[str] = None,
    do_register: bool = True,
    required_param_keys: Optional[List[str]] = None,
    model_artifact_path: str = "model",
):
    required_param_keys = required_param_keys or [
        "model_type",
        "window",
        "c1",
        "c2",
        "max_iterations",
    ]

    # ---- FIX 1: auto inject model_type để khỏi Missing required params
    params = dict(params or {})
    params.setdefault("model_type", "CRF")

    report = init_report(
        experiment_name=experiment_name,
        run_name=run_name,
        params=params,
        tags=tags,
        registry_name=registry_name,
        do_register=do_register,
    )

    t0 = time.time()
    artifact_dir = os.path.join(os.getcwd(), "artifacts", "ner_crf")
    os.makedirs(artifact_dir, exist_ok=True)

    try:
        # 1) Cleanup
        end_active_run_safely()

        # 2) Experiment
        exp_id, tracking_uri = setup_experiment(experiment_name)
        report["run"]["experiment_id"] = exp_id
        report["run"]["tracking_uri"] = tracking_uri

        # 3) Start run
        with start_run_strict(run_name=run_name, tags=tags) as run:
            report["run"]["run_id"] = run.info.run_id
            report["run"]["artifact_uri"] = mlflow.get_artifact_uri()

            # 4) Params (đúng theo utils_mlflow)
            log_params_required(params, required_keys=required_param_keys)

            # 5) Data
            train_data, valid_data = build_ner_dataset()
            report["data"]["train_size"] = len(train_data)
            report["data"]["valid_size"] = len(valid_data)
            mlflow.log_metric("data.train_size", float(len(train_data)))
            mlflow.log_metric("data.valid_size", float(len(valid_data)))

            # 6) Train config
            cfg = CRFConfig(
                window=int(params["window"]),
                c1=float(params["c1"]),
                c2=float(params["c2"]),
                max_iterations=int(params["max_iterations"]),
            )

            # 7) Build X/y
            X_train_tokens, y_train = to_xy(train_data)
            X_valid_tokens, y_valid = to_xy(valid_data)

            X_train_feats = build_features(X_train_tokens, window=cfg.window)
            X_valid_feats = build_features(X_valid_tokens, window=cfg.window)

            # 8) Train
            crf = train_crf(X_train_feats, y_train, cfg)

            # 9) Eval (seqeval)
            y_pred = crf.predict(X_valid_feats)
            base_metrics = evaluate_seqeval(y_valid, y_pred)  # precision/recall/f1/accuracy

            # 9.1) chuẩn hóa "results" cho log_eval_results()
            results = {
                "metrics": {
                    "precision": float(base_metrics.get("precision", 0.0)),
                    "recall": float(base_metrics.get("recall", 0.0)),
                    "f1": float(base_metrics.get("f1", 0.0)),
                    "accuracy": float(base_metrics.get("accuracy", 0.0)),
                },
                "errors": [],
            }

            # 9.2) log theo utils_mlflow
            log_eval_results(results, metric_prefix="valid", artifact_dir="evaluation")

            # 9.3) cập nhật report
            report["eval"]["valid_metrics"] = base_metrics

            # mapping để print_run_summary (cũ) đọc được
            report["eval"]["test_accuracy"] = float(base_metrics.get("accuracy", 0.0))
            report["eval"]["macro_f1"] = float(base_metrics.get("f1", 0.0))
            report["eval"]["weighted_f1"] = float(base_metrics.get("f1", 0.0))

            # 10) Log report txt/json
            rep_txt = get_report_str(y_valid, y_pred)  # dùng hàm bạn đưa
            rep_path = os.path.join(artifact_dir, "seqeval_report.txt")
            with open(rep_path, "w", encoding="utf-8") as f:
                f.write(rep_txt)
            mlflow.log_artifact(rep_path, artifact_path="evaluation")

            # IMPORTANT: output_dict=True phải là seqeval.metrics.classification_report
            rep_dict = classification_report(y_valid, y_pred, output_dict=True)
            mlflow.log_dict(
                to_jsonable(rep_dict),
                "evaluation/seqeval_report.json"
            )

            # 11) Save joblib + config
            model_joblib = os.path.join(artifact_dir, "ner_crf.joblib")
            save_model(crf, model_joblib)

            cfg_path = os.path.join(artifact_dir, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_type": "CRF",
                        "window": cfg.window,
                        "c1": cfg.c1,
                        "c2": cfg.c2,
                        "max_iterations": cfg.max_iterations,
                        "algorithm": cfg.algorithm,
                        "all_possible_transitions": cfg.all_possible_transitions,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # 12) Log MLflow Model (pyfunc)
            pyfunc_model = NerCrfPyFunc(window=cfg.window)
            mlflow.pyfunc.log_model(
                artifact_path=model_artifact_path,
                python_model=pyfunc_model,
                artifacts={"crf_joblib": model_joblib},
                pip_requirements=[
                    "mlflow",
                    "joblib",
                    "sklearn-crfsuite",
                    "seqeval",
                    "underthesea",
                ],
            )

            # debug artifacts
            mlflow.log_artifact(model_joblib, artifact_path=f"{model_artifact_path}/debug")
            mlflow.log_artifact(cfg_path, artifact_path=f"{model_artifact_path}/debug")

            report["artifacts"] = {
                "mlflow_model": model_artifact_path,
                "evaluation": "evaluation",
                "debug_joblib": f"{model_artifact_path}/debug/ner_crf.joblib",
                "debug_config": f"{model_artifact_path}/debug/config.json",
                "seqeval_txt": "evaluation/seqeval_report.txt",
                "seqeval_json": "evaluation/seqeval_report.json",
            }

            # 13) Registry
            if do_register and registry_name:
                report["registry"]["attempted"] = True
                report["registry"]["name"] = registry_name
                try:
                    reg_info = register_and_promote(
                        run_id=run.info.run_id,
                        name=registry_name,
                        artifact_path=model_artifact_path,
                    )
                    report["registry"]["registered"] = True
                    report["registry"]["details"] = reg_info
                    if isinstance(reg_info, dict):
                        report["registry"]["version"] = reg_info.get("version")
                        report["registry"]["stage"] = reg_info.get("stage")
                except Exception as e:
                    report["registry"]["error"] = str(e)

            # 14) Finish
            report["status"] = "completed"
            report["train"]["duration_sec"] = round(time.time() - t0, 3)
            mlflow.log_metric("train.duration_sec", report["train"]["duration_sec"])

            print_run_summary(report)
            return report

    except Exception as e:
        report["status"] = "failed"
        report["error"] = str(e)
        if mlflow.active_run():
            try:
                mlflow.log_param("error", str(e))
            except Exception:
                pass
            end_active_run_safely(status="FAILED")
        raise

    finally:
        end_active_run_safely()
        
           
# def main():
#     params = {
#         "model_type": "CRF",
#         "window": 2,
#         "c1": 0.1,
#         "c2": 0.1,
#         "max_iterations": 200,
#     }

#     tags = {
#         "task": "ner",
#         "framework": "sklearn-crfsuite",
#         "model_type": "CRF",
#     }

#     report = train_ner_crf_and_report(
#         experiment_name="crf.ner",
#         run_name="crf_ner",
#         params=params,
#         tags=tags,
#         registry_name="ner_crf",   
#         do_register=True,
#     )

#     return report

# cfg = CRFConfig()
# params = {
#     "model_type": "CRF",
#     **asdict(cfg),
# }
# train_ner_crf_and_report(experiment_name="ner", params=params, run_name="ner-1", registry_name="ner-model")