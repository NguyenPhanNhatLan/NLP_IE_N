from typing import Dict, Optional, Tuple
import mlflow

# quản lý các lần chạy và tên gọi 
def end_active_run_safely(status: Optional[str] = None):
    if mlflow.active_run() is not None:
        try:
            mlflow.end_run(status=status)
        except Exception:
            mlflow.end_run()


def setup_experiment(experiment_name: str):
    tracking_uri = mlflow.get_tracking_uri()
    exp = mlflow.get_experiment_by_name(experiment_name)

    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id  

    mlflow.set_experiment(experiment_name)
    return exp_id, tracking_uri


def start_run_strict(*, run_name: str, tags: Optional[Dict[str, str]] = None):
    end_active_run_safely()
    return mlflow.start_run(run_name=run_name, tags=tags or {})
