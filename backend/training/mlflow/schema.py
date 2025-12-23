from typing import Any, Dict, Optional


def init_report(
    *,
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    tags: Optional[Dict[str, str]] = None,
    registry_name: Optional[str] = None,
    do_register: bool = True,
):

    return {
        # ===== overall =====
        "status": "started",        # started | completed | failed
        "error": None,

        # ===== run info =====
        "run": {
            "experiment_name": experiment_name,
            "experiment_id": None,
            "run_name": run_name,
            "run_id": None,
            "tracking_uri": None,
            "artifact_uri": None,
        },

        # ===== inputs =====
        "inputs": {
            "params": params,
            "tags": tags or {},
            "registry_name": registry_name,
            "do_register": do_register,
        },

        # ===== data =====
        "data": {
            # ví dụ:
            # "dataset_size": 1234
        },

        # ===== training =====
        "train": {
            # ví dụ:
            # "num_classes": 5,
            # "duration_sec": 12.34
        },

        # ===== evaluation =====
        "eval": {
            # ví dụ:
            # "test_accuracy": 0.82,
            # "macro_f1": 0.79,
            # "weighted_f1": 0.81
        },

        # ===== artifacts =====
        "artifacts": {
            # ví dụ:
            # "model": "model",
            # "labels": {
            #     "id2label": "model/labels/id2label.json",
            #     "label2id": "model/labels/label2id.json"
            # },
            # "evaluation": "evaluation"
        },

        # ===== model registry =====
        "registry": {
            "attempted": False,
            "registered": False,
            "name": None,
            "version": None,
            "stage": None,
            "details": None,
            "error": None,
        },
    }
