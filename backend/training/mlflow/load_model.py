import json
import os
import threading
from typing import Dict, Tuple

import mlflow
from dotenv import load_dotenv


_SKLEARN_CACHE: Dict[str, object] = {}
_PYFUNC_CACHE: Dict[str, object] = {}
_LABELS_CACHE: Dict[str, Tuple[Dict[int, str], Dict[str, int]]] = {}

# Locks
_GLOBAL_LOCK = threading.Lock()
_PYFUNC_LOCKS: Dict[str, threading.Lock] = {}
_SKLEARN_LOCK = threading.Lock()
_LABELS_LOCK = threading.Lock()


def _ensure_tracking_uri() -> str:
    load_dotenv()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("Missing MLFLOW_TRACKING_URI in environment.")
    mlflow.set_tracking_uri(uri)
    return uri


def _model_uri(model_name: str, model_version: str) -> str:
    return f"models:/{model_name}/{model_version}"


def load_sklearn_from_mlflow(model_name: str, model_version: str):
    _ensure_tracking_uri()
    key = f"sklearn:{model_name}:{model_version}"

    # thread-safe lazy load (behavior same, avoids double-load under concurrency)
    with _SKLEARN_LOCK:
        model = _SKLEARN_CACHE.get(key)
        if model is None:
            model = mlflow.sklearn.load_model(_model_uri(model_name, model_version))
            _SKLEARN_CACHE[key] = model
        return model


def _get_per_key_lock(key: str) -> threading.Lock:
    with _GLOBAL_LOCK:
        lock = _PYFUNC_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PYFUNC_LOCKS[key] = lock
        return lock


def load_pyfunc_from_mlflow(model_name: str, model_version: str):
    _ensure_tracking_uri()
    key = f"pyfunc:{model_name}:{model_version}"

    # fast path
    model = _PYFUNC_CACHE.get(key)
    if model is not None:
        return model

    # only one thread loads this model key
    lock = _get_per_key_lock(key)
    with lock:
        model = _PYFUNC_CACHE.get(key)
        if model is None:
            model = mlflow.pyfunc.load_model(_model_uri(model_name, model_version))
            _PYFUNC_CACHE[key] = model
        return model


def load_label_mappings_from_registry(
    model_name: str,
    model_version: str,
    *,
    labels_dir: str = "model/labels",
) -> Tuple[Dict[int, str], Dict[str, int]]:
    _ensure_tracking_uri()

    cache_key = f"labels:{model_name}:{model_version}:{labels_dir}"

    # thread-safe lazy load (same result, avoids concurrent download)
    with _LABELS_LOCK:
        cached = _LABELS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        model_uri = _model_uri(model_name, model_version)
        local_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{model_uri}/{labels_dir}"
        )

        id2_path = os.path.join(local_dir, "id2label.json")
        l2_path = os.path.join(local_dir, "label2id.json")

        if not os.path.exists(id2_path) or not os.path.exists(l2_path):
            raise FileNotFoundError(
                "Label mapping files not found. "
                f"Expected: {id2_path} and {l2_path}. "
                f"Check you logged labels to '{labels_dir}'."
            )

        with open(id2_path, "r", encoding="utf-8") as f:
            id2_raw = json.load(f)
        with open(l2_path, "r", encoding="utf-8") as f:
            l2_raw = json.load(f)

        id2label = {int(k): str(v) for k, v in id2_raw.items()}
        label2id = {str(k): int(v) for k, v in l2_raw.items()}

        _LABELS_CACHE[cache_key] = (id2label, label2id)
        return id2label, label2id
