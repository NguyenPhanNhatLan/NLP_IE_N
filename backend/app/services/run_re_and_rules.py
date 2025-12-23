import json
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient

from training.mlflow.load_model import load_pyfunc_from_mlflow, load_sklearn_from_mlflow


# ----------------------------
# CACHES (rất quan trọng để tránh download artifacts liên tục)
# ----------------------------

_model_lock = Lock()

@lru_cache(maxsize=64)
def load_label_maps_from_registry(
    model_name: str,
    model_version: str,
    labels_dir: str = "model/labels",
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Download id2label/label2id từ artifacts của model version (registry).
    Cache theo (model_name, model_version, labels_dir).
    """
    client = MlflowClient()
    mv = client.get_model_version(model_name, model_version)
    run_id = mv.run_id

    id2label_path = client.download_artifacts(run_id, f"{labels_dir}/id2label.json")
    label2id_path = client.download_artifacts(run_id, f"{labels_dir}/label2id.json")

    with open(id2label_path, "r", encoding="utf-8") as f:
        id2label_raw = json.load(f)
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id_raw = json.load(f)

    id2label = {int(k): v for k, v in id2label_raw.items()}
    label2id = {str(k): int(v) for k, v in label2id_raw.items()}
    return id2label, label2id


@lru_cache(maxsize=32)
def load_model_cached(model_name: str, model_version: str, loader: str):
    """
    Cache model load để tránh 'Downloading artifacts' cho mỗi request.
    """
    loader = (loader or "sklearn").lower().strip()
    if loader in {"mlp", "torch", "pytorch"}:
        loader = "pyfunc"

    with _model_lock:
        if loader == "pyfunc":
            return load_pyfunc_from_mlflow(model_name, model_version)
        return load_sklearn_from_mlflow(model_name, model_version)


# ----------------------------
# UTILITIES
# ----------------------------

def _label_id(label_name: str, label2id: Dict[str, int]) -> Optional[int]:
    return label2id.get(label_name)

def _to_id_predictions(y_pred: List[Any], label2id: Dict[str, int]) -> List[int]:
    out: List[int] = []
    for y in y_pred:
        if isinstance(y, (int, np.integer)):
            out.append(int(y))
        else:
            y_str = str(y)
            if y_str not in label2id:
                raise ValueError(f"Predicted label '{y_str}' not found in label2id.")
            out.append(int(label2id[y_str]))
    return out

def _norm_text_field(x):
    if isinstance(x, list):
        return " ".join(str(t) for t in x).strip()
    return str(x).strip()


# ----------------------------
# POST-RULES (giữ nguyên logic của bạn)
# ----------------------------

def post_rules(
    *,
    pairs_info: List[Dict],
    y_pred_ids: List[int],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    text: str,
) -> List[int]:
    y_pred_post = list(y_pred_ids)
    sentence = (text or "").lower()

    for i, pred_id in enumerate(y_pred_ids):
        pair = pairs_info[i]
        head_type = pair.get("head_type")
        tail_type = pair.get("tail_type")

        new_label = None

        if head_type == "NAME" and tail_type == "INCI":
            new_label = _label_id("has_inci_name", label2id)

        elif tail_type == "ORIGIN" and any(kw in sentence for kw in ["chiết xuất từ", "nguồn gốc", "tổng hợp"]):
            new_label = _label_id("has_origin", label2id)

        elif tail_type == "SKIN_CONCERNS" and any(kw in sentence for kw in ["phù hợp", "vấn đề"]):
            new_label = _label_id("targets_skin_concerns", label2id)

        elif tail_type == "BENEFITS" and any(kw in sentence for kw in ["giúp", "tác dụng"]):
            new_label = _label_id("has_benefits", label2id)

        if new_label is not None:
            y_pred_post[i] = int(new_label)

    return y_pred_post


def _build_relations_output(
    text,
    pairs_info: List[Dict],
    y_pred_post_ids: List[int],
    id2label: Dict[int, str],
) -> List[Dict]:
    relations: List[Dict] = []
    print("A", text)

    for i, label_id in enumerate(y_pred_post_ids):
        label = id2label[int(label_id)]
        if label == "no_relation":
            continue

        pair = pairs_info[i]
        relations.append({
            "head_text": _norm_text_field(pair.get("head_text")),
            "head_type": pair.get("head_type"),
            "tail_text": _norm_text_field(pair.get("tail_text")),
            "tail_type": pair.get("tail_type"),
            "relation": label,

            "pair_id": pair.get("pair_id"),
            "sentence_id": pair.get("sentence_id"),
            "sentence": pair.get("sentence") or text,
        })

    return relations


def run_re_rule_from_xy_sklearn(
    *,
    X: np.ndarray,
    pairs_info: List[Dict],
    text: str,
    model,
    model_name: str,
    model_version: str,
) -> List[Dict]:
    if X is None or getattr(X, "size", 0) == 0 or not pairs_info:
        return []

    id2label, label2id = load_label_maps_from_registry(model_name=model_name, model_version=model_version)

    raw_pred = list(model.predict(X))  # str hoặc int
    y_pred_ids = _to_id_predictions(raw_pred, label2id)

    y_pred_post_ids = post_rules(
        pairs_info=pairs_info,
        y_pred_ids=y_pred_ids,
        id2label=id2label,
        label2id=label2id,
        text=text,
    )

    return _build_relations_output(text, pairs_info, y_pred_post_ids, id2label)


def run_re_rule_from_xy_pyfunc(
    *,
    X: np.ndarray,
    pairs_info: List[Dict],
    text: str,
    model,
    model_name: str,
    model_version: str,
):
    if X is None or getattr(X, "size", 0) == 0 or not pairs_info:
        return []

    X_payload = X.astype(np.float32, copy=False)
    df = pd.DataFrame({"X": [X_payload]})

    pred_df = model.predict(df)  # expect rows = N

    if len(pred_df) != len(pairs_info):
        raise ValueError(f"Mismatch: pred={len(pred_df)} vs pairs={len(pairs_info)}")

    # bạn có thể dùng relation_id trực tiếp, nhưng vẫn cần label maps cho post_rules + mapping id->label
    id2label, label2id = load_label_maps_from_registry(model_name=model_name, model_version=model_version)

    if "relation_id" in pred_df.columns:
        y_pred_ids = pred_df["relation_id"].astype(int).tolist()
    elif "relation" in pred_df.columns:
        y_pred_ids = _to_id_predictions(pred_df["relation"].tolist(), label2id)
    else:
        raise ValueError("PyFunc output must contain 'relation_id' or 'relation'")

    y_pred_post_ids = post_rules(
        pairs_info=pairs_info,
        y_pred_ids=y_pred_ids,
        id2label=id2label,
        label2id=label2id,
        text=text,
    )

    return _build_relations_output(text, pairs_info, y_pred_post_ids, id2label)


def predict_re_from_mlflow_xy(
    *,
    X: Optional[np.ndarray],
    pairs_info: List[Dict],
    text: str,
    model_name: str,
    model_version: str,
    loader: str = "sklearn",  # "sklearn" | "pyfunc"
) -> List[Dict]:
    loader = (loader or "sklearn").lower().strip()
    if loader in {"mlp", "torch", "pytorch"}:
        loader = "pyfunc"

    if X is None or getattr(X, "size", 0) == 0 or not pairs_info:
        return []

    model = load_model_cached(model_name, model_version, loader)

    if loader == "pyfunc":
        return run_re_rule_from_xy_pyfunc(
            X=X,
            pairs_info=pairs_info,
            text=text,
            model=model,
            model_name=model_name,
            model_version=model_version,
        )

    return run_re_rule_from_xy_sklearn(
        X=X,
        pairs_info=pairs_info,
        text=text,
        model=model,
        model_name=model_name,
        model_version=model_version,
    )
