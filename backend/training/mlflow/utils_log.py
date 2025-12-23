from typing import Any, Dict, List, Optional

import mlflow

import math

def _to_float_or_none(x):
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def log_metric_safe(key: str, value):
    v = _to_float_or_none(value)
    if v is None:
        # im lặng hoặc print nếu bạn muốn
        # print(f"[mlflow] skip metric {key}={value}")
        return
    mlflow.log_metric(key, v)

def log_metrics_safe(metrics: Dict[str, Any]):
    safe = {}
    for k, v in (metrics or {}).items():
        fv = _to_float_or_none(v)
        if fv is not None:
            safe[k] = fv
    if safe:
        mlflow.log_metrics(safe)
        
def log_params_required(
    params: Dict[str, Any],
    *,
    required_keys: List[str],
    prefix: Optional[str] = None,
):
    missing = [k for k in required_keys if k not in params or params[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required params: {missing}")

    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float, str, bool)) or v is None:
            mlflow.log_param(key, v)
        else:
            mlflow.log_param(key, str(v))



def log_label_maps(
    *,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    artifact_dir: str = "model/labels",
):
    id2label_path = f"{artifact_dir}/id2label.json"
    label2id_path = f"{artifact_dir}/label2id.json"

    mlflow.log_dict({str(k): v for k, v in id2label.items()}, id2label_path)
    mlflow.log_dict({k: int(v) for k, v in label2id.items()}, label2id_path)

    return {"id2label": id2label_path, "label2id": label2id_path}



def flatten_numeric(d: Dict[str, Any], prefix: str = "", sep: str = "."):
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_numeric(v, prefix=key, sep=sep))
        elif isinstance(v, (int, float)) or hasattr(v, "item"):
            try:
                out[key] = float(v)
            except Exception:
                pass
        else:
            pass
    return out


def extract_confusion_matrix(results: Dict[str, Any]) -> Optional[Any]:
    if not isinstance(results, dict):
        return None

    # common patterns
    cm = results.get("metrics", {}).get("confusion_matrix")
    if cm is None:
        cm = results.get("confusion_matrix")

    if cm is None:
        return None

    # make it JSON-friendly if it's numpy-like
    if hasattr(cm, "tolist"):
        try:
            return cm.tolist()
        except Exception:
            return cm

    return cm


def format_errors_text(errors: List[Dict[str, Any]], max_errors: int = 50):
    lines: List[str] = []
    for e in (errors or [])[:max_errors]:
        if not isinstance(e, dict):
            lines.append(str(e))
            continue

        # a few typical fields (safe get)
        _id = e.get("id", e.get("task_id", "N/A"))
        reason = e.get("reason", e.get("message", ""))
        head = e.get("head_text", "")
        tail = e.get("tail_text", "")
        relation = e.get("relation", e.get("label", ""))
        sent = e.get("sentence", "")

        lines.append(f"ID: {_id}")
        if relation:
            lines.append(f"Relation: {relation}")
        if head or tail:
            lines.append(f"Head: {head} | Tail: {tail}")
        if reason:
            lines.append(f"Reason: {reason}")
        if sent:
            lines.append(f"Sentence: {sent}")
        lines.append("-" * 60)

    if not lines:
        return "No errors."
    return "\n".join(lines)


def log_eval_results(
    results: Dict[str, Any],
    *,
    metric_prefix: str = "eval",
    artifact_dir: str = "evaluation",
):
    metrics = flatten_numeric(results, prefix=metric_prefix)
    log_metrics_safe(metrics)

    # full json
    mlflow.log_dict(results, f"{artifact_dir}/full_evaluation.json")

    # confusion matrix
    cm = extract_confusion_matrix(results)
    if cm is not None:
        mlflow.log_dict({"confusion_matrix": cm}, f"{artifact_dir}/confusion_matrix.json")

    # errors
    errors = results.get("errors")
    if errors:
        mlflow.log_dict({"errors": errors}, f"{artifact_dir}/errors.json")
        mlflow.log_text(format_errors_text(errors, max_errors=50), f"{artifact_dir}/errors.txt")
        
def pick_core_eval_metrics(results: Dict[str, Any], *, metric_prefix: str = "test"):
    if not isinstance(results, dict):
        return {"accuracy": None, "macro_f1": None, "weighted_f1": None}

    m = results.get("metrics", {}) if isinstance(results.get("metrics"), dict) else {}

    return {
        "accuracy": m.get("accuracy") or results.get("accuracy") or results.get(f"{metric_prefix}_accuracy"),
        "macro_f1": m.get("f1_macro") or m.get("macro_f1") or results.get("macro_f1") or results.get(f"{metric_prefix}_macro_f1"),
        "weighted_f1": m.get("f1_weighted") or m.get("weighted_f1") or results.get("weighted_f1") or results.get(f"{metric_prefix}_weighted_f1"),
    }




def print_run_summary(report: Dict[str, Any]) -> None:
    r = report.get("run", {})
    e = report.get("eval", {})
    reg = report.get("registry", {})

    print("========== MLflow Run Summary ==========")
    print(f"- status: {report.get('status')}")
    print(f"- experiment: {r.get('experiment_name')} (id={r.get('experiment_id')})")
    print(f"- run: {r.get('run_name')} (id={r.get('run_id')})")
    print(f"- tracking_uri: {r.get('tracking_uri')}")
    print(f"- artifact_uri: {r.get('artifact_uri')}")
    print(f"- test_accuracy: {e.get('test_accuracy')}")
    print(f"- macro_f1: {e.get('macro_f1')}")
    print(f"- weighted_f1: {e.get('weighted_f1')}")

    if reg.get("attempted"):
        print(f"- registry_registered: {reg.get('registered')}")
        print(f"- registry_name: {reg.get('name')}")
        print(f"- registry_version: {reg.get('version')}")
        print(f"- registry_stage: {reg.get('stage')}")
        if reg.get("error"):
            print(f"- registry_error: {reg.get('error')}")
    print("========================================")
