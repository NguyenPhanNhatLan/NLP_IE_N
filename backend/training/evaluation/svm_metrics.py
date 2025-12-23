# training/evaluations/re_metrics.py
from typing import Dict, Optional, Sequence, Any
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_re(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    print_report: bool = True,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # 1. Standard metrics
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    
    metrics = {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }
    
    # 2. Positive-only F1 (CRITICAL for RE - most important metric)
    no_id = label2id.get("no_relation")
    if no_id is not None:
        mask_pos = y_true != no_id
        if mask_pos.any():
            metrics["f1_positive_only"] = float(
                f1_score(y_true[mask_pos], y_pred[mask_pos], average="macro", zero_division=0)
            )
            metrics["positive_support"] = int(mask_pos.sum())
        else:
            metrics["f1_positive_only"] = 0.0
            metrics["positive_support"] = 0
    
    # 3. Per-class F1 (for analysis)
    report_dict = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = {
        name: {
            "f1": report_dict[name]["f1-score"],
            "support": report_dict[name]["support"],
        }
        for name in target_names
    }
    
    # 4. Confusion matrix
    labels = sorted(id2label.keys())
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    
    # Print summary
    if print_report:
        print("\n" + "="*70)
        print("RE EVALUATION RESULTS")
        print("="*70)
        print(f"Overall Accuracy:        {metrics['accuracy']:.4f}")
        print(f"F1 Micro:                {metrics['f1_micro']:.4f}")
        print(f"F1 Macro:                {metrics['f1_macro']:.4f}")
        print(f"F1 Weighted:             {metrics['f1_weighted']:.4f}")
        if "f1_positive_only" in metrics:
            print(f"F1 Positive-Only (KEY): {metrics['f1_positive_only']:.4f}  (n={metrics['positive_support']})")
        print("="*70)
        
        # Per-class breakdown
        print("\nPer-class F1 scores:")
        for name in target_names:
            f1 = metrics["per_class"][name]["f1"]
            sup = metrics["per_class"][name]["support"]
            print(f"  {name:25s}  F1={f1:.4f}  (n={sup})")
        print("="*70 + "\n")
    
    return metrics



def analyze_errors(
    samples: Sequence[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    id2label: Dict[int, str],
    idx_test: Optional[Sequence[int]] = None,
    top_n: int = 10,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if idx_test is None:
        idx_test = list(range(len(y_true)))
    
    errors = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt == yp:
            continue
            
        sample_idx = int(idx_test[i])
        ex = samples[sample_idx]
        
        # Extract text
        if "sentence" in ex and ex["sentence"]:
            text = ex["sentence"]
        else:
            text = f"{ex.get('sentence_a', '')} [SEP] {ex.get('sentence_b', '')}".strip()
        
        errors.append({
            "id": ex.get("task_id", ex.get("id", sample_idx)),
            "true_label": id2label[int(yt)],
            "pred_label": id2label[int(yp)],
            "head": f"{ex.get('head_text')} ({ex.get('head_type', '?')})",
            "tail": f"{ex.get('tail_text')} ({ex.get('tail_type', '?')})",
            "text": text[:300],  # truncate long texts
        })
        
        if len(errors) >= top_n:
            break
    
    # Print nicely
    print(f"\nTop {len(errors)} Errors:")
    print("="*90)
    for err in errors:
        print(f"\n[{err['id']}] TRUE: {err['true_label']} → PRED: {err['pred_label']}")
        print(f"  HEAD: {err['head']}")
        print(f"  TAIL: {err['tail']}")
        print(f"  TEXT: {err['text']}")
        print("-"*90)
    
    return errors



def evaluate_re_by_distance(
    samples: Sequence[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label2id: Dict[str, int],
    idx_test: Optional[Sequence[int]] = None,
    bins: tuple = (50, 150),  # (near_max, mid_max) in chars
) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if idx_test is None:
        idx_test = list(range(len(y_true)))
    
    no_id = label2id.get("no_relation")
    groups = {"near": [], "mid": [], "far": []}
    
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        ex = samples[int(idx_test[i])]
        
        # Calculate char distance
        hs = ex.get("head_span", {})
        ts = ex.get("tail_span", {})
        
        if not isinstance(hs, dict) or not isinstance(ts, dict):
            continue
        
        # Cross-sentence = FAR
        if hs.get("seg", "A") != ts.get("seg", "A"):
            dist = 999999
        else:
            try:
                dist = abs(ts.get("start", 0) - hs.get("end", 0))
            except:
                continue
        
        # Bin assignment
        if dist <= bins[0]:
            groups["near"].append((int(yt), int(yp)))
        elif dist <= bins[1]:
            groups["mid"].append((int(yt), int(yp)))
        else:
            groups["far"].append((int(yt), int(yp)))
    
    # Compute metrics per group
    results = {}
    for name, pairs in groups.items():
        if not pairs:
            continue
        
        yt = np.array([p[0] for p in pairs])
        yp = np.array([p[1] for p in pairs])
        
        # Positive-only F1
        mask_pos = yt != no_id if no_id is not None else np.ones_like(yt, dtype=bool)
        
        results[name] = {
            "count": len(yt),
            "accuracy": float((yt == yp).mean()),
            "f1_positive": float(
                f1_score(yt[mask_pos], yp[mask_pos], average="macro", zero_division=0)
            ) if mask_pos.any() else 0.0,
        }
    
    # Print
    print("\nF1 by Entity Distance:")
    print("="*60)
    for name in ["near", "mid", "far"]:
        if name not in results:
            continue
        r = results[name]
        print(f"{name.upper():5s}:  F1={r['f1_positive']:.4f}  Acc={r['accuracy']:.4f}  (n={r['count']})")
    print("="*60 + "\n")
    
    return results



def full_evaluation(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    samples: Optional[Sequence[Dict[str, Any]]] = None,
    idx_test: Optional[Sequence[int]] = None,
    do_analyze_errors: bool = True,
    do_analyze_distance: bool = True,
    error_top_n: int = 10,
) -> Dict[str, Any]:
    results = {}
    
    # 1. Core metrics (always)
    results["metrics"] = evaluate_re(
        y_true, y_pred,
        id2label=id2label,
        label2id=label2id,
        print_report=True,
    )
    
    # 2. Error analysis (if samples provided)
    if samples is not None and do_analyze_errors:
        results["errors"] = analyze_errors(
            samples, y_true, y_pred,
            id2label=id2label,
            idx_test=idx_test,
            top_n=error_top_n,
        )
    
    # 3. Distance analysis (if samples with spans provided)
    if samples is not None and do_analyze_distance:
        results["distance_analysis"] = evaluate_re_by_distance(
            samples, y_true, y_pred,
            label2id=label2id,
            idx_test=idx_test,
        )
    
    return results