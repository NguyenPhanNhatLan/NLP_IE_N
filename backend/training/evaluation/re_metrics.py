# training/evaluations/re_metrics.py
from typing import Dict, Optional, Sequence, Any
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
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
    """
    Analyze misclassified Relation Extraction samples by printing gold vs. predicted labels
    along with entity mentions and text context for qualitative error analysis.
    """

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

def _span_gap_chars(h: Dict[str, Any], t: Dict[str, Any]) -> Optional[int]:
    """
    Return char-gap between two spans (>=0).
    If spans overlap/touch => gap = 0.
    """
    try:
        hs, he = int(h.get("start", 0)), int(h.get("end", 0))
        ts, te = int(t.get("start", 0)), int(t.get("end", 0))
    except Exception:
        return None

    # Ensure ordering inside each span
    if he < hs:
        hs, he = he, hs
    if te < ts:
        ts, te = te, ts

    # h before t
    if he <= ts:
        return ts - he
    # t before h
    if te <= hs:
        return hs - te
    # overlap
    return 0


def evaluate_re_by_distance(
    samples: Sequence[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label2id: Dict[str, int],
    idx_test: Optional[Sequence[int]] = None,
    bins: tuple = (50, 150),  # (near_max, mid_max) in chars
    far_if_cross_seg: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Distance-aware evaluation for Relation Extraction.

    - Distance: char gap between head_span and tail_span.
    - Buckets: near/mid/far by bins.
    - Metrics per bucket:
        * accuracy (all classes)
        * micro precision/recall/f1 excluding 'no_relation'
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    if idx_test is None:
        idx_test = list(range(len(y_true)))
    else:
        idx_test = list(map(int, idx_test))

    if len(idx_test) != len(y_true):
        raise ValueError(f"idx_test length ({len(idx_test)}) must match y_true length ({len(y_true)})")
      
    no_id = label2id.get("no_relation", None)
    groups = {"near": [], "mid": [], "far": []}
    
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        ex = samples[int(idx_test[i])]
        
        # Calculate char distance
        hs = ex.get("head_span", {})
        ts = ex.get("tail_span", {})
        
        if not isinstance(hs, dict) or not isinstance(ts, dict):
            continue
        
        # Cross-sentence = FAR
        if far_if_cross_seg and hs.get("seg", "A") != ts.get("seg", "A"):
            dist = 10**9
        else:
            # try:
            #     dist = abs(ts.get("start", 0) - hs.get("end", 0))
            # except:
            #     continue

            dist = _span_gap_chars(hs, ts)
            if dist is None:
                continue
        
        # Bin assignment
        if dist <= bins[0]:
            groups["near"].append(i)
        elif dist <= bins[1]:
            groups["mid"].append(i)
        else:
            groups["far"].append(i)
    
    # Compute metrics per group
    results = {}
    for name, idxs in groups.items():
        if not idxs:
            continue
        
        # yt = np.array([p[0] for p in pairs], dtype=int)
        # yp = np.array([p[1] for p in pairs], dtype=int)
        
        yt = y_true[idxs]
        yp = y_pred[idxs]
        
        # Positive-only F1
        # mask_pos = yt != no_id if no_id is not None else np.ones_like(yt, dtype=bool)

        # Micro P/R/F1 excluding no_relation label (RE-standard)
        if no_id is None:
            p, r, f1, _ = precision_recall_fscore_support(
                yt, yp, average="micro", zero_division=0
            )
        else:
            # compute over all labels except no_relation
            labels = sorted(set(yt.tolist()) | set(yp.tolist()))
            labels = [lab for lab in labels if lab != no_id]

            if len(labels) == 0:
                p = r = f1 = 0.0
            else:
                p, r, f1, _ = precision_recall_fscore_support(
                    yt, yp, labels=labels, average="micro", zero_division=0
                )
        
        results[name] = {
            "count": len(yt),
            "accuracy": float((yt == yp).mean()),
            # "f1_positive": float(
            #     f1_score(yt[mask_pos], yp[mask_pos], average="macro", zero_division=0)
            # ) if mask_pos.any() else 0.0,
            'precision_micro': float(p),
            'recall_micro': float(r),
            'f1_micro': float(f1),
        }
    
    # Print
    print("\nRE metrics by Entity Distance (micro exclude no_relation):")
    print("="*60)
    for name in ["near", "mid", "far"]:
        if name not in results:
            continue
        r = results[name]
        print(
            f"{name.upper():5s}:  "
            f"F1={r['f1_micro']:.4f}  "
            f"P={r['precision_micro']:.4f}  R={r['recall_micro']:.4f}  "
            f"Acc={r['accuracy']:.4f}  (n={r['count']})"
        )
        
    print("="*60 + "\n")
    
    return results


def log_loss_multiclass(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute multiclass log loss (categorical cross-entropy).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(y_prob, eps, 1.0 - eps)

    log_probs = -np.log(y_prob[np.arange(len(y_true)), y_true])
    return float(np.mean(log_probs))


def full_evaluation(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    samples: Optional[Sequence[Dict[str, Any]]] = None,
    idx_test: Optional[Sequence[int]] = None,
    do_error_analysis: bool = True,
    do_distance_analysis: bool = True,
    error_top_n: int = 10,
    y_prob: Optional[np.ndarray] = None
    
) -> Dict[str, Any]:
    """
    Run a full evaluation pipeline for Relation Extraction:
      1) Core metrics (always)
      2) Qualitative error analysis (optional, needs samples)
      3) Distance-based metrics (optional, needs spans in samples)
    """
    results: Dict[str, Any] = {}
    
    # 1. Core metrics (always)
    results["metrics"] = evaluate_re(
        y_true, y_pred,
        id2label=id2label,
        label2id=label2id,
        print_report=True,
    )
    # 1b) Log loss (needs probabilities)
    if y_prob is not None:
        results["metrics"]["log_loss"] = log_loss_multiclass(y_true, y_prob)
        print(f"Log loss:               {results['metrics']['log_loss']:.4f}")

    # If no samples, skip analyses that require text/spans
    if samples is None:
        return results

    
    # 2. Error analysis (if samples provided)
    if do_error_analysis:
        results["errors"] = analyze_errors(
            samples, y_true, y_pred,
            id2label=id2label,
            idx_test=idx_test,
            top_n=error_top_n,
        )
    
    # 3. Distance analysis (if samples with spans provided)
    if do_distance_analysis:
        # quick check: at least one sample has the needed keys
        has_spans = any(
            isinstance(ex.get("head_span"), dict) and isinstance(ex.get("tail_span"), dict)
            for ex in samples[: min(len(samples), 50)]
        )
        if has_spans:
            results["distance_analysis"] = evaluate_re_by_distance(
                samples, y_true, y_pred,
                label2id=label2id,
                idx_test=idx_test,
            )
        else:
            results["distance_analysis"] = {
                "skipped": True,
                "reason": "samples do not contain head_span/tail_span dicts",
            }
    
    return results