from typing import Dict, List
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)


def evaluate_seqeval(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

def get_report_str(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=4)

def print_report(y_true: List[List[str]], y_pred: List[List[str]]):
    print(classification_report(y_true, y_pred, digits=4))
