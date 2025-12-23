import json
from typing import Dict, List

def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def ensure_format(ex: Dict) -> None:
    if "tokens" not in ex or "labels" not in ex:
        raise ValueError("Each example must have 'tokens' and 'labels'")
    if len(ex["tokens"]) != len(ex["labels"]):
        raise ValueError("tokens and labels must have same length")

def to_xy(dataset: List[Dict]):
    X_tokens, y_labels = [], []
    for ex in dataset:
        ensure_format(ex)
        X_tokens.append(ex["tokens"])
        y_labels.append(ex["labels"])
    return X_tokens, y_labels
