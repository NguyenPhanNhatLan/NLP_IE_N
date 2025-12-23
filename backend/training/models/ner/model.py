from typing import List
import joblib
import sklearn_crfsuite
from training.models.ner.config import CRFConfig

def train_crf(X_feats, y_labels, cfg: CRFConfig):
    crf = sklearn_crfsuite.CRF(
        algorithm=cfg.algorithm,
        c1=cfg.c1,
        c2=cfg.c2,
        max_iterations=cfg.max_iterations,
        all_possible_transitions=cfg.all_possible_transitions,
    )
    crf.fit(X_feats, y_labels)
    return crf

def save_model(crf, path: str):
    joblib.dump(crf, path)

def load_model(path: str):
    return joblib.load(path)

def predict_raw_crf(crf, X_feats) -> List[List[str]]:
    return crf.predict(X_feats)
