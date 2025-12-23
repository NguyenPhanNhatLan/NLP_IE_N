# training/models/mlp/train_re.py
# Run:
#   python -u -m training.models.mlp.train_re
#   MLFLOW_EXPERIMENT_NAME="mlp_re" python -u -m training.models.mlp.train_re

import os
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow
import mlflow.pyfunc

from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from training.mlflow.schema import init_report
from training.mlflow.run_utils import end_active_run_safely, setup_experiment, start_run_strict
from training.mlflow.utils_log import (
    log_metric_safe,
    log_params_required,
    log_label_maps,
    log_eval_results,
    pick_core_eval_metrics,
    print_run_summary,
)
from training.mlflow.registry import register_and_promote

from training.features.build_data.build_re_dataset import build_re_datasets
from training.features.vectorize.utils import build_label_maps
from training.features.vectorize.re_vectorize import build_xy_dl
from training.evaluation.re_metrics import full_evaluation

PHOBERT_NAME = os.getenv("PHOBERT_NAME", "vinai/phobert-base")


@dataclass(frozen=True)
class MLPTrainConfig:
    max_len: int = 256
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42

    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    hidden_dim: int = 256
    dropout: float = 0.3

    phobert_name: str = PHOBERT_NAME
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlp_re")
    run_name: str = "mlp_v1"
    registry_name: str = os.getenv("MLFLOW_REGISTRY_NAME", "mlp_re_model")
    do_register: bool = True

    model_type: str = "mlp_re"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def _to_cpu_tensor_float(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu().float()
    return torch.tensor(x, dtype=torch.float32)


def _to_cpu_tensor_long(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu().long()
    return torch.tensor(x, dtype=torch.long)

class RE_MLP_X_PyFunc(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        *,
        device: str,
        id2label: Dict[int, str],
    ):
        super().__init__()
        self.device = device
        self.id2label = id2label
        self.clf = None

    def load_context(self, context):
        state = torch.load(context.artifacts["clf_state_dict"], map_location="cpu")
        meta = torch.load(context.artifacts["clf_meta"], map_location="cpu")
        input_dim = int(meta["input_dim"])
        n_classes = int(meta["n_classes"])
        hidden_dim = int(meta["hidden_dim"])
        dropout = float(meta["dropout"])

        self.clf = MLP(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)
        self.clf.load_state_dict(state)
        self.clf.eval()

    def predict(self, context, model_input):
        import pandas as pd

        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("PyFunc input must be a pandas.DataFrame")

        # Preferred: one column "X" contains 2D array/list of shape (N, D)
        if "X" in model_input.columns:
            X_obj = model_input.iloc[0]["X"]
            X_np = np.asarray(X_obj, dtype=np.float32)
            if X_np.ndim == 1:
                X_np = X_np.reshape(1, -1)
        else:
            X_np = model_input.to_numpy(dtype=np.float32)
            if X_np.ndim == 1:
                X_np = X_np.reshape(1, -1)

        if X_np.size == 0:
            return pd.DataFrame({"relation": [], "relation_id": [], "prob": []})

        Xt = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.clf(Xt)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            pred_id = probs.argmax(axis=-1)

        pred_label = [self.id2label[int(i)] for i in pred_id]
        pred_prob = probs.max(axis=-1).tolist()

        return pd.DataFrame(
            {"relation": pred_label, "relation_id": pred_id.tolist(), "prob": pred_prob}
        )
        
def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    n_classes: int,
    cfg: MLPTrainConfig,
) -> nn.Module:
    set_seed(cfg.seed)
    device = cfg.device

    Xtr = _to_cpu_tensor_float(X_train)
    ytr = _to_cpu_tensor_long(y_train)
    Xva = _to_cpu_tensor_float(X_val)
    yva = _to_cpu_tensor_long(y_val)

    assert Xtr.dim() == 2, f"Expected X_train (N,D), got {tuple(Xtr.shape)}"
    assert Xva.dim() == 2, f"Expected X_val (N,D), got {tuple(Xva.shape)}"

    input_dim = int(Xtr.shape[1])
    clf = MLP(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    # class weights (balanced)
    ytr_np = ytr.numpy().astype(int).reshape(-1)
    classes = np.unique(ytr_np)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=ytr_np)

    cw_full = np.ones(n_classes, dtype=np.float32)
    for c, w in zip(classes, cw):
        cw_full[int(c)] = float(w)
    cw_t = torch.tensor(cw_full, dtype=torch.float32).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=cw_t)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.lr)

    best_f1 = -1.0
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        clf.train()
        idx = torch.randperm(Xtr.size(0))
        Xtr_s = Xtr[idx]
        ytr_s = ytr[idx]

        total_loss = 0.0
        for i in range(0, Xtr_s.size(0), cfg.batch_size):
            xb = Xtr_s[i : i + cfg.batch_size].to(device)
            yb = ytr_s[i : i + cfg.batch_size].to(device)

            opt.zero_grad(set_to_none=True)
            logits = clf(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        clf.eval()
        with torch.no_grad():
            pred = clf(Xva.to(device)).argmax(-1)
            f1 = f1_score(yva.numpy(), pred.cpu().numpy(), average="macro", zero_division=0)

        print(
            f"Epoch {ep:02d} | loss={total_loss / Xtr_s.size(0):.4f} | val_f1_macro={f1:.4f}",
            flush=True,
        )

        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}

    if best_state is not None:
        clf.load_state_dict(best_state)

    return clf


def build_encoder_and_tokenizer(cfg: MLPTrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.phobert_name, use_fast=False)
    encoder = AutoModel.from_pretrained(cfg.phobert_name).to(cfg.device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return tokenizer, encoder


def split_samples(samples: List[dict], label2id: Dict[str, int], cfg: MLPTrainConfig):
    y_str = [ex["relation"] for ex in samples]
    y_all = np.array([label2id[s] for s in y_str], dtype=np.int64)

    train_samples, test_samples, y_train, y_test = train_test_split(
        samples,
        y_all,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y_all,
    )
    tr_samples, val_samples, y_tr, y_val = train_test_split(
        train_samples,
        y_train,
        test_size=cfg.val_size,
        random_state=cfg.seed,
        stratify=y_train,
    )
    return tr_samples, val_samples, test_samples, y_tr, y_val, y_test


def vectorize_split(
    tr_samples: List[dict],
    val_samples: List[dict],
    test_samples: List[dict],
    tokenizer,
    encoder,
    label2id: Dict[str, int],
    cfg: MLPTrainConfig,
):
    X_tr, y_tr2, idx_tr = build_xy_dl(tr_samples, tokenizer, encoder, device=cfg.device, max_len=cfg.max_len, label2id=label2id)
    X_val, y_val2, idx_val = build_xy_dl(val_samples, tokenizer, encoder, device=cfg.device, max_len=cfg.max_len, label2id=label2id)
    X_te, y_te2, idx_te = build_xy_dl(test_samples, tokenizer, encoder, device=cfg.device, max_len=cfg.max_len, label2id=label2id)
    return (X_tr, y_tr2, idx_tr), (X_val, y_val2, idx_val), (X_te, y_te2, idx_te)


def evaluate_on_test(
    clf: nn.Module,
    X_te,
    y_te,
    id2label,
    label2id,
    test_samples: List[dict],
    cfg: MLPTrainConfig,
):
    clf.eval()
    with torch.no_grad():
        Xt = _to_cpu_tensor_float(X_te).to(cfg.device)
        logits = clf(Xt)
        y_pred = logits.argmax(-1).cpu().numpy()
        y_prob = torch.softmax(logits, dim=-1).cpu().numpy()

    results = full_evaluation(
        y_true=y_te,
        y_pred=y_pred,
        y_prob=y_prob,
        id2label=id2label,
        label2id=label2id,
        samples=test_samples,
        idx_test=None,
        do_error_analysis=True,
        do_distance_analysis=True,
        error_top_n=10,
    )

    return y_pred, y_prob, results


def log_pyfunc_model(
    *,
    clf: nn.Module,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    cfg: MLPTrainConfig,
    artifact_path: str = "model",
):
    import tempfile
    from pathlib import Path
    import pandas as pd
    from mlflow.models.signature import infer_signature

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        clf_state = tmpdir / "clf_state_dict.pt"
        clf_meta = tmpdir / "clf_meta.pt"

        torch.save({k: v.detach().cpu() for k, v in clf.state_dict().items()}, str(clf_state))

        input_dim = int(clf.fc1.in_features)
        n_classes = int(clf.fc2.out_features)
        meta = {
            "input_dim": input_dim,
            "n_classes": n_classes,
            "hidden_dim": int(cfg.hidden_dim),
            "dropout": float(cfg.dropout),
        }
        torch.save(meta, str(clf_meta))

        pyfunc = RE_MLP_X_PyFunc(
            device=cfg.device,
            id2label=id2label,
        )

        input_example = pd.DataFrame({"X": [np.zeros((1, input_dim), dtype=np.float32)]})
        output_example = pd.DataFrame({"relation": ["no_relation"], "relation_id": [0], "prob": [1.0]})
        signature = infer_signature(input_example, output_example)

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=pyfunc,
            artifacts={
                "clf_state_dict": str(clf_state),
                "clf_meta": str(clf_meta),
            },
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "mlflow",
                "torch",
                "numpy",
                "pandas",
            ],
        )



def train_re_mlp_and_report(
    *,
    cfg: MLPTrainConfig,
    tags: Optional[Dict[str, str]] = None,
    required_param_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:

    required_param_keys = required_param_keys or ["model_type", "phobert_name", "device"]

    params = {
        "model_type": cfg.model_type,
        "phobert_name": cfg.phobert_name,
        "device": cfg.device,
        "mlp.max_len": cfg.max_len,
        "mlp.test_size": cfg.test_size,
        "mlp.val_size": cfg.val_size,
        "mlp.seed": cfg.seed,
        "mlp.lr": cfg.lr,
        "mlp.batch_size": cfg.batch_size,
        "mlp.epochs": cfg.epochs,
        "mlp.hidden_dim": cfg.hidden_dim,
        "mlp.dropout": cfg.dropout,
    }

    report = init_report(
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        params=params,
        tags=tags,
        registry_name=cfg.registry_name,
        do_register=cfg.do_register,
    )

    t0 = time.time()

    try:
        end_active_run_safely()

        exp_id, tracking_uri = setup_experiment(cfg.experiment_name)
        report["run"]["experiment_id"] = exp_id
        report["run"]["tracking_uri"] = tracking_uri

        # data
        samples = build_re_datasets()
        report["data"]["dataset_size"] = len(samples)

        # label maps
        id2label, label2id = build_label_maps(samples)

        # encoder/tokenizer (frozen)
        tokenizer, encoder = build_encoder_and_tokenizer(cfg)

        # split
        tr_samples, val_samples, test_samples, _, _, _ = split_samples(samples, label2id, cfg)

        # vectorize
        print("vectorizing", flush=True)
        (X_tr, y_tr2, _), (X_val, y_val2, _), (X_te, y_te2, _) = vectorize_split(
            tr_samples, val_samples, test_samples, tokenizer, encoder, label2id, cfg
        )

        # train
        print("training", flush=True)
        clf = train_mlp(X_tr, y_tr2, X_val, y_val2, n_classes=len(label2id), cfg=cfg)

        # eval
        y_pred, y_prob, results = evaluate_on_test(clf, X_te, y_te2, id2label, label2id, test_samples, cfg)

        # log to mlflow
        with start_run_strict(run_name=cfg.run_name, tags=tags) as run:
            report["run"]["run_id"] = run.info.run_id
            report["run"]["artifact_uri"] = mlflow.get_artifact_uri()

            mlflow.log_params(params)
            log_params_required(params, required_keys=required_param_keys)

            mlflow.log_metric("data.dataset_size", float(len(samples)))
            mlflow.log_metric("train.num_classes", float(len(label2id)))

            # eval artifacts
            try:
                log_eval_results(results, metric_prefix="test", artifact_dir="evaluation")
            except Exception as e:
                print("Skip logging evaluation artifacts:", e, flush=True)

            core = pick_core_eval_metrics(results, metric_prefix="test")

            def _f(x):
                try:
                    return float(x) if x is not None else float("nan")
                except Exception:
                    return float("nan")

            report["eval"].update({
                "test_accuracy": _f(core.get("accuracy")),
                "macro_f1": _f(core.get("macro_f1")),
                "weighted_f1": _f(core.get("weighted_f1")),
            })

            log_metric_safe("report.test_accuracy", core.get("accuracy"))
            log_metric_safe("report.macro_f1", core.get("macro_f1"))
            log_metric_safe("report.weighted_f1", core.get("weighted_f1"))

            log_pyfunc_model(
                clf=clf,
                id2label=id2label,
                label2id=label2id,
                cfg=cfg,
                artifact_path="model",
            )

            labels_paths = log_label_maps(id2label=id2label, label2id=label2id, artifact_dir="model/labels")

            report["artifacts"] = {
                "model": "model",
                "labels": labels_paths,
                "evaluation": "evaluation",
            }

            # registry + promote
            if cfg.do_register and cfg.registry_name:
                report["registry"]["attempted"] = True
                try:
                    reg_info = register_and_promote(
                        run_id=run.info.run_id,
                        name=cfg.registry_name,
                        artifact_path="model",
                    )
                    report["registry"]["registered"] = True
                    report["registry"]["details"] = reg_info
                except Exception as e:
                    report["registry"]["error"] = str(e)

            report["status"] = "completed"
            report["train"]["duration_sec"] = round(time.time() - t0, 3)
            mlflow.log_metric("train.duration_sec", report["train"]["duration_sec"])

            print_run_summary(report)
            return report

    except Exception as e:
        report["status"] = "failed"
        report["error"] = str(e)
        if mlflow.active_run():
            mlflow.log_param("error", str(e))
            end_active_run_safely(status="FAILED")
        raise
    finally:
        end_active_run_safely()


if __name__ == "__main__":
    cfg = MLPTrainConfig(
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlp_re"),
        run_name="mlp_v1",
        registry_name=os.getenv("MLFLOW_REGISTRY_NAME", "re_mlp_phobert"),
        do_register=True,
    )
    train_re_mlp_and_report(cfg=cfg)
