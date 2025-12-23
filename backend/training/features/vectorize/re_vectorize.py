import numpy as np
import torch
from training.features.vectorize.utils import inject_markers_segmented
from transformers import AutoTokenizer, AutoModel
from training.features.build_data.build_re_dataset import build_re_datasets

HEAD_S, HEAD_E = "[HEAD]", "[/HEAD]"
TAIL_S, TAIL_E = "[TAIL]", "[/TAIL]"

@torch.no_grad()
def phobert_vectorize_re_sample_tensor(
    sample: dict,
    tokenizer,
    model,
    device: str = "cpu",
    max_len: int = 256,
) -> torch.Tensor:
    # 1) Inject marker
    marked_text = inject_markers_segmented(sample)
    if not isinstance(marked_text, str):
        raise ValueError(
            f"inject_markers_segmented must return str, got {type(marked_text)}; keys={list(sample.keys())}"
        )

    # 2) Tokenize + move to device
    encoding = tokenizer(
        marked_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 3) Run PhoBERT
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_size)

    # 4) ids -> tokens to find marker indices
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    def find_marker_index(marker: str) -> int | None:
        for i, tok in enumerate(tokens):
            if marker in tok:
                return i
        return None

    head_start = find_marker_index(HEAD_S)
    head_end   = find_marker_index(HEAD_E)
    tail_start = find_marker_index(TAIL_S)
    tail_end   = find_marker_index(TAIL_E)

    mask = attention_mask[0].unsqueeze(-1).to(token_embeddings.dtype)  # (seq_len, 1)
    denom = mask.sum().clamp_min(1.0)
    sent_vector = (token_embeddings * mask).sum(dim=0) / denom          # (hidden,)

    # 6) Entity span pooling
    def mean_pool_span(start: int | None, end: int | None) -> torch.Tensor | None:
        if start is None or end is None:
            return None
        if end <= start + 1:
            return None
        return token_embeddings[start + 1 : end].mean(dim=0)

    head_vector = mean_pool_span(head_start, head_end) or sent_vector
    tail_vector = mean_pool_span(tail_start, tail_end) or sent_vector

    #7. Interaction features
    diff_vector = torch.abs(head_vector - tail_vector)
    mul_vector  = head_vector * tail_vector

    # 8) Final feature
    feature = torch.cat(
        [sent_vector, head_vector, tail_vector, diff_vector, mul_vector],
        dim=0,
    )  # (5*hidden,)

    return feature.detach().cpu()


def build_xy_ml(samples, tokenizer, model, device="cpu", max_len=256, label2id=None):
    """Trả về numpy array để huấn luyện các mô hình ML truyền thống (SVM, LR, RF)."""
    X_list, y_list, idx_list = [], [], []
    bad = 0

    for i, ex in enumerate(samples):
        try:
            feat = phobert_vectorize_re_sample_tensor(
                ex, tokenizer, model, device=device, max_len=max_len
            )  # torch.Tensor (cpu)
            X_list.append(feat.numpy())  # -> np
            y_list.append(ex["relation"] if label2id is None else label2id[ex["relation"]])
            idx_list.append(i)
        except ValueError as e:
            bad += 1
            print(f"[SKIP] {e} | relation={ex.get('relation')} | task_id={ex.get('task_id')}")

    print(f"Skipped {bad} bad samples")
    return np.vstack(X_list), np.array(y_list), np.array(idx_list)


def build_xy_dl(samples, tokenizer, model, device="cpu", max_len=256, label2id=None):
    X_list, y_list, idx_list = [], [], []
    bad = 0

    for i, ex in enumerate(samples):
        try:
            feat = phobert_vectorize_re_sample_tensor(
                ex, tokenizer, model, device=device, max_len=max_len
            )  # torch.Tensor (cpu)
            X_list.append(feat)  # keep torch
            y_list.append(ex["relation"] if label2id is None else label2id[ex["relation"]])
            idx_list.append(i)
        except ValueError as e:
            bad += 1
            print(f"[SKIP] {e} | relation={ex.get('relation')} | task_id={ex.get('task_id')}")

    print(f"Skipped {bad} bad samples")

    X = torch.stack(X_list, dim=0)  # (N, feat_dim)

    # y: nếu label2id đã map ra int -> tensor long; nếu chưa map -> bạn cần map trước (DL thường cần int)
    if len(y_list) > 0 and isinstance(y_list[0], str):
        raise ValueError("build_xy_dl cần label2id để map relation (str) -> int (class id).")

    y = torch.tensor(y_list, dtype=torch.long)      # (N,)
    idx = torch.tensor(idx_list, dtype=torch.long)  # (N,)
    return X, y, idx


def phobert_vectorize_re_sample_tensor_batch(
    samples: list[dict],
    tokenizer,
    model,
    device: str = "cpu",
    max_len: int = 256,
):
    marked_texts = []
    for s in samples:
        marked = inject_markers_segmented(s)
        if not isinstance(marked, str):
            raise ValueError(
                f"inject_markers_segmented must return str, got {type(marked)}; keys={list(s.keys())}"
            )
        marked_texts.append(marked)

    enc = tokenizer(
        marked_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 3) PhoBERT forward (ONCE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L, H)

    feats: list[torch.Tensor] = []

    # 4) per-sample pooling (same logic as single)
    for i in range(hidden.size(0)):
        token_emb = hidden[i]               # (L, H)
        mask = attention_mask[i].unsqueeze(-1).to(token_emb.dtype)
        denom = mask.sum().clamp_min(1.0)
        sent_vec = (token_emb * mask).sum(dim=0) / denom

        tokens = tokenizer.convert_ids_to_tokens(input_ids[i])

        def find_marker(marker: str):
            for k, tok in enumerate(tokens):
                if marker in tok:
                    return k
            return None

        h_s = find_marker(HEAD_S)
        h_e = find_marker(HEAD_E)
        t_s = find_marker(TAIL_S)
        t_e = find_marker(TAIL_E)

        def span_mean(start, end):
            if start is None or end is None or end <= start + 1:
                return None
            return token_emb[start + 1 : end].mean(dim=0)

        h_vec = span_mean(h_s, h_e) or sent_vec
        t_vec = span_mean(t_s, t_e) or sent_vec

        diff = torch.abs(h_vec - t_vec)
        mul = h_vec * t_vec

        feat = torch.cat([sent_vec, h_vec, t_vec, diff, mul], dim=0)
        feats.append(feat.detach().cpu())

    return feats
def build_xy_dl_batched(
    samples,
    tokenizer,
    model,
    device="cpu",
    max_len=256,
    label2id=None,
    batch_size: int = 16,
):
    X_list, y_list, idx_list = [], [], []
    bad = 0

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]

        try:
            feats = phobert_vectorize_re_sample_tensor_batch(
                batch,
                tokenizer,
                model,
                device=device,
                max_len=max_len,
            )
        except Exception as e:
            bad += len(batch)
            print(f"[SKIP BATCH] {e}")
            continue

        for i, feat in enumerate(feats):
            ex = batch[i]
            X_list.append(feat)
            y_list.append(
                ex["relation"] if label2id is None else label2id[ex["relation"]]
            )
            idx_list.append(start + i)

    if not X_list:
        return torch.empty(0), torch.empty(0), torch.empty(0)

    X = torch.stack(X_list, dim=0)

    if len(y_list) > 0 and isinstance(y_list[0], str):
        raise ValueError("build_xy_dl_batched cần label2id để map relation")

    y = torch.tensor(y_list, dtype=torch.long)
    idx = torch.tensor(idx_list, dtype=torch.long)

    print(f"Skipped {bad} bad samples")
    return X, y, idx
