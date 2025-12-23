from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from app.services.text_service import normalized
from training.features.vectorize.re_vectorize import phobert_vectorize_re_sample_tensor

MODEL_NAME = "vinai/phobert-base-v2"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME)
_model.eval()

def _as_str(x):
    if isinstance(x, list):
        return " ".join(str(i) for i in x).strip()
    return str(x).strip()

def build_re_pairs_and_samples_from_ner_list(
    ner_list: list[dict],
):
    samples_all: List[Dict[str, Any]] = []
    pairs_all: List[Dict[str, Any]] = []
    global_pair_id = 1

    for sent_idx, item in enumerate(ner_list):
        text = item.get("text", "")
        if not text:
            continue

        entities = prepare_entities_for_re(item)
        if len(entities) < 2:
            continue

        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue

                h, t = entities[i], entities[j]

                pairs_all.append({
                    "head_id": h["id"],
                    "head_text": h["text"],
                    "head_type": h["type"],
                    "tail_id": t["id"],
                    "tail_text": t["text"],
                    "tail_type": t["type"],
                    "pair_id": global_pair_id,
                    "sentence_id": sent_idx,
                    "sentence": text,
                })

                # sample format đúng với vectorizer (inject_markers_segmented expects these keys)
                samples_all.append({
                    "text": text,
                    "head_start": int(h["start"]),
                    "head_end": int(h["end"]),
                    "tail_start": int(t["start"]),
                    "tail_end": int(t["end"]),
                    "head_text":  _as_str(h["text"]),
                    "tail_text":  _as_str(h["text"]),
                    "relation": "no_relation",  # dummy for inference
                    "pair_id": global_pair_id,
                    "sentence_id": sent_idx,
                })

                global_pair_id += 1

    return samples_all, pairs_all

def prepare_entities_for_re(ner_output: dict) -> list[dict]:
    entities = ner_output.get("entities", [])
    prepared = []
    used_ids = set()
    counter = 1

    for ent in entities:
        ent = ent.copy()

        if "id" not in ent:
            new_id = f"e{counter}"
            while new_id in used_ids:
                counter += 1
                new_id = f"e{counter}"
            ent["id"] = new_id
            counter += 1

        used_ids.add(ent["id"])
        ent["text"] = normalized(ent["text"])

        for f in ("id", "type", "text", "start", "end"):
            if f not in ent:
                raise ValueError(f"Missing {f} in entity: {ent}")

        prepared.append(ent)

    return prepared

def vectorize(entities: list, text: str, *, max_len: int = 256):    
    if len(entities) < 2:
        return np.array([]), []

    features = []
    pairs_info = []

    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue

            h = entities[i]
            t = entities[j]

            marked = _inject(text, h, t)

            enc = _tokenizer(
                marked,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding=False,
            )

            with torch.no_grad():
                out = _model(**enc)
                emb = out.last_hidden_state[0]  # (seq_len, hidden)

            mask = enc["attention_mask"][0].unsqueeze(-1).float()
            denom = mask.sum().clamp_min(1.0)
            sent = (emb * mask).sum(0) / denom

            tokens = _tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

            def find_idx(marker: str):
                # cố gắng exact match trước, fallback contains
                for k, tok in enumerate(tokens):
                    if tok == marker:
                        return k
                for k, tok in enumerate(tokens):
                    if marker in tok:
                        return k
                return None

            def span_mean(start_m: str, end_m: str):
                s = find_idx(start_m)
                e = find_idx(end_m)
                if s is None or e is None or e <= s + 1:
                    return sent
                return emb[s + 1 : e].mean(0)

            h_vec = span_mean("[HEAD]", "[/HEAD]")
            t_vec = span_mean("[TAIL]", "[/TAIL]")

            diff = torch.abs(h_vec - t_vec)
            mul = h_vec * t_vec

            feat = torch.cat([sent, h_vec, t_vec, diff, mul], dim=0)
            features.append(feat.detach().cpu().numpy())

            pairs_info.append({
                "head_id": h["id"],
                "head_text": h["text"],
                "head_type": h["type"],
                "tail_id": t["id"],
                "tail_text": t["text"],
                "tail_type": t["type"],
            })

    return (np.vstack(features) if features else np.array([])), pairs_info

def _inject(text: str, head: dict, tail: dict) -> str:
    markers = [
        (head["start"], head["end"], "[HEAD]", "[/HEAD]"),
        (tail["start"], tail["end"], "[TAIL]", "[/TAIL]")
    ]
    markers.sort(key=lambda x: x[0], reverse=True)
    
    result = text
    for start, end, open_m, close_m in markers:
        result = result[:end] + f" {close_m}" + result[end:]
        result = result[:start] + f"{open_m} " + result[start:]
    
    return result

def build_re_samples_from_ner_list_dl(ner_list: list[dict]):
    samples_all, pairs_all = [], []
    global_pair_id = 1

    for sent_idx, item in enumerate(ner_list):
        text = item.get("text", "")
        if not text:
            continue

        entities = prepare_entities_for_re(item)
        if len(entities) < 2:
            continue

        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue

                h, t = entities[i], entities[j]

                pairs_all.append({
                    "head_id": h["id"],
                    "head_text": h["text"],
                    "head_type": h["type"],
                    "tail_id": t["id"],
                    "tail_text": t["text"],
                    "tail_type": t["type"],
                    "pair_id": global_pair_id,
                    "sentence_id": sent_idx,
                    "sentence": text,
                })

                samples_all.append({
                    "text": text,
                    "head_start": int(h["start"]),
                    "head_end": int(h["end"]),
                    "tail_start": int(t["start"]),
                    "tail_end": int(t["end"]),
                    "head_text": h["text"],
                    "tail_text": t["text"],
                    "relation": "no_relation",   # dummy
                    "pair_id": global_pair_id,
                    "sentence_id": sent_idx,
                })

                global_pair_id += 1

    return samples_all, pairs_all

def vectorize_X_ml(
    samples,
    tokenizer,
    encoder,
    device="cpu",
    max_len=256,
) -> np.ndarray:
    X_list = []
    bad = 0

    for ex in samples:
        try:
            feat = phobert_vectorize_re_sample_tensor(
                ex, tokenizer, encoder, device=device, max_len=max_len
            )
            X_list.append(feat.numpy())
        except ValueError as e:
            bad += 1
            print(f"[SKIP] {e} | pair_id={ex.get('pair_id')} | sent_id={ex.get('sentence_id')}", flush=True)

    print(f"Skipped {bad} bad samples", flush=True)
    return np.vstack(X_list).astype(np.float32) if X_list else np.array([], dtype=np.float32)


def vectorize_X_dl_numpy(
    samples,
    tokenizer,
    encoder,
    device="cpu",
    max_len=256,
) -> np.ndarray:
    return vectorize_X_ml(samples, tokenizer, encoder, device=device, max_len=max_len)

def build_re_inputs_ml_from_ner_list(
    ner_list: list[dict],
    *,
    tokenizer,
    encoder,
    device="cpu",
    max_len=256,
) -> tuple[np.ndarray, list[dict]]:
    samples_all, pairs_all = build_re_pairs_and_samples_from_ner_list(ner_list)
    if not samples_all or not pairs_all:
        return np.array([], dtype=np.float32), []

    X_all = vectorize_X_ml(samples_all, tokenizer, encoder, device=device, max_len=max_len)
    if X_all.size == 0:
        return np.array([], dtype=np.float32), []

    # đảm bảo alignment
    if X_all.shape[0] != len(pairs_all):
        raise ValueError(f"X rows ({X_all.shape[0]}) != pairs ({len(pairs_all)})")

    return X_all, pairs_all


def build_re_inputs_dl_from_ner_list(
    ner_list: list[dict],
    *,
    tokenizer,
    encoder,
    device="cpu",
    max_len=256,
) -> tuple[np.ndarray, list[dict]]:
    samples_all, pairs_all = build_re_pairs_and_samples_from_ner_list(ner_list)
    if not samples_all or not pairs_all:
        return np.array([], dtype=np.float32), []

    X_all = vectorize_X_dl_numpy(samples_all, tokenizer, encoder, device=device, max_len=max_len)
    if X_all.size == 0:
        return np.array([], dtype=np.float32), []

    if X_all.shape[0] != len(pairs_all):
        raise ValueError(f"X rows ({X_all.shape[0]}) != pairs ({len(pairs_all)})")

    return X_all, pairs_all

