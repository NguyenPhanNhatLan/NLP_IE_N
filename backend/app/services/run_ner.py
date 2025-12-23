from typing import List, Tuple
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from app.db.minio import load_ie_config_from_minio
from training.features.build_data.build_ner_dataset import build_ner_dataset
from training.mlflow.load_model import load_pyfunc_from_mlflow
from training.models.ner.config import CRFConfig
from training.models.ner.features import build_features, tokenize_underthesea



pack = load_ie_config_from_minio(key="schema/schema.yml")

EntityType = pack['ENTITY_TYPES']
ENTITY_TYPES = set(pack["ENTITY_TYPES"])          # ví dụ {"NAME","INCI",...}
LABEL_MAP = dict(pack["LABEL_MAP"])    


@dataclass
class Entity:
    id: str
    type: str
    text: str
    start: int
    end: int
    conf: float = 0.75
    
def map_label_from_schema(
    tag_or_type: str,
    *,
    label_map: Dict[str, str],
    entity_types: set[str],
):
    if not tag_or_type:
        return None

    s = str(tag_or_type).strip()

    # bỏ BIO prefix
    if "-" in s:
        p, rest = s.split("-", 1)
        if p.upper() in {"B", "I", "L", "U", "E", "S"}:
            s = rest

    key = s.lower()

    # 1) map qua label_map (lowercase -> ENTITY_TYPE)
    mapped = label_map.get(key)
    if mapped and mapped in entity_types:
        return mapped

    # 2) fallback: đã là entity type
    up = s.upper()
    if up in entity_types:
        return up

    return None


def token_offsets(sentence: str, tokens: List[str]) -> List[Tuple[int, int]]:
    offsets = []
    idx = 0
    for tok in tokens:
        while idx < len(sentence) and sentence[idx].isspace():
            idx += 1
        start = idx
        end = start + len(tok)

        if sentence[start:end] != tok:
            found = sentence.find(tok, idx)
            if found == -1:
                offsets.append((start, start))
                continue
            start = found
            end = start + len(tok)

        offsets.append((start, end))
        idx = end
    return offsets

def map_label(label: str):
    l = label.lower()
    if "-" in l:
        l = l.split("-", 1)[1]

    mapping = {
        "name": "NAME",
        "inci": "INCI",
        "origin": "ORIGIN",
        "benefits": "BENEFITS",
        "skin_concerns": "SKIN_CONCERNS",
    }
    return mapping.get(l)

def decode_entities_with_spans(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    cur_type = None
    cur_start = None
    cur_tokens: List[str] = []
    def flush(end_idx: int):
        nonlocal cur_type, cur_start, cur_tokens
        if cur_type is not None and cur_start is not None and cur_tokens:
            entities.append({
                "type": cur_type,     # raw type from tag (e.g., NAME/INCI...)
                "start": cur_start,   # token start
                "end": end_idx,       # token end (exclusive)
                "text": " ".join(cur_tokens).strip()
            })
        cur_type, cur_start, cur_tokens = None, None, []

    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        tag = tag or "O"
        if tag.startswith("B-"):
            flush(i)
            cur_type = tag[2:]
            cur_start = i
            cur_tokens = [tok]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur_tokens.append(tok)
        else:
            flush(i)

    flush(len(tokens))
    return entities

def run_ner(normalized_sentences: List[str]) -> List[Dict[str, Any]]:
    if not normalized_sentences:
        return []

    crf = load_pyfunc_from_mlflow(
        model_name="ner-model",
        model_version="1",  # hoặc "Production"
    )

    results: List[Dict[str, Any]] = []

    for sent in normalized_sentences:
        tokens = tokenize_underthesea(sent)
        if not tokens:
            results.append({"text": sent, "entities": []})
            continue

        # PyFunc expects tokens
        df_in = pd.DataFrame({"tokens": [tokens]})
        tags = crf.predict(df_in)[0]

        decoded = decode_entities_with_spans(tokens, tags)
        offsets = token_offsets(sent, tokens)

        sent_entities: List[Dict[str, Any]] = []

        for ent in decoded:
            raw_type = str(ent["type"])

            etype = map_label_from_schema(
                raw_type,
                label_map=LABEL_MAP,
                entity_types=ENTITY_TYPES,
            )
            if etype is None:
                continue

            t_start = int(ent["start"])
            t_end = int(ent["end"])
            if t_start < 0 or t_end <= t_start or t_end > len(tokens):
                continue

            start = offsets[t_start][0]
            end = offsets[t_end - 1][1]

            sent_entities.append({
                "start": start,
                "end": end,
                "text": sent[start:end],
                "type": etype,
            })

        results.append({
            "text": sent,
            "entities": sent_entities,
        })

    return results

# normalized_sentences = [
#     'Vitamin C giúp làm sáng da và giảm mụn hiệu quả.',
#     'Thành phần này phù hợp cho da dầu và da mụn.'
# ]

# # ===== CHẠY RIÊNG run_ner =====
# entities = run_ner(normalized_sentences)
# print(entities)
