from collections import Counter, defaultdict
import os
import random
import numpy as np
from app.db.minio import read_yaml_from_minio
from training.features.build_data.utils_data import fetch_data_from_mongo
from dotenv import load_dotenv
from itertools import product
import re

load_dotenv()
def normalize_text(t):
    return re.sub(r"\s+", " ", t.lower().strip())

def build_relation_schema(cfg: dict):
    relation_schema = {
        rel: (v["head"], v["tail"])
        for rel, v in cfg["relations"].items()
    }
    valid_tail_types = set(cfg["valid_tail_types"])
    return relation_schema, valid_tail_types


RELATION_SCHEMA = {
    "has_inci_name": ("NAME", "INCI"),
    "has_origin": ("NAME", "ORIGIN"),
    "has_benefits": ("NAME", "BENEFITS"),
    "targets_skin_concerns": ("NAME", "SKIN_CONCERNS"),
}

VALID_TAIL_TYPES = {"INCI", "ORIGIN", "BENEFITS", "SKIN_CONCERNS"}

SEP = " [SEP] "
def add_cross_sentence_no_relation(
    dataset,
    *,
    relation_schema,
    valid_tail_types,
    target_no_relation_ratio=0.5,
    seed=42,
    max_add_per_head=60,
    max_trials=300000,
    group_key_candidates=("source_id", "doc_id", "task_id", "id"),
):
    rng = random.Random(seed)

    existing_pairs = set()   
    positive_pairs = set()

    for ex in dataset:
        hn = normalize_text(ex["head_text"])
        tn = normalize_text(ex["tail_text"])
        tt = ex["tail_type"]
        key = (hn, tn, tt)
        existing_pairs.add(key)
        if ex["relation"] != "no_relation":
            positive_pairs.add(key)

    # 2) cần thêm bao nhiêu
    cur_no = sum(1 for x in dataset if x["relation"] == "no_relation")
    cur_total = len(dataset)
    if not (0.0 < target_no_relation_ratio < 1.0):
        return dataset

    need = int(max(0, (target_no_relation_ratio * cur_total - cur_no) / (1 - target_no_relation_ratio)))
    if need <= 0:
        return dataset

    def get_group_id(ex):
        for k in group_key_candidates:
            if k in ex and ex[k] is not None:
                return f"{k}:{ex[k]}"
        return None

    # 3) pools theo group
    heads_by_group = defaultdict(list)
    tails_by_group = defaultdict(list)

    for ex in dataset:
        g = get_group_id(ex) or f"sent:{normalize_text(ex.get('sentence',''))}"

        if ex.get("head_type") == "NAME":
            heads_by_group[g].append({
                "g": g,
                "sentence": ex["sentence"],
                "sentence_norm": normalize_text(ex["sentence"]),
                "head_text": ex["head_text"],
                "head_norm": normalize_text(ex["head_text"]),
                "head_span": ex.get("head_span"),
            })

        if ex.get("tail_type") in valid_tail_types:
            tails_by_group[g].append({
                "g": g,
                "sentence": ex["sentence"],
                "sentence_norm": normalize_text(ex["sentence"]),
                "tail_text": ex["tail_text"],
                "tail_norm": normalize_text(ex["tail_text"]),
                "tail_type": ex["tail_type"],
                "tail_span": ex.get("tail_span"),
            })

    head_groups = [g for g, lst in heads_by_group.items() if lst]
    tail_groups = [g for g, lst in tails_by_group.items() if lst]
    if not head_groups or not tail_groups:
        return dataset

    added = []
    per_head = defaultdict(int)

    trials = 0
    while len(added) < need and trials < max_trials:
        trials += 1

        g_h = rng.choice(head_groups)
        g_t = rng.choice(tail_groups)
        if g_h == g_t:
            continue

        h = rng.choice(heads_by_group[g_h])
        t = rng.choice(tails_by_group[g_t])
        if h["sentence_norm"] == t["sentence_norm"]:
            continue

        if not h["head_norm"] or not t["tail_norm"]:
            continue
        if h["head_norm"] == t["tail_norm"]:
            continue

        pair_key = (h["head_norm"], t["tail_norm"], t["tail_type"])
        if pair_key in positive_pairs:
            continue
        if pair_key in existing_pairs:
            continue

        per_head[h["head_norm"]] += 1
        if per_head[h["head_norm"]] > max_add_per_head:
            continue

        existing_pairs.add(pair_key)
        added.append({
            "sentence_a": h["sentence"].strip(),
            "sentence_b": t["sentence"].strip(),
            "head_text": h["head_text"],
            "head_type": "NAME",
            "head_span": {"seg": "A", "start": (h["head_span"] or {}).get("start"), "end": (h["head_span"] or {}).get("end")},
            "tail_text": t["tail_text"],
            "tail_type": t["tail_type"],
            "tail_span": {"seg": "B", "start": (t["tail_span"] or {}).get("start"), "end": (t["tail_span"] or {}).get("end")},
            "relation": "no_relation",
            "meta": {"source": "cross_sentence", "head_group": g_h, "tail_group": g_t},
        })

    return dataset + added

def parse_re(tasks,relation_schema, valid_tail_types, add_no_relation=True, max_no_relation_per_sentence=200):
    dataset = []
    GLOBAL_SEEN = set()  # (task_id, head_norm, tail_norm, tail_type, relation)

    for task in tasks:
        text = task["data"]["text"]
        task_id = task.get("id")
        sentence_norm = normalize_text(text)

        annotations = task.get("annotations", [])
        if not annotations:
            continue
        results = annotations[0].get("result", [])

        # 1) entities with spans
        entities = {}
        for r in results:
            if r.get("type") != "labels":
                continue
            val = r.get("value", {})
            labels = val.get("labels") or []
            if not labels:
                continue
            entities[r["id"]] = {
                "text": val.get("text", ""),
                "label": labels[0],
                "start": val.get("start"),
                "end": val.get("end"),
            }
        if not entities:
            continue

        # 2) positives
        positive_pairs = set()  # (head_norm, tail_norm, tail_type)

        for r in results:
            if r.get("type") != "relation":
                continue

            rel_label = (r.get("labels") or [None])[0]
            if rel_label not in relation_schema:
                continue

            h_id, t_id = r.get("from_id"), r.get("to_id")
            h_ent, t_ent = entities.get(h_id), entities.get(t_id)
            if not h_ent or not t_ent:
                continue

            exp_h, exp_t = relation_schema[rel_label]

            # ép chiều nếu kéo ngược
            if h_ent["label"] != exp_h:
                h_ent, t_ent = t_ent, h_ent

            if h_ent["label"] != exp_h or t_ent["label"] != exp_t:
                continue

            h_norm = normalize_text(h_ent["text"])
            t_norm = normalize_text(t_ent["text"])

            seen_key = (task_id, h_norm, t_norm, t_ent["label"], rel_label)
            if seen_key in GLOBAL_SEEN:
                continue
            GLOBAL_SEEN.add(seen_key)

            positive_pairs.add((h_norm, t_norm, t_ent["label"]))

            dataset.append({
                "task_id": task_id,
                "sentence": text,
                "head_text": h_ent["text"],
                "head_type": h_ent["label"],
                "head_span": {"seg": "A", "start": h_ent["start"], "end": h_ent["end"]},
                "tail_text": t_ent["text"],
                "tail_type": t_ent["label"],
                "tail_span": {"seg": "A", "start": t_ent["start"], "end": t_ent["end"]},
                "relation": rel_label,
            })

        # 3) in-sentence no_relation with spans
        if add_no_relation:
            name_ents = [e for e in entities.values() if e["label"] == "NAME"]
            tail_ents = [e for e in entities.values() if e["label"] in valid_tail_types]

            count = 0
            for h, t in product(name_ents, tail_ents):
                h_norm = normalize_text(h["text"])
                t_norm = normalize_text(t["text"])
                if h_norm == t_norm:
                    continue

                if (h_norm, t_norm, t["label"]) in positive_pairs:
                    continue

                seen_key = (task_id, h_norm, t_norm, t["label"], "no_relation")
                if seen_key in GLOBAL_SEEN:
                    continue
                GLOBAL_SEEN.add(seen_key)

                dataset.append({
                    "task_id": task_id,
                    "sentence": text,
                    "head_text": h["text"],
                    "head_type": "NAME",
                    "head_span": {"seg": "A", "start": h["start"], "end": h["end"]},
                    "tail_text": t["text"],
                    "tail_type": t["label"],
                    "tail_span": {"seg": "A", "start": t["start"], "end": t["end"]},
                    "relation": "no_relation",
                })

                count += 1
                if count >= max_no_relation_per_sentence:
                    break

    return dataset

#Câu lệnh chạy file này trong terminal: python -m training.features.build_data.build_re_dataset
# t đổi tên hàm này để import 
def build_re_datasets(schema_key: str = "label-config.yml"):
    relation_schema =RELATION_SCHEMA
    valid_tail_types = VALID_TAIL_TYPES
    print(relation_schema)
    print(valid_tail_types)
    
    col = fetch_data_from_mongo(
        mongo_uri=os.getenv("MONGO_URI"),
        db_name=os.getenv("MONGO_DB_NAME"),
        collection=os.getenv("RE_LABELED_COLLECTION"),
    )

    cursor= col.find({})
    tasks = list(cursor)

    re_dataset = parse_re(tasks, relation_schema=relation_schema, valid_tail_types=valid_tail_types)
    
    dataset = add_cross_sentence_no_relation(
    re_dataset,
    relation_schema=relation_schema,
    valid_tail_types=valid_tail_types,
    target_no_relation_ratio=0.5,  # hoặc 0.45
    max_add_per_head=60,
    )
    return dataset


    
