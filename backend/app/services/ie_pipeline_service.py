import traceback
import uuid
from collections import defaultdict
from typing import Dict, Any, List

import torch
from app.db.minio import load_ie_config_from_minio
from app.services.decode_re import decode_re_rule_grouped
from app.services.format_utils import to_ui_cards_v1
from app.services.providers import get_phobert
from app.services.re_service import build_re_inputs_dl_from_ner_list, build_re_inputs_ml_from_ner_list, build_re_pairs_and_samples_from_ner_list, build_re_samples_from_ner_list_dl
from app.services.run_ner import run_ner
from app.services.run_re_and_rules import  predict_re_from_mlflow_xy
from app.services.text_service import normalized
from training.features.vectorize.re_vectorize import phobert_vectorize_re_sample_tensor

def group_relations_by_sentence(
    triples: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    grouped = defaultdict(lambda: {
        "sentence_id": None,
        "sentence": None,
        "relations": []
    })

    for t in triples:
        sid = t["sentence_id"]

        if grouped[sid]["sentence_id"] is None:
            grouped[sid]["sentence_id"] = str(uuid.uuid4())[:5]
            grouped[sid]["sentence"] = t["sentence"]

        grouped[sid]["relations"].append({
            "head_text": t["head_text"],
            "head_type": t["head_type"],
            "tail_text": t["tail_text"],
            "tail_type": t["tail_type"],
            "relation": t["relation"],
            "pair_id": t["pair_id"],
        })

    return list(grouped.values())

def normalize_ie_return(result: Any):
    if isinstance(result, dict):
        return result

    if isinstance(result, (tuple, list)):
        if len(result) == 2 and isinstance(result[0], dict) and isinstance(result[1], list):
            # chỉ có (schema_config, formatted_sections)
            return {
                "normalized_text": "",
                "ner_entities": [],
                "pairs": [],
                "re_relations": [],
                "schema_config": result[0],
                "formatted_sections": result[1],
            }
        if len(result) == 3 and isinstance(result[0], dict):
            full_payload = result[0]
            full_payload["schema_config"] = result[1]
            full_payload["formatted_sections"] = result[2]
            return full_payload

    raise TypeError(
        "Unsupported run_ie_pipeline output. "
        "Expected dict or (schema_config, formatted_sections) or (full_payload, schema_config, formatted_sections)."
    )

from typing import Any, Dict, List, Tuple
from typing import Any
import torch

def run_ie_pipeline(payload: Any):
    # 1) normalize payload object -> dict
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    elif hasattr(payload, "dict"):
        payload = payload.dict()

    text = payload["text"]

    # 2) load config
    ie_conf = load_ie_config_from_minio(key="schema/config.yml", bucket="nlp-ie")

    # 3) NER
    normalized_text = normalized(text)
    entities_output = run_ner(normalized_text)  # expected: list[{"text","entities"}]

    # 4) model info
    model_name = payload.get("model_name", "re_svm_phobert")
    model_version = str(payload.get("model_version", "1"))
    loader = str(payload.get("loader", "sklearn")).lower().strip()

    # alias loader rõ ràng
    if loader in {"torch", "pytorch", "mlp"}:
        loader = "pyfunc"

    # 5) init encoder/tokenizer ONCE (cả ML và DL đều cần)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, encoder = get_phobert(device=device)


    try:
        if loader == "pyfunc":
            print("[DBG] calling build_re_inputs_dl_from_ner_list", flush=True)
            X_all, pairs_all = build_re_inputs_dl_from_ner_list(
                entities_output,
                tokenizer=tokenizer,
                encoder=encoder,
                device=device,
                max_len=int(payload.get("max_len", 256)),
            )
        else:
            print("[DBG] calling build_re_inputs_ml_from_ner_list", flush=True)
            X_all, pairs_all = build_re_inputs_ml_from_ner_list(
                entities_output,
                tokenizer=tokenizer,
                encoder=encoder,
                device=device,
                max_len=int(payload.get("max_len", 256)),
            )

        print("[DBG] build done", "X_shape=", getattr(X_all, "shape", None),
            "X_size=", getattr(X_all, "size", None), "pairs=", len(pairs_all), flush=True)

    except Exception as e:
        print("[ERR] build X/pairs failed:", repr(e), flush=True)
        print(traceback.format_exc(), flush=True)
        raise

    if X_all is None or getattr(X_all, "size", 0) == 0 or not pairs_all:
        return [], model_name, model_version, []
    re_output = predict_re_from_mlflow_xy(
        X=X_all,
        pairs_info=pairs_all,
        text=text,
        model_name=model_name,
        model_version=model_version,
        loader=loader,
    )

    re_formatted = group_relations_by_sentence(re_output)
    structured = decode_re_rule_grouped(
        re_formatted,
        ie_conf,
        allowed_subject_types=None,
        strict_schema=False,
        return_structured=True,
    )
    cards = to_ui_cards_v1(structured, ie_conf)

    return structured, model_name, model_version, cards


def run_ie_pipeline_from_fe(payload: Any):
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    elif hasattr(payload, "dict"):
        payload = payload.dict()

    text = payload["text"]
    entities_output = payload.get("entities_output") or []
    ie_conf = load_ie_config_from_minio(key="schema/config.yml", bucket="nlp-ie") 

    model_name = payload.get("model_name", "re_svm_phobert")
    model_version = str(payload.get("model_version", "1"))
    loader = str(payload.get("loader", "sklearn")).lower().strip()

    if loader in {"torch", "pytorch", "mlp"}:
        loader = "pyfunc"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, encoder = get_phobert(device=device)

    try:
        if loader == "pyfunc":
            print("[DBG] calling build_re_inputs_dl_from_ner_list", flush=True)
            X_all, pairs_all = build_re_inputs_dl_from_ner_list(
                entities_output,
                tokenizer=tokenizer,
                encoder=encoder,
                device=device,
                max_len=int(payload.get("max_len", 256)),
            )
        else:
            print("[DBG] calling build_re_inputs_ml_from_ner_list", flush=True)
            X_all, pairs_all = build_re_inputs_ml_from_ner_list(
                entities_output,
                tokenizer=tokenizer,
                encoder=encoder,
                device=device,
                max_len=int(payload.get("max_len", 256)),
            )

            print("[DBG] build done", "X_shape=", getattr(X_all, "shape", None),
                "X_size=", getattr(X_all, "size", None), "pairs=", len(pairs_all), flush=True)

    except Exception as e:
        print("[ERR] build X/pairs failed:", repr(e), flush=True)
        print(traceback.format_exc(), flush=True)
        raise


    if X_all is None or getattr(X_all, "size", 0) == 0 or not pairs_all:
        return {
            "model": {"name": model_name, "version": model_version},
            "entity_pairs": pairs_all or [],
            "re_output": [],
            "re_formatted": [],
            "structured": [],
            "ui": [],
        }

    # 5) Predict
    re_output = predict_re_from_mlflow_xy(
        X=X_all,
        pairs_info=pairs_all,
        text=text,
        model_name=model_name,
        model_version=model_version,
        loader=loader,     
    )

    # 6) Post-process
    re_formatted = group_relations_by_sentence(re_output)
    structured = decode_re_rule_grouped(
        re_formatted,
        ie_conf,
        allowed_subject_types=None,
        strict_schema=False,
        return_structured=True,
    )
    cards = to_ui_cards_v1(structured, ie_conf)

    return {
        "model": {"name": model_name, "version": model_version},
        "entity_pairs": pairs_all,
        "re_output": re_output,
        "re_formatted": re_formatted,
        "structured": structured,
        "ui": cards,
    }