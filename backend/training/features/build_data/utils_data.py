import logging
import re
import json
from typing import Any, Dict, List, Tuple
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError, InvalidName
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_data_from_mongo(mongo_uri: str, collection: str, db_name: str):
    logging.info(f"Input params: mongo_uri={mongo_uri}, db={db_name}, collection={collection}")
    try:
        client = MongoClient(mongo_uri)
        logging.info("MongoClient created successfully.")

        db = client[db_name]
        logging.info(f"Database '{db_name}' accessed successfully.")
        
        col = db[collection]
        logging.info(f"Collection '{collection}' ready to use.")

        return col

    except ConfigurationError as e:
        logging.error(f"MongoDB Configuration Error: {e}")
    except ConnectionFailure as e:
        logging.error(f"Cannot connect to MongoDB server: {e}")
    except InvalidName as e:
        logging.error(f"Invalid database or collection name: {e}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

    logging.warning("Returning None due to previous errors.")
    return None

def convert_to_bio(tasks):
    bio_samples = []
    for task in tasks:
        text = task.get("data", {}).get("text")
        if not text:
            continue

        spans = _get_spans(task)
        sample = _find_bio_in_text(text, spans)

        # lọc sample rỗng
        if sample and sample.get("tokens"):
            bio_samples.append(sample)

    return bio_samples      
    
        
def _get_spans(task: dict) -> list[tuple[int, int, list[str]]]:
    spans = []

    annotations = task.get("annotations")
    if not annotations:
        return spans

    results = annotations[0].get("result")
    if not results:
        return spans

    for anno in results:
        value = anno.get("value")
        if not value:
            continue

        start = value.get("start")
        end = value.get("end")
        labels = value.get("labels")

        if start is None or end is None:
            continue
        if not labels or not isinstance(labels, list):
            continue

        # bỏ qua label O
        if "O" in labels:
            continue

        spans.append((start, end, labels))

    return spans


def _find_bio_in_text(text: str, spans):
    tokens = []
    labels = []

    # sort spans để ưu tiên span bắt đầu sớm hơn, dài hơn
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))

    prev_tag = "O"
    prev_type = None

    # Tokenization: bắt token theo "word-ish", giữ dấu ngoặc/đặc biệt tách riêng để giảm lệch biên
    # Ví dụ: "(Centella", "extract)" sẽ tách thành "(", "Centella", "extract", ")"
    pattern = r"\w+|[^\w\s]"  # word OR single punctuation

    for m in re.finditer(pattern, text, flags=re.UNICODE):
        tok = m.group()
        s_tok, e_tok = m.start(), m.end()

        # bỏ qua pure whitespace (pattern này thường không match whitespace)
        if tok.strip() == "":
            continue

        tag = "O"
        chosen_type = None

        # tìm span mà token overlap
        best = None
        best_overlap = 0

        for s_span, e_span, lab_list in spans:
            # overlap length
            overlap = min(e_tok, e_span) - max(s_tok, s_span)
            if overlap > 0:  # có giao nhau
                # chọn span có overlap lớn nhất (để tránh token dính 2 span)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = (s_span, e_span, lab_list)

        if best is not None:
            lab = best[2][0]  # label list -> lấy label đầu
            chosen_type = lab

            # RULE QUAN TRỌNG: xác định B/I dựa trên tag trước đó (không dựa start_tok==start)
            if prev_tag == "O" or prev_type != chosen_type:
                tag = f"B-{chosen_type}"
            else:
                tag = f"I-{chosen_type}"

        tokens.append(tok)
        labels.append(tag)

        prev_tag = tag
        prev_type = chosen_type if tag != "O" else None

    return {"tokens": tokens, "labels": labels}
        
    
def split_train_valid(
    bio_samples: List[Dict[str, Any]], 
    valid_ratio: float = 0.1,
    seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not bio_samples:
        return [], []
    
    train_samples, valid_samples = train_test_split(
        bio_samples,
        test_size=valid_ratio,
        random_state=seed,
        shuffle=True
    )
    
    return train_samples, valid_samples


def save_ner_tuples_to_jsonl(dataset, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in dataset:
            # Format B: dict
            if isinstance(sentence, dict) and "tokens" in sentence and "labels" in sentence:
                tokens = sentence["tokens"]
                labels = sentence["labels"]

            # Format A: list tuple
            else:
                tokens = [tok for tok, label in sentence]
                labels = [label for tok, label in sentence]

            obj = {"tokens": tokens, "labels": labels}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return output_path

