import os
import boto3
import yaml
from dotenv import load_dotenv
from typing import Any, Dict, Tuple, Set, List

from io import BytesIO

load_dotenv()

def get_s3_client():
    try:
        return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_ACCESS_KEY"),
        region_name="us-east-1",   
    )

    except Exception as e:
        raise ValueError("Something was wrong with MiniO", e)
    
def upload_file(file_path: str, object_name: str, bucket: str = 'nlp-ie'):
    s3 = get_s3_client()
    bucket = bucket

    s3.upload_file(
        Filename=file_path,
        Bucket=bucket,
        Key=object_name
    )
    print(f"Uploaded {file_path} → s3://{bucket}/{object_name}")

def read_yaml_from_minio(key: str, bucket: str = "nlp-ie") -> dict:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)

    content_bytes: bytes = obj["Body"].read()
    content_text = content_bytes.decode("utf-8")

    cfg = yaml.safe_load(content_text)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a dict/object.")
    return cfg

def parse_ie_config(cfg: Dict[str, Any]) -> Tuple[
    Dict[str, Tuple[str, str]],  # RELATION_SCHEMA
    Set[str],                    # VALID_TAIL_TYPES
    Set[str],                    # ENTITY_TYPES
    Dict[str, str],              # LABEL_MAP
    List[str],                   # RELATIONS (ordered)
    Dict[str, str],              # SECTION_TITLES
]:
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict")

    entity_types = set(cfg.get("entity_types", []))
    if not entity_types:
        raise ValueError("entity_types is empty")

    valid_tail_types = set(cfg.get("valid_tail_types", []))
    if not valid_tail_types:
        raise ValueError("valid_tail_types is empty")
    if not valid_tail_types.issubset(entity_types):
        raise ValueError("valid_tail_types must be subset of entity_types")

    label_map = {str(k).strip().lower(): str(v).strip() for k, v in (cfg.get("label_map") or {}).items()}
    if not label_map:
        raise ValueError("label_map is empty")

    relations_raw = cfg.get("relations") or {}
    if not isinstance(relations_raw, dict) or not relations_raw:
        raise ValueError("relations is empty")

    relation_schema: Dict[str, Tuple[str, str]] = {}
    for rel, spec in relations_raw.items():
        head = (spec or {}).get("head")
        tail = (spec or {}).get("tail")
        if not head or not tail:
            raise ValueError(f"relation {rel} missing head/tail")
        if head not in entity_types or tail not in entity_types:
            raise ValueError(f"relation {rel} has invalid entity type")
        relation_schema[rel] = (head, tail)

    # RELATIONS: dùng relations_order nếu có, không thì theo keys relations
    order = cfg.get("relations_order") or list(relation_schema.keys())
    relations: List[str] = [r for r in order if r in relation_schema]
    if not relations:
        relations = list(relation_schema.keys())

    # SECTION_TITLES map theo relations (đảm bảo đầy đủ key)
    titles_cfg = cfg.get("section_titles") or {}
    section_titles = {r: str(titles_cfg.get(r, r)) for r in relations}

    return relation_schema, valid_tail_types, entity_types, label_map, relations, section_titles

def load_ie_config_from_minio(key: str, bucket: str ="nlp-ie") -> Dict:
    cfg = read_yaml_from_minio(key=key, bucket=bucket)

    (
        relation_schema,
        valid_tail_types,
        entity_types,
        label_map,
        relations,
        section_titles,
    ) = parse_ie_config(cfg)

    return {
        "cfg": cfg,
        "RELATION_SCHEMA": relation_schema,
        "VALID_TAIL_TYPES": valid_tail_types,
        "ENTITY_TYPES": entity_types,
        "LABEL_MAP": label_map,
        "RELATIONS": relations,            
        "SECTION_TITLES": section_titles
    }



# upload_file(file_path="/Users/nhatlan/Documents/nlp/config.yml", object_name='label-config.yml')
# pack = load_ie_config_from_minio(key="schema/schema.yml")
# print(pack["LABEL_MAP"])

