from __future__ import annotations
from typing import Dict, List, Optional, Any
from training.models.ner.labels import Entity

def _tag_type(tag: str) -> Optional[str]:
    if not tag or tag == "O":
        return None
    if "-" not in tag:
        return None
    return tag.split("-", 1)[1].strip()

def decode_bio_to_entities(
    text: str,
    tokens: List[str],
    spans: List[tuple[int, int]],
    tags: List[str],
    *,
    marginals: Optional[List[Dict[str, float]]] = None,
    id_prefix: str = "e",
) -> List[Entity]:
    out: List[Entity] = []

    cur_type: Optional[str] = None
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    confs: List[float] = []

    def flush():
        nonlocal cur_type, cur_start, cur_end, confs
        if cur_type is None or cur_start is None or cur_end is None:
            return
        ent_text = text[cur_start:cur_end]
        conf = (sum(confs) / len(confs)) if confs else 0.75  # default 0.75 nếu không có marginals
        out.append(Entity(
            id=f"{id_prefix}{len(out)+1}",
            type=cur_type,
            text=ent_text,
            start=cur_start,
            end=cur_end,
            conf=float(conf),
        ))
        cur_type, cur_start, cur_end, confs = None, None, None, []

    for i, tag in enumerate(tags):
        typ = _tag_type(tag)
        if typ is None:
            flush()
            continue

        prefix = tag.split("-", 1)[0]  # B/I
        s, e = spans[i]

        token_conf = 0.0
        if marginals is not None and i < len(marginals):
            token_conf = float(marginals[i].get(tag, 0.0))

        if prefix == "B" or cur_type is None or typ != cur_type:
            flush()
            cur_type = typ
            cur_start = s
            cur_end = e
            confs = [token_conf] if marginals is not None else []
        else:
            # I- tiếp tục
            cur_end = e
            if marginals is not None:
                confs.append(token_conf)

    flush()
    return out

def format_output(text: str, entities: List[Entity], *, mode: str = "json") -> Any:
    if mode == "list":
        return entities
    if mode == "json":
        return {"text": text, "entities": [e.to_dict() for e in entities]}
    raise ValueError("mode must be 'list' or 'json'")
