import re
from typing import List
from app.schema import Entity, Pair, Relation

def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def run_ner(text: str) -> List[Entity]:
    entities: List[Entity] = []
    i = 1

    def add_entity(span_text: str, etype: str):
        nonlocal i
        start = text.lower().find(span_text.lower())
        if start >= 0:
            end = start + len(span_text)
            entities.append(
                Entity(
                    id=f"e{i}",
                    type=etype,  # type: ignore
                    text=text[start:end],
                    start=start,
                    end=end,
                    conf=0.9,
                )
            )
            i += 1

    add_entity("Vitamin C", "NAME")
    add_entity("rau củ quả", "ORIGIN")
    return entities

def build_pairs(entities: List[Entity], allow_same_span: bool = False) -> List[Pair]:
    heads = [e for e in entities if e.type == "NAME"]
    tails = [e for e in entities if e.type in {"INCI", "ORIGIN", "BENEFITS", "SKIN_CONCERNS"}]

    pairs: List[Pair] = []
    for h in heads:
        for t in tails:
            if not allow_same_span and (h.start == t.start and h.end == t.end):
                continue
            if h.id == t.id:
                continue
            pairs.append(Pair(head_id=h.id, tail_id=t.id))
    return pairs

def run_re(text: str, entities: List[Entity], pairs: List[Pair], re_model: str) -> List[Relation]:
    ent_map = {e.id: e for e in entities}
    out: List[Relation] = []

    for p in pairs:
        h = ent_map.get(p.head_id)
        t = ent_map.get(p.tail_id)
        if not h or not t:
            continue

        if t.type == "ORIGIN":
            rel = "has_origin"
            conf = 0.85
        elif t.type == "INCI":
            rel = "has_inci_name"
            conf = 0.80
        elif t.type == "BENEFITS":
            rel = "has_benefits"
            conf = 0.75
        elif t.type == "SKIN_CONCERNS":
            rel = "targets_skin_concerns"
            conf = 0.70
        else:
            rel = "NO_RELATION"
            conf = 0.55

        conf = min(1.0, conf + (0.02 if re_model == "phobert" else 0.0))

        out.append(
            Relation(
                head_id=p.head_id,
                tail_id=p.tail_id,
                relation=rel,  # type: ignore
                confidence=conf,
            )
        )
    return out
