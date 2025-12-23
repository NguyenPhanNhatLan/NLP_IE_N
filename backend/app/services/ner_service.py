from typing import Any, Dict, List
from app.services.run_ner import run_ner
from app.services.text_service import normalized


def run_only_ner(payload: Any):
    if hasattr(payload, "model_dump"):   # Pydantic v2
        payload = payload.model_dump()
    elif hasattr(payload, "dict"):        # Pydantic v1
        payload = payload.dict()



    text = payload["text"]

    normalized_sentences = normalized(text)

    entities_output = run_ner(normalized_sentences)
    ui = ner_outputs_to_ui_segments(entity_outputs=entities_output)
    return entities_output, ui


def ner_sentence_to_segments(doc: Dict[str, Any], sentence_id: int) -> Dict[str, Any]:
    text: str = doc["text"]
    ents: List[Dict[str, Any]] = sorted(doc.get("entities", []), key=lambda e: (e["start"], e["end"]))

    segments: List[Dict[str, Any]] = []
    legend: Dict[str, int] = {}
    cursor = 0

    for e in ents:
        s, ed = int(e["start"]), int(e["end"])
        etype = str(e.get("type", "")).strip()

        # clamp + skip span lỗi
        s = max(0, min(s, len(text)))
        ed = max(0, min(ed, len(text)))
        if ed <= s:
            continue

        legend[etype] = legend.get(etype, 0) + 1

        if s > cursor:
            segments.append({"kind": "text", "text": text[cursor:s]})

        segments.append({
            "kind": "entity",
            "text": text[s:ed],  # lấy theo span cho chắc
            "type": etype,
            "start": s,
            "end": ed,
        })
        cursor = ed

    if cursor < len(text):
        segments.append({"kind": "text", "text": text[cursor:]})

    return {
        "sentence_id": sentence_id,
        "text": text,
        "segments": segments,
        "legend": legend,
    }


def ner_outputs_to_ui_segments(entity_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [ner_sentence_to_segments(doc, i) for i, doc in enumerate(entity_outputs)]

def get_ie_payload(payload: Dict[str, Any]):
    return payload
