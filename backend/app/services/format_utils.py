from typing import Any, Dict, List, Optional
import re


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "item"


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _get_section_title(cfg: Dict[str, Any], relation: str, fallback: str) -> str:
    return (cfg.get("section_titles", {}) or {}).get(relation, fallback) or fallback


def _order_index(cfg: Dict[str, Any], relation: str) -> int:
    order = cfg.get("relations_order") or []
    try:
        return order.index(relation)
    except ValueError:
        return 10_000


def _infer_display(items: List[str]):
    if not items:
        return "bullets"
    avg_len = sum(len(x) for x in items) / max(len(items), 1)
    avg_words = sum(len(x.split()) for x in items) / max(len(items), 1)

    # rule nhẹ, không phụ thuộc relation
    if len(items) <= 6 and avg_len <= 20 and avg_words <= 3:
        return "chips"
    return "bullets"


def _subtitle_from_sections(sections: List[Dict[str, Any]]):
    parts = []
    for sec in sections:
        title = (sec.get("title") or "").strip().lower()
        n = len(sec.get("items") or [])
        if title and n:
            parts.append(f"{n} {title}")
    return " • ".join(parts)


def to_ui_cards_v1(
    raw: List[Dict[str, Any]],
    ie_conf: Dict[str, Any],
    *,
    include_display_hint: bool = True,
    merge_subject_across_sentences: bool = False,
):
    if not isinstance(raw, list):
        raise ValueError(f"raw must be list, got {type(raw)}")

    # -------------------------
    # Helper: build section obj
    # -------------------------
    def build_section(sec_in: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        relation = (sec_in.get("relation") or "").strip()
        fallback_title = (sec_in.get("title") or "").strip() or relation or "Thông tin"
        title = _get_section_title(ie_conf, relation, fallback_title)

        items = sec_in.get("items") or []
        if not isinstance(items, list):
            items = [str(items)]
        items = _uniq_keep_order([str(x) for x in items])
        if not items:
            return None

        out = {
            "id": _slug(title),
            "title": title,
            "relation": relation,
            "items": items,
            "order": _order_index(ie_conf, relation),  # để FE sort dễ nếu muốn
        }
        if include_display_hint:
            out["display"] = _infer_display(items)  # không dựa relation
        return out

    # -------------------------
    # Mode 1: No merge
    # -------------------------
    if not merge_subject_across_sentences:
        cards: List[Dict[str, Any]] = []

        for sent in raw:
            sid = str(sent.get("sentence_id", "")).strip() or "unknown"
            sentence_text = sent.get("sentence") or ""
            subjects = sent.get("subjects") or []

            for subj in subjects:
                subject_text = (subj.get("subject_text") or "").strip() or "(unknown)"
                subject_type = (subj.get("subject_type") or "").strip() or "UNKNOWN"

                sections_out: List[Dict[str, Any]] = []
                for sec in (subj.get("sections") or []):
                    sec_obj = build_section(sec)
                    if sec_obj:
                        sections_out.append(sec_obj)

                # sort by relations_order (stable)
                sections_out.sort(key=lambda s: (s.get("order", 10_000), s.get("title", "")))

                card = {
                    "id": f"sent_{sid}__subj_{_slug(subject_text)}",
                    "title": subject_text,
                    "tag": subject_type,   # giữ type cho UI badge/filter nếu cần
                    "subtitle": _subtitle_from_sections(sections_out),
                    "source": {"sentence_id": sid, "sentence": sentence_text},
                    "sections": sections_out,
                }
                cards.append(card)

        return {"type": "ui_cards_v1", "cards": cards}

    merged: Dict[str, Dict[str, Any]] = {}  # key: subject_text_lower

    for sent in raw:
        sid = str(sent.get("sentence_id", "")).strip() or "unknown"
        sentence_text = sent.get("sentence") or ""
        subjects = sent.get("subjects") or []

        for subj in subjects:
            subject_text = (subj.get("subject_text") or "").strip() or "(unknown)"
            key = subject_text.lower()
            subject_type = (subj.get("subject_type") or "").strip() or "UNKNOWN"

            if key not in merged:
                merged[key] = {
                    "id": f"subj_{_slug(subject_text)}",
                    "title": subject_text,
                    "tag": subject_type,
                    "sources": [],
                    "sections_map": {},  # relation -> section
                }

            merged[key]["sources"].append({"sentence_id": sid, "sentence": sentence_text})

            for sec in (subj.get("sections") or []):
                sec_obj = build_section(sec)
                if not sec_obj:
                    continue

                # merge by relation first (safer), fallback by title
                rel = sec_obj.get("relation") or sec_obj.get("title")
                existing = merged[key]["sections_map"].get(rel)

                if not existing:
                    merged[key]["sections_map"][rel] = sec_obj
                else:
                    # merge items
                    existing["items"] = _uniq_keep_order(existing["items"] + sec_obj["items"])
                    # recompute display if enabled
                    if include_display_hint:
                        existing["display"] = _infer_display(existing["items"])

    cards: List[Dict[str, Any]] = []
    for obj in merged.values():
        sections_out = list(obj["sections_map"].values())
        sections_out.sort(key=lambda s: (s.get("order", 10_000), s.get("title", "")))

        cards.append(
            {
                "id": obj["id"],
                "title": obj["title"],
                "tag": obj["tag"],
                "subtitle": _subtitle_from_sections(sections_out),
                "sources": obj["sources"],  # UI có thể cho expand "Xem câu gốc"
                "sections": sections_out,
            }
        )

    return {"type": "ui_cards_v1", "cards": cards}
