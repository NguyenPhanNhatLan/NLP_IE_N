from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def decode_re_rule_grouped(
    grouped_by_sentence: List[Dict[str, Any]],
    ie_conf: Dict[str, Any],
    *,
    allowed_subject_types: Optional[Tuple[str, ...]] = ("NAME",),  # set None to disable
    drop_self_loop: bool = True,
    strict_schema: bool = True,      # ✅ thêm flag
    return_structured: bool = False,
):
    RELATIONS = ie_conf["RELATIONS"]
    SECTION_TITLES = ie_conf["SECTION_TITLES"]
    RELATION_SCHEMA = ie_conf["RELATION_SCHEMA"]

    structured: List[Dict[str, Any]] = []

    for sent_pack in grouped_by_sentence:
        sid = sent_pack.get("sentence_id", 0)
        sentence = (sent_pack.get("sentence") or "").strip()
        rel_list = sent_pack.get("relations") or []

        subjects = defaultdict(lambda: {r: [] for r in RELATIONS})

        for t in rel_list:
            h_text = (t.get("head_text") or "").strip()
            h_type = (t.get("head_type") or "").strip()
            tail_text = (t.get("tail_text") or "").strip()
            tail_type = (t.get("tail_type") or "").strip()
            rel = (t.get("relation") or "").strip()

            if not h_text or not h_type or not tail_text or not tail_type:
                continue
            if rel not in RELATIONS:
                continue
            
            if h_type == "BENEFITS" and tail_type == "BENEFITS":
                continue

            # ✅ schema check optional
            if strict_schema:
                exp_head, exp_tail = RELATION_SCHEMA.get(rel, ("", ""))
                if (h_type, tail_type) != (exp_head, exp_tail):
                    continue

            # ✅ subject type filter optional
            if allowed_subject_types is not None and h_type not in allowed_subject_types:
                continue

            if drop_self_loop and h_text.lower() == tail_text.lower():
                continue

            subjects[(h_text, h_type)][rel].append(tail_text)

        subjects_out = []
        for (subj_text, subj_type), rels_map in subjects.items():
            sections = []
            for rel in RELATIONS:
                items = _dedup_keep_order(rels_map.get(rel, []))
                if not items:
                    continue
                sections.append({
                    "relation": rel,
                    "title": SECTION_TITLES.get(rel, rel),
                    "items": items
                })

            if sections:
                subjects_out.append({
                    "subject_text": subj_text,
                    "subject_type": subj_type,
                    "sections": sections
                })

        if not subjects_out:
            continue

        subjects_out.sort(key=lambda x: x["subject_text"].lower())
        structured.append({
            "sentence_id": sid,
            "sentence": sentence,
            "subjects": subjects_out
        })

    if return_structured:
        return structured

    cards: List[str] = []
    for s in structured:
        lines = []
        if s["sentence"]:
            lines.append(s["sentence"])
            lines.append("")

        for subj in s["subjects"]:
            lines.append(subj["subject_text"])
            for sec in subj["sections"]:
                lines.append(f"{sec['title']}:")
                lines += [f"• {x}" for x in sec["items"]]
                lines.append("")
            lines.append("")

        card = "\n".join(lines).strip()
        if card:
            cards.append(card)

    return cards
