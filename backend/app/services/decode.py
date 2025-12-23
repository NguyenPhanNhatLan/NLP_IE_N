from typing import List, Dict, Any, Tuple
from collections import defaultdict

RELATIONS = ["has_inci_name", "has_benefits", "has_origin", "targets_skin_concerns"]

SECTION_TITLES = {
    "has_inci_name": "Tên khoa học",
    "has_benefits": "Công dụng",
    "has_origin": "Nguồn gốc",
    "targets_skin_concerns": "Vấn đề da",
}

def decode_re_rule(triples: List[Dict[str, Any]]) -> List[str]:
    """
    Input:
      [{'head_text':..., 'head_type':..., 'tail_text':..., 'tail_type':..., 'relation':...}, ...]
    """
    grouped: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(
        lambda: {r: [] for r in RELATIONS}
    )

    # collect
    for t in triples:
        h_text = (t.get("head_text") or "").strip()
        h_type = (t.get("head_type") or "").strip()  # giữ nếu bạn cần, UI text có thể không dùng
        tail_text = (t.get("tail_text") or "").strip()
        rel = (t.get("relation") or "").strip()

        if not h_text or not h_type or not tail_text:
            continue
        if rel not in RELATIONS:
            continue

        grouped[(h_text, h_type)][rel].append(tail_text)

    # build UI text
    cards: List[str] = []
    for (h_text, _h_type), rels in grouped.items():
        lines = [h_text]

        for rel in RELATIONS:
            items = rels[rel]
            if not items:
                continue

            # dedup while keeping order
            seen = set()
            uniq = []
            for x in items:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)

            lines.append("")  # blank line
            lines.append(f"{SECTION_TITLES[rel]}:")
            lines += [f"• {x}" for x in uniq]

        cards.append("\n".join(lines).strip())

    return cards


# demo
inp = [
    {"head_text": "Niacinamide", "head_type": "NAME", "tail_text": "Nicotinamide", "tail_type": "INCI", "relation": "has_inci_name"},
    {"head_text": "Niacinamide", "head_type": "NAME", "tail_text": "giảm mụn", "tail_type": "BENEFITS", "relation": "has_benefits"},
    {"head_text": "Niacinamide", "head_type": "NAME", "tail_text": "làm đều màu da", "tail_type": "BENEFITS", "relation": "has_benefits"},
    {"head_text": "Niacinamide", "head_type": "NAME", "tail_text": "da dầu", "tail_type": "SKIN_CONCERNS", "relation": "targets_skin_concerns"},
]

for card in decode_re_rule(inp):
    print("-----")
    print(card)
