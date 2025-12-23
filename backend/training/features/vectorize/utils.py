from training.features.build_data.build_re_dataset import build_re_datasets

# python -m training.features.svm_re.utils
HEAD_S, HEAD_E = "[HEAD]", "[/HEAD]"
TAIL_S, TAIL_E = "[TAIL]", "[/TAIL]"

def build_label_maps(samples):
    labels = sorted({s["relation"] for s in samples})
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in id2label.items()}
    return id2label, label2id


def span_to_tuple(span) -> tuple[int, int]:
    if isinstance(span, dict):
        return int(span["start"]), int(span["end"])
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return int(span[0]), int(span[1])
    raise TypeError(f"Unsupported span format: {type(span)} -> {span}")

#cell 93
def soft_match(a: str, b: str) -> bool:
    na = " ".join(a.strip().split())
    nb = " ".join(b.strip().split())
    return na == nb


def inject_markers_segmented(sample: dict):
    sep = " [SEP] "

    if "sentence_a" in sample and "sentence_b" in sample:
        base_a = sample.get("sentence_a") or ""
        base_b = sample.get("sentence_b") or ""
    else:
        # support both keys: sentence or text
        base_a = sample.get("sentence") or sample.get("text") or ""
        base_b = ""

    if not base_a:
        raise ValueError("Missing text: expected 'sentence' or 'text' (or sentence_a/sentence_b)")
    
    if "head_span" in sample and "tail_span" in sample:
        hspan = sample["head_span"]
        tspan = sample["tail_span"]

        hseg = hspan.get("seg", "A") if isinstance(hspan, dict) else "A"
        tseg = tspan.get("seg", "A") if isinstance(tspan, dict) else "A"

        hs, he = span_to_tuple(hspan)
        ts, te = span_to_tuple(tspan)

    else:
        # Schema B (current pipeline): head_start/head_end/tail_start/tail_end (+ optional seg)
        hs = int(sample.get("head_start"))
        he = int(sample.get("head_end"))
        ts = int(sample.get("tail_start"))
        te = int(sample.get("tail_end"))

        # if you later add seg info, respect it; otherwise default A
        hseg = str(sample.get("head_seg", "A"))
        tseg = str(sample.get("tail_seg", "A"))

    # --------- 3) Inject helper (giữ nguyên logic của bạn) ---------
    def inject_one(sentence: str, s: int, e: int, S: str, E: str, expected_text: str | None = None):
        if sentence is None:
            raise ValueError("Missing sentence segment")
        if not (0 <= s < e <= len(sentence)):
            raise ValueError(f"Span out of range for segment: {s}-{e} len={len(sentence)}")

        actual = sentence[s:e]
        if expected_text is not None and not soft_match(actual, expected_text):
            print(f"[WARN] span mismatch: '{actual}' != '{expected_text}'")

        return sentence[:s] + f"{S} " + sentence[s:e] + f" {E}" + sentence[e:]


    a = base_a
    b = base_b

    edits_a = []
    edits_b = []

    if hseg == "A":
        edits_a.append(("HEAD", hs, he, HEAD_S, HEAD_E, sample.get("head_text")))
    else:
        edits_b.append(("HEAD", hs, he, HEAD_S, HEAD_E, sample.get("head_text")))

    if tseg == "A":
        edits_a.append(("TAIL", ts, te, TAIL_S, TAIL_E, sample.get("tail_text")))
    else:
        edits_b.append(("TAIL", ts, te, TAIL_S, TAIL_E, sample.get("tail_text")))

    edits_a.sort(key=lambda x: x[1], reverse=True)
    edits_b.sort(key=lambda x: x[1], reverse=True)

    for _, s, e, S, E, exp in edits_a:
        a = inject_one(a, s, e, S, E, exp)
    for _, s, e, S, E, exp in edits_b:
        b = inject_one(b, s, e, S, E, exp)

    if b:
        return a.strip() + sep + b.strip()
    return a
