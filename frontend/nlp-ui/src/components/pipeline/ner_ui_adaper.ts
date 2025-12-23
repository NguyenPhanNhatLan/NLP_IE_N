// src/components/pipeline/ner-ui-adapter.ts

import {
  EntitiesOutputSentence,
  NerEntityType,
  NerSegment,
  NerUiSentence,
} from "./ner_ui_types";

/**
 * Convert ONE sentence UI segments -> entities-output with auto start/end
 * start/end được tính bằng cursor trên chuỗi nối của segments.
 */
export function uiSentenceToEntitiesOutput(
  s: NerUiSentence
): EntitiesOutputSentence {
  let cursor = 0;
  const entities: EntitiesOutputSentence["entities"] = [];
  const rebuilt: string[] = [];

  for (const seg of s.segments) {
    const t = seg.text ?? "";
    if (seg.kind === "entity") {
      const start = cursor;
      const end = cursor + t.length;
      entities.push({ start, end, text: t, type: seg.type });
      rebuilt.push(t);
      cursor = end;
    } else {
      rebuilt.push(t);
      cursor += t.length;
    }
  }

  // return rebuilt text để offsets luôn đúng nếu có thay đổi text segments
  return { text: rebuilt.join(""), entities };
}

/** Convert ALL sentences ui -> entities-output[] */
export function uiToEntitiesOutput(
  ui: NerUiSentence[]
): EntitiesOutputSentence[] {
  return [...ui]
    .sort((a, b) => a.sentence_id - b.sentence_id)
    .map(uiSentenceToEntitiesOutput);
}

/** Update entity type by sentenceId + segment index (immutable) */
export function updateEntityType(
  ui: NerUiSentence[],
  sentenceId: number,
  segIndex: number,
  nextType: NerEntityType
): NerUiSentence[] {
  return ui.map((s) => {
    if (s.sentence_id !== sentenceId) return s;

    const nextSegs: NerSegment[] = s.segments.map((seg, i) => {
      if (i !== segIndex) return seg;
      if (seg.kind !== "entity") return seg;
      return { ...seg, type: nextType };
    });

    return { ...s, segments: nextSegs };
  });
}

/**
 * Delete entity: chuyển entity segment -> text segment
 * (để câu vẫn giữ nguyên nội dung)
 */
export function deleteEntity(
  ui: NerUiSentence[],
  sentenceId: number,
  segIndex: number
): NerUiSentence[] {
  return ui.map((s) => {
    if (s.sentence_id !== sentenceId) return s;

    const seg = s.segments[segIndex];
    if (!seg || seg.kind !== "entity") return s;

    const nextSegs: NerSegment[] = s.segments.map((x, i) => {
      if (i !== segIndex) return x;
      return { kind: "text", text: seg.text };
    });

    return { ...s, segments: nextSegs };
  });
}
