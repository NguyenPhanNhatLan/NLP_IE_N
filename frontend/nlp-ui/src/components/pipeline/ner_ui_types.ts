// src/components/pipeline/ner-ui-types.ts
export type NerEntityType = string;

export type NerSegment =
  | { kind: "text"; text: string }
  | {
      kind: "entity";
      text: string;
      type: NerEntityType;
      start: number;
      end: number;
    };

export interface NerUiSentence {
  sentence_id: number;
  text: string;
  segments: NerSegment[];
  legend?: Record<string, number>;
}

export interface NerSentenceEntities {
  text: string;
  entities: Array<{
    start: number;
    end: number;
    text: string;
    type: NerEntityType;
  }>;
}

export interface NerOutput {
  status: number;
  entities: NerSentenceEntities[];
  ui?: NerUiSentence[];
}

/**
 * Đây là format bạn muốn gửi về backend (entities-output)
 * chính là output dạng:
 * [{ text: "...", entities: [{start,end,text,type}, ...] }, ...]
 */
export interface EntitiesOutputSentence {
  text: string;
  entities: Array<{
    start: number;
    end: number;
    text: string;
    type: NerEntityType;
  }>;
}

