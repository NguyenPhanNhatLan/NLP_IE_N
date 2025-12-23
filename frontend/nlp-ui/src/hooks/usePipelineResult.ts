/* eslint-disable @typescript-eslint/no-explicit-any */
import { StepOutputs, PipelineResult } from "@/lib/pipeline/types";
import { getNormalizedText, pickCards, pickPredictRelations } from "@/lib/pipeline/utils";
import { useMemo } from "react";

export function usePipelineResult(input: string, outputs: StepOutputs): PipelineResult {
    return useMemo(() => {
        const normalizedText = getNormalizedText(input, outputs); // ✅ luôn string

        const predict = outputs.predict;

        const cards = pickCards(predict); // ✅ robust (sửa pickCards ở utils)
        const model =
            predict && (predict as any)?.data?.model_name
                ? { name: (predict as any).data.model_name, version: (predict as any).data.model_version }
                : undefined;

        return {
            input,
            normalized_text: normalizedText,
            entities: outputs.validate?.entities ?? outputs.ner?.entities ?? [],
            relations: pickPredictRelations(predict),
            cards,
            model,
            raw_outputs: outputs,
        };
    }, [input, outputs]);
}
