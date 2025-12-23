import { useCallback } from "react";
import type {
    ModelConfig,
    StepKey,
    StepOutputs,
    NormalizeOutput,
    NerOutput,
    PredictOutput,
    TabKey,
} from "@/lib/pipeline/types";
import { STEPS } from "@/lib/pipeline/steps";
import { buildPayload, fetchStepData } from "@/lib/pipeline/utils";


export function usePipelineRunner(args: {
    input: string;
    outputs: StepOutputs;
    setOutputs: React.Dispatch<React.SetStateAction<StepOutputs>>;
    modelConfig: ModelConfig;
    setLoading: React.Dispatch<React.SetStateAction<StepKey | null>>;
    setActiveTab: React.Dispatch<React.SetStateAction<TabKey>>;
}) {
    const { input, outputs, setOutputs, modelConfig, setLoading, setActiveTab } =
        args;

    const runPipeline = useCallback(
        async (onlyStep?: StepKey) => {
            const stepsToRun = onlyStep
                ? STEPS.filter((s) => s.key === onlyStep)
                : STEPS;

            // local snapshot used + updated sequentially
            let current = { ...outputs };

            for (const step of stepsToRun) {
                setLoading(step.key);

                try {
                    const payload = buildPayload(step.key, input, current, modelConfig);
                    const data = await fetchStepData(step.endpoint, payload);

                    if (step.key === "predict")
                        current = { ...current, predict: data as PredictOutput };
                    if (step.key === "ner")
                        current = { ...current, ner: data as NerOutput };
                    if (step.key === "normalize")
                        current = { ...current, normalize: data as NormalizeOutput };

                    setOutputs(current);
                    setActiveTab(step.key);
                } finally {
                    setLoading(null);
                }
            }

            if (!onlyStep) setActiveTab("result");
        },
        [input, outputs, modelConfig, setOutputs, setActiveTab, setLoading]
    );

    return { runPipeline };
}
