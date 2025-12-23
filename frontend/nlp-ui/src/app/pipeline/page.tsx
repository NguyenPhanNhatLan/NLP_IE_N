/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { startTransition, useCallback, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { PlayCircle } from "lucide-react";

import type { NerSentenceEntities, StepKey, StepOutputs, TabKey, ModelConfig, IeUiCard } from "@/lib/pipeline/types";
import { DEFAULT_MODEL_CONFIG, INITIAL_OUTPUTS, MODEL_CONFIG_STORAGE_KEY, STEPS } from "@/lib/pipeline/steps";
import { safeParseModelConfig } from "@/lib/pipeline/utils";

import { usePipelineRunner } from "@/hooks/usePipeline";
import { usePipelineResult } from "@/hooks/usePipelineResult";

import { NerTab } from "@/components/pipeline/NerTab";
import { StepOutputEditor } from "@/components/pipeline/StepOutputEditor";
import { ResultCards } from "@/components/pipeline/ResultCards";
import { ReModelPicker } from "@/components/pipeline/ReModelPicker";

export default function PipelinePage() {
    const [input, setInput] = useState("");
    const [outputs, setOutputs] = useState<StepOutputs>(INITIAL_OUTPUTS);
    const [loading, setLoading] = useState<StepKey | null>(null);
    const [activeTab, setActiveTab] = useState<TabKey>("normalize");
    const [editedEntitiesOutput, setEditedEntitiesOutput] = useState<NerSentenceEntities[] | null>(null);
    const [resultOverride, setResultOverride] = useState<{
        structured?: unknown;
        cards?: IeUiCard[];
        model?: { name: string; version: string };
    } | null>(null);
    // tránh reset ngay lần mount đầu tiên (StrictMode dev chạy effect 2 lần)
    const didMountRef = useRef(false);

    useEffect(() => {
        if (!didMountRef.current) {
            didMountRef.current = true;
            return;
        }

        startTransition(() => {
            setResultOverride(null);
            setOutputs(INITIAL_OUTPUTS);
            setEditedEntitiesOutput(null);
            setActiveTab("normalize");
        });

    }, [input]);

    useEffect(() => {
        if (!outputs.ner?.entities) return;
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setEditedEntitiesOutput(outputs.ner.entities);
    }, [outputs.ner?.entities]);

    const [modelConfig, setModelConfig] = useState<ModelConfig>(() => {
        if (typeof window === "undefined") return DEFAULT_MODEL_CONFIG;
        return safeParseModelConfig(localStorage.getItem(MODEL_CONFIG_STORAGE_KEY)) ?? DEFAULT_MODEL_CONFIG;
    });

    useEffect(() => {
        if (typeof window !== "undefined") {
            localStorage.setItem(MODEL_CONFIG_STORAGE_KEY, JSON.stringify(modelConfig));
        }
    }, [modelConfig]);

    const result = usePipelineResult(input, outputs);

    const { runPipeline } = usePipelineRunner({
        input,
        outputs,
        setOutputs,
        modelConfig,
        setLoading,
        setActiveTab,
    });



    const handleOutputChange = useCallback((key: StepKey, value: unknown) => {
        setOutputs((prev) => ({ ...prev, [key]: value as any }));
    }, []);

    return (
        <div className="space-y-4">
            <Card className="rounded-2xl m-4">
                <CardHeader>
                    <CardTitle className="text-2xl font-semibold">Trích xuất thông tin sản phẩm</CardTitle>
                </CardHeader>

                <CardContent className="space-y-4">
                    <Textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        rows={5}
                        placeholder="Nhập đoạn mô tả tại đây ..."
                        className="h-60 text-2xl leading-relaxed"
                    />

                    <div className="flex items-center justify-between gap-2">
                        <div className="flex flex-wrap gap-2">
                            {STEPS.map((step) => (
                                <Button
                                    key={step.key}
                                    onClick={() => runPipeline(step.key)}
                                    disabled={!!loading}
                                    variant="secondary"
                                    className="hover:bg-gray-300"
                                >
                                    {loading === step.key ? "Running..." : step.label}
                                </Button>
                            ))}
                        </div>

                        <Button
                            onClick={() => runPipeline()}
                            disabled={!!loading}
                            className="ml-auto"
                            title="Chạy toàn bộ pipeline"
                        >
                            <PlayCircle className="h-4 w-4 mr-2" />
                            {loading ? "Running..." : "Run pipeline"}
                        </Button>
                    </div>

                    <div className="text-sm text-muted-foreground ml-1">
                        Tip: bạn có thể sửa output JSON ở từng tab để &apos;feedback&apos; vào bước sau.
                    </div>

                    <ReModelPicker modelConfig={modelConfig} setModelConfig={setModelConfig} />
                </CardContent>
            </Card>

            <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as TabKey)} className="w-full ml-5">
                <TabsList className="flex flex-wrap h-auto">
                    {STEPS.map((step) => (
                        <TabsTrigger key={step.key} value={step.key}>
                            {step.label}
                        </TabsTrigger>
                    ))}
                    <TabsTrigger value="result">Result</TabsTrigger>
                </TabsList>

                {STEPS.map((step) => (
                    <TabsContent key={step.key} value={step.key} className="mt-4 space-y-4">
                        {step.key === "ner" && (
                            <NerTab
                                input={input}
                                outputs={outputs}
                                editedEntitiesOutput={editedEntitiesOutput}
                                setEditedEntitiesOutput={setEditedEntitiesOutput}
                                modelConfig={modelConfig}
                                setOutputs={setOutputs}
                                onResultOverride={(v) => setResultOverride(v)}
                                setActiveTab={setActiveTab} // optional
                            />
                        )}

                        <StepOutputEditor
                            title={step.label}
                            value={outputs[step.key]}
                            onChange={(v) => handleOutputChange(step.key, v)}
                        />
                    </TabsContent>
                ))}

                <TabsContent value="result" className="mt-4 space-y-4">
                    <ResultCards
                        cards={resultOverride?.cards ?? result.cards}
                        model={resultOverride?.model ?? result.model}
                    />
                    <StepOutputEditor
                        title="Structured (final)"
                        value={resultOverride?.structured ?? result}
                        onChange={() => { }}
                        readOnly
                    />

                </TabsContent>

            </Tabs>

            {/* optional: show what normalized text currently is */}
            {/* <div className="text-xs text-muted-foreground ml-5">
        Normalized: {getNormalizedText(input, outputs)}
      </div> */}
        </div>
    );
}
