"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AVAILABLE_MODELS, DEFAULT_MODEL_CONFIG } from "@/lib/pipeline/steps";
import type { ModelConfig } from "@/lib/pipeline/types";

export function ReModelPicker({
    modelConfig,
    setModelConfig,
}: {
    modelConfig: ModelConfig;
    setModelConfig: React.Dispatch<React.SetStateAction<ModelConfig>>;
}) {
    const reModels = AVAILABLE_MODELS.re;

    const reVersions =
        reModels.find((m) => m.name === modelConfig.re.model_name)?.versions ?? reModels[0].versions;

    const setReModel = (name: string) => {
        const versions = reModels.find((m) => m.name === name)?.versions ?? ["1"];
        setModelConfig((prev) => ({ ...prev, re: { model_name: name, model_version: versions[0] } }));
    };

    return (
        <Card className="rounded-2xl">
            <CardHeader>
                <CardTitle className="text-base">RE model selection</CardTitle>
            </CardHeader>

            <CardContent className="space-y-4">
                <div className="space-y-2">
                    <div className="text-sm font-medium">RE model</div>
                    <div className="flex flex-wrap gap-2">
                        {reModels.map((m) => (
                            <Button
                                key={m.name}
                                type="button"
                                variant={m.name === modelConfig.re.model_name ? "default" : "secondary"}
                                onClick={() => setReModel(m.name)}
                            >
                                {m.name}
                            </Button>
                        ))}
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                        <div className="text-xs text-muted-foreground">Version</div>
                        {reVersions.map((v) => (
                            <Button
                                key={v}
                                type="button"
                                size="sm"
                                variant={v === modelConfig.re.model_version ? "default" : "secondary"}
                                onClick={() =>
                                    setModelConfig((prev) => ({ ...prev, re: { ...prev.re, model_version: v } }))
                                }
                            >
                                {v}
                            </Button>
                        ))}
                    </div>
                </div>

                <div className="text-xs text-muted-foreground">
                    Current RE: <b>{modelConfig.re.model_name}</b> v<b>{modelConfig.re.model_version}</b>
                </div>

                <Button type="button" variant="secondary" onClick={() => setModelConfig(DEFAULT_MODEL_CONFIG)}>
                    Reset RE model
                </Button>
            </CardContent>
        </Card>
    );
}
