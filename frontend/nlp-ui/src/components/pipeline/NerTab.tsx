/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { NerSpanEditor } from "@/components/pipeline/NerSpanEditor";
import type {
  NerSentenceEntities,
  StepOutputs,
  PredictOutput,
  IeUiCard,
  TabKey,
  ModelConfig,
} from "@/lib/pipeline/types";
import { buildSaveCorrectionsPayload, fetchStepData } from "@/lib/pipeline/utils";

type BackendEnvelope<T> = { status: number; message?: string; data: T };

type NerSaveData = {
  model?: { name: string; version: string };
  entity_pairs?: unknown;
  re_output?: unknown;
  re_formatted: PredictOutput["data"]; // quan trọng: đúng shape predict.data
  structured?: unknown;
  cards?: IeUiCard[];
};

export function NerTab({
  input,
  editedEntitiesOutput,
  setEditedEntitiesOutput,
  modelConfig,
  setOutputs,
  onResultOverride,
  setActiveTab, // optional
}: {
  input: string;
  outputs: StepOutputs;
  editedEntitiesOutput: NerSentenceEntities[] | null;
  setEditedEntitiesOutput: React.Dispatch<React.SetStateAction<NerSentenceEntities[] | null>>;
  modelConfig: ModelConfig;
  setOutputs: React.Dispatch<React.SetStateAction<StepOutputs>>;
  onResultOverride: (v: {
    structured?: unknown;
    cards?: IeUiCard[];
    model?: { name: string; version: string };
  }) => void;
  setActiveTab?: React.Dispatch<React.SetStateAction<TabKey>>;
}) {
  const canSave = !!editedEntitiesOutput;

  const [saveResp, setSaveResp] = React.useState<unknown>(null);
  const [saving, setSaving] = React.useState(false);

  return (
    <Card className="rounded-2xl">
      <CardHeader>
        <CardTitle className="text-base">NER highlight (drag để sửa span)</CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {!editedEntitiesOutput ? (
          <div className="text-sm text-muted-foreground">
            Chưa có dữ liệu NER (hãy chạy bước NER).
          </div>
        ) : (
          <div className="space-y-4">
            {editedEntitiesOutput.map((s, sid) => (
              <Card key={sid} className="rounded-2xl">
                <CardHeader>
                  <CardTitle className="text-base">Sentence #{sid}</CardTitle>
                </CardHeader>

                <CardContent className="space-y-3">
                  <NerSpanEditor
                    sentence={s.text}
                    entities={s.entities}
                    mode="replace-overlap"
                    onChange={(nextEntities) => {
                      setEditedEntitiesOutput((prev) => {
                        if (!prev) return prev;
                        const copy = [...prev];
                        copy[sid] = { ...copy[sid], entities: nextEntities };
                        return copy;
                      });
                    }}
                  />
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        <div className="text-xs text-muted-foreground">
          Kéo bôi đen để tạo entity mới → chọn type → Save corrections để gửi backend.
        </div>

        <Button
          variant="secondary"
          disabled={!canSave || saving}
          className="hover:bg-gray-400 transition"
          onClick={async () => {
            if (!editedEntitiesOutput) return;

            const payload = buildSaveCorrectionsPayload({
              input,
              editedEntitiesOutput,
              modelConfig,
            });
            console.log(payload)

            setSaving(true);
            try {
              const resp = (await fetchStepData(
                "/api/ie/from-fe",
                payload
              )) as BackendEnvelope<NerSaveData>;

              setSaveResp(resp);

              const data = (resp as any)?.data;

              if (!data) return;

              // 1) ✅ set lại outputs.predict = re_formatted
              setOutputs((prev) => ({
                ...prev,
                predict: {
                  status: 200,
                  step: "result",
                  message: "OK",
                  data: data.re_formatted,
                } as any,
              }));

          
              onResultOverride({
                structured: data.structured,
                cards: data.ui.cards,
                model: data.model,
              });

              // 3) optional: nhảy sang tab result luôn
              setActiveTab?.("result");
            } finally {
              setSaving(false);
            }
          }}
        >
          {saving ? "Saving..." : "Save corrections"}
        </Button>

        {saveResp ? (
          <div className="rounded-xl border p-3">
            <div className="text-xs text-muted-foreground mb-2">Save response</div>
            <pre className="text-xs overflow-auto">{JSON.stringify(saveResp, null, 2)}</pre>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
