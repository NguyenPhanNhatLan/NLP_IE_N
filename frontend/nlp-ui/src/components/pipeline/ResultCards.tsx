"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { IeUiCard, IeUiSection } from "@/lib/pipeline/types";

function SectionView({ s }: { s: IeUiSection }) {
    const items = s.items ?? [];
    const display = s.display ?? "bullets";

    if (items.length === 0) {
        return (
            <div className="text-sm text-muted-foreground">
                (Không có dữ liệu)
            </div>
        );
    }

    if (display === "inline") {
        return <div className="text-sm">{items.join(", ")}</div>;
    }

    // default: bullets
    return (
        <ul className="list-disc ml-5 text-sm space-y-1">
            {items.map((it, idx) => (
                <li key={`${s.id}_${idx}`}>{it}</li>
            ))}
        </ul>
    );
}

export function ResultCards({
    cards,
    model,
}: {
    cards: IeUiCard[];
    model?: { name: string; version: string };
}) {
    return (
        <Card className="rounded-2xl">
            <CardHeader>
                <CardTitle className="text-base">
                    Kết quả trích xuất
                    {model ? (
                        <span className="ml-2 text-xs text-muted-foreground">
                            (RE: {model.name} v{model.version})
                        </span>
                    ) : null}
                </CardTitle>
            </CardHeader>

            <CardContent className="space-y-4">
                {cards.length === 0 ? (
                    <div className="text-sm text-muted-foreground">
                        Chưa có cards để hiển thị (hãy chạy bước RE Predict).
                    </div>
                ) : (
                    cards.map((c) => (
                        <Card key={c.id} className="rounded-2xl">
                            <CardHeader>
                                <div className="flex items-center gap-2">
                                    <CardTitle className="text-lg">{c.title}</CardTitle>
                                    {c.tag ? (
                                        <span className="text-xs px-2 py-1 rounded-full bg-muted text-muted-foreground">
                                            {c.tag}
                                        </span>
                                    ) : null}
                                </div>

                                {c.subtitle ? (
                                    <div className="text-sm text-muted-foreground">{c.subtitle}</div>
                                ) : null}

                                {c.source?.sentence ? (
                                    <div className="text-xs text-muted-foreground line-clamp-2">
                                        {c.source.sentence}
                                    </div>
                                ) : null}
                            </CardHeader>

                            <CardContent className="space-y-4">
                                {(c.sections ?? [])
                                    .slice()
                                    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
                                    .map((s) => (
                                        <div key={s.id} className="space-y-2">
                                            <div className="text-sm font-medium">{s.title}</div>
                                            <SectionView s={s} />
                                        </div>
                                    ))}
                            </CardContent>
                        </Card>
                    ))
                )}
            </CardContent>
        </Card>
    );
}
