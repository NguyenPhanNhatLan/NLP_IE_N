"use client";

import React, { useMemo, useRef, useState } from "react";
import { NerEntityType, NerSentenceEntities } from "./ner_ui_types";


// ===== helpers =====
type RenderSeg =
    | { kind: "text"; text: string }
    | { kind: "entity"; text: string; type: string; start: number; end: number };

function buildSegments(text: string, entities: NerSentenceEntities["entities"]): RenderSeg[] {
    const sorted = [...entities].sort((a, b) => a.start - b.start);
    const segs: RenderSeg[] = [];

    let cur = 0;
    for (const e of sorted) {
        const s = Math.max(0, Math.min(text.length, e.start));
        const t = Math.max(0, Math.min(text.length, e.end));
        if (s > cur) segs.push({ kind: "text", text: text.slice(cur, s) });
        segs.push({ kind: "entity", text: text.slice(s, t), type: e.type, start: s, end: t });
        cur = t;
    }
    if (cur < text.length) segs.push({ kind: "text", text: text.slice(cur) });
    return segs;
}

function getOffsetsWithin(container: HTMLElement, range: Range) {
    const pre = range.cloneRange();
    pre.selectNodeContents(container);
    pre.setEnd(range.startContainer, range.startOffset);
    const start = pre.toString().length;
    const len = range.toString().length;
    return { start, end: start + len };
}

// ===== UI styles =====
function typeBadgeClass(t: string) {
    switch (t) {
        case "NAME":
            return "bg-emerald-100 text-emerald-900 border-emerald-200";
        case "INCI":
            return "bg-sky-100 text-sky-900 border-sky-200";
        case "BENEFITS":
            return "bg-amber-100 text-amber-900 border-amber-200";
        case "ORIGIN":
            return "bg-violet-100 text-violet-900 border-violet-200";
        case "SKIN_CONCERNS":
            return "bg-rose-100 text-rose-900 border-rose-200";
        default:
            return "bg-slate-100 text-slate-900 border-slate-200";
    }
}

export function NerSpanEditor({
    sentence,
    entities,
    onChange,
    availableTypes = ["NAME", "INCI", "BENEFITS", "ORIGIN", "SKIN_CONCERNS"],
    mode = "add", // "add" hoặc "replace-overlap"
}: {
    sentence: string;
    entities: NerSentenceEntities["entities"];
    onChange: (next: NerSentenceEntities["entities"]) => void;
    availableTypes?: NerEntityType[];
    mode?: "add" | "replace-overlap";
}) {
    const ref = useRef<HTMLDivElement | null>(null);
    const [selectedType, setSelectedType] = useState<NerEntityType>("BENEFITS");

    const segs = useMemo(() => buildSegments(sentence, entities), [sentence, entities]);

    return (
        <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
                <div className="text-xs text-muted-foreground">Type</div>
                <select
                    className="border rounded-md px-2 py-1 text-sm"
                    value={selectedType}
                    onChange={(e) => setSelectedType(e.target.value)}
                >
                    {availableTypes.map((t) => (
                        <option key={t} value={t}>
                            {t}
                        </option>
                    ))}
                </select>

                <div className="text-xs text-muted-foreground">
                    Kéo bôi đen → thả chuột để tạo entity (start/end tự tính)
                </div>
            </div>

            <div
                ref={ref}
                className="leading-relaxed text-base border rounded-xl p-3"
                onMouseUp={() => {
                    const el = ref.current;
                    const sel = window.getSelection();
                    if (!el || !sel || sel.rangeCount === 0) return;

                    const range = sel.getRangeAt(0);
                    if (range.collapsed) return;
                    if (!el.contains(range.commonAncestorContainer)) return;

                    const { start, end } = getOffsetsWithin(el, range);
                    const text = sentence.slice(start, end);

                    sel.removeAllRanges();
                    if (!text.trim()) return;

                    let next = [...entities];

                    if (mode === "replace-overlap") {
                        // remove entities overlapped with new span
                        next = next.filter((e) => end <= e.start || start >= e.end);
                    }

                    next.push({ start, end, text, type: selectedType });
                    next.sort((a, b) => a.start - b.start);

                    onChange(next);
                }}
            >
                {segs.map((s, i) =>
                    s.kind === "text" ? (
                        <span key={i}>{s.text}</span>
                    ) : (
                        <span
                            key={i}
                            className={`mx-px inline-flex items-center rounded-md border px-1.5 py-0.5 ${typeBadgeClass(
                                s.type
                            )}`}
                            title={`${s.type} [${s.start}, ${s.end}]`}
                        >
                            {s.text}
                        </span>
                    )
                )}
            </div>
        </div>
    );
}
