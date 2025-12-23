"use client"; 
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

export function StepOutputEditor({
    title,
    value,
    onChange,
    readOnly = false,
}: {
    title: string;
    value: unknown;
    onChange: (v: unknown) => void;
    readOnly?: boolean;
}) {
    const [isEdited, setIsEdited] = useState(false);

    const valueText = useMemo(() => (value ? JSON.stringify(value, null, 2) : ""), [value]);
    const [raw, setRaw] = useState<string>(() => valueText);

    // keep raw synced when new value arrives, unless user is currently editing
    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        if (!isEdited) setRaw(valueText);
    }, [valueText, isEdited]);

    const displayValue = isEdited ? raw : valueText;

    const handleApply = useCallback(() => {
        try {
            onChange(JSON.parse(raw));
            setIsEdited(false);
        } catch {
            // ignore parse error
        }
    }, [raw, onChange]);

    const handleReset = useCallback(() => {
        setIsEdited(false);
        setRaw(valueText);
    }, [valueText]);

    return (
        <Card className="rounded-2xl">
            <CardHeader>
                <CardTitle>{title} Output</CardTitle>
            </CardHeader>

            <CardContent className="space-y-3">
                <Textarea
                    value={displayValue}
                    onChange={(e) => {
                        setIsEdited(true);
                        setRaw(e.target.value);
                    }}
                    rows={14}
                    className="font-mono text-xs"
                    placeholder='{"example":"json"}'
                    disabled={readOnly}
                />

                {!readOnly && (
                    <>
                        <div className="flex gap-2">
                            <Button onClick={handleApply}>Apply JSON</Button>
                            <Button variant="secondary" onClick={handleReset}>
                                Reset
                            </Button>
                        </div>

                        <div className="text-xs text-muted-foreground">
                            Dán/sửa JSON rồi bấm <b>Apply JSON</b> để bước sau dùng output bạn chỉnh.
                        </div>
                    </>
                )}
            </CardContent>
        </Card>
    );
}
