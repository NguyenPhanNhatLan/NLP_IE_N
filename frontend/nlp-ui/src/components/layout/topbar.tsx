"use client";

import { Button } from "@/components/ui/button";

export function Topbar() {
    return (
        <header className="h-14 border-b bg-background flex items-center justify-between px-4">
            <div className="text-sm text-muted-foreground">
                Run pipeline step-by-step, edit outputs, and reuse for RE.
            </div>
            <div className="flex gap-2">
                <Button variant="secondary">Docs</Button>
                <Button>New Run</Button>
            </div>
        </header>
    );
}
