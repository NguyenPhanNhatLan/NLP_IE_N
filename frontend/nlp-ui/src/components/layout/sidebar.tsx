"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Separator } from "@/components/ui/separator";

const nav = [
    { href: "/pipeline", label: "Pipeline" },
    { href: "/models", label: "Models" },
    { href: "/settings", label: "Settings" },
];

export function Sidebar() {
    const pathname = usePathname();
    return (
        <aside className="w-64 border-r bg-background">
            <div className="px-4 py-4">
                <div className="text-lg font-semibold">NLP/ML Dashboard</div>
                <div className="text-sm text-muted-foreground">IE • NER • RE</div>
            </div>
            <Separator />
            <nav className="p-2">
                {nav.map((item) => {
                    const active = pathname === item.href || pathname?.startsWith(item.href + "/");
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={cn(
                                "block rounded-lg px-3 py-2 text-sm hover:bg-accent",
                                active && "bg-accent font-medium"
                            )}
                        >
                            {item.label}
                        </Link>
                    );
                })}
            </nav>
        </aside>
    );
}
