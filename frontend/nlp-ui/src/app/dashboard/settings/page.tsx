import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function SettingsPage() {
    return (
        <Card className="rounded-2xl">
            <CardHeader>
                <CardTitle>Settings</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
                Cấu hình backend URL, auth token, profile môi trường (dev/staging/prod)…
            </CardContent>
        </Card>
    );
}
