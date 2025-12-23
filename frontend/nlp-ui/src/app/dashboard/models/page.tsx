import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function ModelsPage() {
    return (
        <Card className="rounded-2xl">
            <CardHeader>
                <CardTitle>Models</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
                Ở đây bạn có thể list models từ MLflow Registry (version, stage, tags) và chọn model active cho NER/RE.
            </CardContent>
        </Card>
    );
}
