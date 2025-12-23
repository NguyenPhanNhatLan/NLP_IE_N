import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
    return (
        <div className="min-h-screen flex">
            <Sidebar />
            <div className="flex-1">
                <Topbar />
                <main className="p-4">{children}</main>
            </div>
        </div>
    );
}
