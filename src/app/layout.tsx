import type { Metadata } from "next";
import { PlatformProvider } from "@/context/PlatformContext";
import LayoutShell from "@/components/layout/LayoutShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "GARUDA - Traffic Violation Intelligence Platform",
  description: "Production-grade, event-driven traffic control, review, and analytics dashboard.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning={true}>
      <body>
        <PlatformProvider>
          <LayoutShell>{children}</LayoutShell>
        </PlatformProvider>
      </body>
    </html>
  );
}
