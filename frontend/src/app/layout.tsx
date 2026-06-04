import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Customer Care Bot",
  description:
    "Agentic customer care bot — resolves problems by taking real actions.",
};

// No web-font fetch: we use a warm Calibri/system stack (set in globals.css),
// which keeps first paint instant and matches the design reference.
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
