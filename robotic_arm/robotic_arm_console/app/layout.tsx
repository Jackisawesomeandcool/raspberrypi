import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const description = "Maker Arm v0.6 的实时数字孪生、关节遥测、视觉流和 Pick & Place 控制台。";

  return {
    title: "AXIS — Maker Arm Mission Control",
    description,
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
    openGraph: {
      title: "AXIS — Maker Arm Mission Control",
      description,
      images: [{ url: `${origin}/og.png`, width: 1680, height: 936 }],
    },
    twitter: {
      card: "summary_large_image",
      title: "AXIS — Maker Arm Mission Control",
      description,
      images: [`${origin}/og.png`],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
