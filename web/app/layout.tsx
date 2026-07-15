import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "LOB Trading Platform",
  description: "Risk-tiered algorithmic paper-trading dashboard",
};

const nav = [
  { href: "/", label: "Dashboard" },
  { href: "/positions", label: "Positions" },
  { href: "/trades", label: "Trades" },
  { href: "/signals", label: "Signals" },
  { href: "/settings", label: "Settings" },
];

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-black text-gray-100 antialiased">
        <nav className="border-b border-gray-800 bg-gray-950">
          <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3 text-sm">
            {nav.map((n) => (
              <Link key={n.href} href={n.href} className="text-gray-400 hover:text-white">
                {n.label}
              </Link>
            ))}
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}
