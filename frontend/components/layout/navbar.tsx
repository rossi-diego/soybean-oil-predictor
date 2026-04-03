"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: "📊" },
  { href: "/predict", label: "Predict", icon: "🎯" },
  { href: "/eda", label: "EDA", icon: "🔍" },
  { href: "/models", label: "Models", icon: "🧠" },
  { href: "/monitoring", label: "Monitoring", icon: "📡" },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-[#262626] bg-[#0a0a0a]/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🫘</span>
            <span className="font-bold text-lg gradient-text">
              Soybean Oil Predictor
            </span>
          </div>

          <div className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-brand-500/10 text-brand-400"
                      : "text-gray-400 hover:text-gray-200 hover:bg-white/5"
                  }`}
                >
                  <span className="mr-1.5">{item.icon}</span>
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
