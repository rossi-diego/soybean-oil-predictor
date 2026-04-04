"use client";

import { isDemoMode } from "@/lib/api";

export function DemoBanner() {
  if (!isDemoMode()) return null;

  return (
    <div className="mb-6 rounded-lg border border-yellow-600/20 bg-yellow-500/5 px-4 py-2.5 text-[13px] text-yellow-500/90 flex items-center gap-2">
      <div className="w-1.5 h-1.5 rounded-full bg-yellow-500 shrink-0" />
      Showing sample market data. Live predictions available when the API is
      connected.
    </div>
  );
}
