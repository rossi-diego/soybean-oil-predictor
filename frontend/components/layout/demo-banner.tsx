"use client";

import { isDemoMode } from "@/lib/api";

export function DemoBanner() {
  if (!isDemoMode()) return null;

  return (
    <div className="mb-6 rounded-lg border border-gold-500/30 bg-gold-500/5 px-4 py-3 text-sm text-gold-400">
      <strong>Interactive Preview</strong> — Displaying sample data from the
      soybean oil futures market. Full predictions available when connected
      to the FastAPI backend.
    </div>
  );
}
