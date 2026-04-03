"use client";

import { isDemoMode } from "@/lib/api";

export function DemoBanner() {
  if (!isDemoMode()) return null;

  return (
    <div className="mb-6 rounded-lg border border-gold-500/30 bg-gold-500/5 px-4 py-3 text-sm text-gold-400">
      <strong>Demo Mode</strong> — Backend API not connected. Showing
      realistic sample data from the soybean oil market. Deploy with{" "}
      <code className="rounded bg-black/30 px-1">docker compose up</code>{" "}
      to see live predictions.
    </div>
  );
}
