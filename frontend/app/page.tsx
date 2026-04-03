"use client";

import { useEffect, useState } from "react";
import {
  api,
  isDemoMode,
  HealthResponse,
  ModelInfo,
  LivePrice,
  LivePredictionResponse,
} from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";

function StatusCard({
  title,
  value,
  subtitle,
  status,
}: {
  title: string;
  value: string;
  subtitle: string;
  status: "ok" | "warn" | "error";
}) {
  const colors = {
    ok: "border-brand-500/30 bg-brand-500/5",
    warn: "border-gold-500/30 bg-gold-500/5",
    error: "border-red-500/30 bg-red-500/5",
  };

  return (
    <div className={`glass-card p-6 ${colors[status]}`}>
      <p className="text-sm text-gray-400 mb-1">{title}</p>
      <p className="text-2xl font-bold">{value}</p>
      <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
    </div>
  );
}

const COMMODITY_LABELS: Record<string, string> = {
  boc1: "Soybean Oil",
  sc1: "Soybeans",
  smc1: "Soybean Meal",
  lcoc1: "Brent Crude",
  hoc1: "Heating Oil",
  fcpoc1: "Palm Oil",
  rsc1: "Canola",
  zc1: "Corn",
};

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [prices, setPrices] = useState<LivePrice[]>([]);
  const [livePred, setLivePred] = useState<LivePredictionResponse | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    Promise.all([
      api.health().then(setHealth),
      api.listModels().then((data) => setModels(data.models)),
      api.livePrices().then((data) => setPrices(data.prices)),
      api.predictLive().then(setLivePred),
    ]).finally(() => setLoaded(true));
  }, []);

  const demo = loaded && isDemoMode();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold gradient-text">Dashboard</h1>
        <p className="text-gray-400 mt-2">
          BOC1 (Soybean Oil Futures) price forecasting with domain-driven ML.
          Built for commodity traders and risk managers.
        </p>
      </div>

      {demo && <DemoBanner />}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatusCard
          title="BOC1 Prediction"
          value={livePred ? `${livePred.predicted_price.toFixed(2)} c/lb` : "—"}
          subtitle={livePred ? `Model: ${livePred.model_name}` : "Loading..."}
          status={livePred ? "ok" : "warn"}
        />
        <StatusCard
          title="API Status"
          value={demo ? "Demo" : health?.status === "healthy" ? "Healthy" : "Offline"}
          subtitle={`v${health?.version || "?"}`}
          status={demo ? "warn" : health?.status === "healthy" ? "ok" : "error"}
        />
        <StatusCard
          title="Models"
          value={`${models.length}`}
          subtitle={models.length > 0 ? `Active: ${models[0].name}` : "None loaded"}
          status={models.length > 0 ? "ok" : "warn"}
        />
        <StatusCard
          title="Data Status"
          value={health?.data_fresh ? "Fresh" : "No data"}
          subtitle={prices.length > 0 ? `${prices.length} commodities tracked` : ""}
          status={health?.data_fresh ? "ok" : "warn"}
        />
      </div>

      {prices.length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-xl font-semibold mb-4">Live Commodity Prices</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
            {prices.map((p) => (
              <div
                key={p.name}
                className={`p-3 rounded-lg ${
                  p.name === "boc1"
                    ? "bg-brand-500/10 border border-brand-500/30"
                    : "bg-black/20"
                }`}
              >
                <p className="text-xs text-gray-500 uppercase">{p.name}</p>
                <p className="text-lg font-bold">
                  {p.price.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
                <p className="text-xs text-gray-600">
                  {COMMODITY_LABELS[p.name] || p.ticker}
                </p>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-600 mt-3">
            Source: Yahoo Finance
            {prices.length > 0 && ` | Updated: ${new Date(prices[0].timestamp).toLocaleString()}`}
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-card p-6">
          <h2 className="text-xl font-semibold mb-4">Business Context</h2>
          <p className="text-gray-400 text-sm leading-relaxed">
            This model estimates the <strong>BOC1 (soybean oil front-month
            futures)</strong> price using correlated commodity prices as features.
            Vegetable oil traders use BOC1 forecasts to time hedging decisions
            and identify relative value between soybean oil and palm oil (CPO).
          </p>
          <p className="text-gray-400 text-sm leading-relaxed mt-3">
            A 1% improvement in hedge timing on a 10,000 MT position is worth
            approximately <strong className="text-brand-400">$50,000</strong>.
          </p>
        </div>

        <div className="glass-card p-6">
          <h2 className="text-xl font-semibold mb-4">Pipeline Architecture</h2>
          <div className="text-sm text-gray-400 space-y-2 font-mono">
            <p><span className="text-brand-400">INGEST</span> &rarr; yfinance &middot; USDA &middot; FRED</p>
            <p><span className="text-gold-400">BRONZE</span> &rarr; Raw Parquet (append-only)</p>
            <p><span className="text-blue-400">SILVER</span> &rarr; dbt-core + DuckDB (cleaned)</p>
            <p><span className="text-purple-400">GOLD</span> &rarr; dbt-core + DuckDB (features)</p>
            <p><span className="text-red-400">MODEL</span> &rarr; XGBoost &middot; Ridge &middot; statsforecast</p>
            <p><span className="text-cyan-400">SERVE</span> &rarr; FastAPI + MLflow</p>
          </div>
        </div>
      </div>

      <div className="glass-card p-6">
        <h2 className="text-xl font-semibold mb-4">Available Models</h2>
        {models.length === 0 ? (
          <p className="text-gray-500 text-sm">No models loaded.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#262626] text-gray-400">
                  <th className="text-left py-2 px-3">Name</th>
                  <th className="text-left py-2 px-3">Type</th>
                  <th className="text-right py-2 px-3">R2</th>
                  <th className="text-right py-2 px-3">MAE</th>
                  <th className="text-right py-2 px-3">RMSE</th>
                  <th className="text-right py-2 px-3">Dir. Acc.</th>
                  <th className="text-left py-2 px-3">Features</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.name} className="border-b border-[#262626]/50 hover:bg-white/5">
                    <td className="py-2 px-3 font-medium">{m.name}</td>
                    <td className="py-2 px-3 text-gray-400">{m.model_type}</td>
                    <td className="py-2 px-3 text-right text-gray-400">{m.metrics.r2?.toFixed(4) ?? "-"}</td>
                    <td className="py-2 px-3 text-right text-gray-400">{m.metrics.mae?.toFixed(2) ?? "-"}</td>
                    <td className="py-2 px-3 text-right text-gray-400">{m.metrics.rmse?.toFixed(2) ?? "-"}</td>
                    <td className="py-2 px-3 text-right text-gray-400">
                      {m.metrics.directional_accuracy ? `${(m.metrics.directional_accuracy * 100).toFixed(0)}%` : "-"}
                    </td>
                    <td className="py-2 px-3 text-gray-400">{m.n_features}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
