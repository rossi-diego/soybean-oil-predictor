"use client";

import { useEffect, useState } from "react";
import {
  api,
  isDemoMode,
  HealthResponse,
  ModelInfo,
  LivePrice,
  LivePredictionResponse,
  PriceHistoryPoint,
} from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

const COMMODITY_LABELS: Record<string, string> = {
  boc1: "Soybean Oil",
  sc1: "Soybeans",
  smc1: "Soybean Meal",
  lcoc1: "Brent Crude",
  hoc1: "Heating Oil",
  fcpoc1: "Palm Oil",
  rsc1: "Wheat",
  zc1: "Corn",
};

const COMMODITY_UNITS: Record<string, string> = {
  boc1: "c/lb",
  sc1: "c/bu",
  smc1: "$/ton",
  lcoc1: "$/bbl",
  hoc1: "$/gal",
  fcpoc1: "GBp",
  rsc1: "c/bu",
  zc1: "c/bu",
};

function computeSpreads(prices: LivePrice[]) {
  const p: Record<string, number> = {};
  prices.forEach((pr) => (p[pr.name] = pr.price));

  const crush =
    p.boc1 && p.smc1 && p.sc1
      ? 11 * p.boc1 + p.smc1 - p.sc1
      : null;

  const oilPalm =
    p.boc1 && p.fcpoc1
      ? p.boc1 - p.fcpoc1 / 100
      : null;

  return { crush, oilPalm };
}

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [prices, setPrices] = useState<LivePrice[]>([]);
  const [livePred, setLivePred] = useState<LivePredictionResponse | null>(null);
  const [history, setHistory] = useState<PriceHistoryPoint[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    Promise.all([
      api.health().then(setHealth),
      api.listModels().then((data) => setModels(data.models)),
      api.livePrices().then((data) => setPrices(data.prices)),
      api.predictLive().then(setLivePred),
      api.priceHistory(90).then((data) => setHistory(data.history)),
    ]).finally(() => setLoaded(true));
  }, []);

  const demo = loaded && isDemoMode();
  const boc1Price = prices.find((p) => p.name === "boc1")?.price;
  const spreads = computeSpreads(prices);
  const predDiff =
    livePred && boc1Price
      ? livePred.predicted_price - boc1Price
      : null;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold gradient-text">
          Soybean Oil Market
        </h1>
        <p className="text-gray-400 mt-2">
          BOC1 front-month futures — live prices, model forecast, and key
          spreads for vegetable oil traders and risk managers.
        </p>
      </div>

      {demo && <DemoBanner />}

      {/* Hero: BOC1 Current vs Prediction */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-card p-6 border-brand-500/30 bg-brand-500/5">
          <p className="text-sm text-gray-400 mb-1">BOC1 Current Price</p>
          <p className="text-4xl font-bold text-brand-400">
            {boc1Price?.toFixed(2) ?? "—"}
          </p>
          <p className="text-sm text-gray-500 mt-1">cents per pound (CBOT)</p>
        </div>
        <div className="glass-card p-6 border-blue-500/30 bg-blue-500/5">
          <p className="text-sm text-gray-400 mb-1">Model Forecast</p>
          <p className="text-4xl font-bold text-blue-400">
            {livePred?.predicted_price.toFixed(2) ?? "—"}
          </p>
          <div className="flex items-center gap-2 mt-1">
            <p className="text-sm text-gray-500">
              {livePred?.model_name ?? "Loading..."}
            </p>
            {predDiff !== null && (
              <span
                className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                  predDiff > 0
                    ? "bg-green-500/20 text-green-400"
                    : predDiff < 0
                      ? "bg-red-500/20 text-red-400"
                      : "bg-gray-500/20 text-gray-400"
                }`}
              >
                {predDiff > 0 ? "+" : ""}
                {predDiff.toFixed(2)} vs current
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Key Spreads */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wide">
            Crush Spread
          </p>
          <p className="text-xl font-bold mt-1">
            {spreads.crush?.toFixed(2) ?? "—"}
          </p>
          <p className="text-xs text-gray-600 mt-1">
            11 x Oil + Meal - Beans
          </p>
        </div>
        <div className="glass-card p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wide">
            Oil / Palm Spread
          </p>
          <p className="text-xl font-bold mt-1">
            {spreads.oilPalm?.toFixed(2) ?? "—"}
          </p>
          <p className="text-xs text-gray-600 mt-1">BOC1 - CPO/100</p>
        </div>
        <div className="glass-card p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wide">
            Active Model
          </p>
          <p className="text-xl font-bold mt-1">
            {models[0]?.name ?? "None"}
          </p>
          <p className="text-xs text-gray-600 mt-1">
            {models[0]?.model_type ?? ""}
            {models[0]?.metrics?.r2
              ? ` | R² ${models[0].metrics.r2.toFixed(3)}`
              : ""}
          </p>
        </div>
        <div className="glass-card p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wide">
            API Status
          </p>
          <p
            className={`text-xl font-bold mt-1 ${
              demo
                ? "text-gold-400"
                : health?.status === "healthy"
                  ? "text-brand-400"
                  : "text-red-400"
            }`}
          >
            {demo ? "Demo" : health?.status === "healthy" ? "Live" : "Offline"}
          </p>
          <p className="text-xs text-gray-600 mt-1">v{health?.version ?? "?"}</p>
        </div>
      </div>

      {/* Actual vs Predicted Chart */}
      {history.length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold mb-1">
            BOC1 Price — Actual vs Model Prediction
          </h2>
          <p className="text-xs text-gray-500 mb-4">
            Last {history.length} trading days. The model predicts each day
            using the previous day&apos;s feature values (no look-ahead bias).
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis
                  dataKey="date"
                  stroke="#525252"
                  fontSize={11}
                  tickFormatter={(d: string) => d.slice(5)}
                />
                <YAxis
                  stroke="#525252"
                  fontSize={11}
                  domain={["auto", "auto"]}
                  tickFormatter={(v: number) => v.toFixed(1)}
                />
                <Tooltip
                  contentStyle={{
                    background: "#141414",
                    border: "1px solid #262626",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(value: number) => value.toFixed(2)}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={false}
                  name="Actual BOC1"
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  strokeDasharray="4 2"
                  dot={false}
                  name="Model Prediction"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Commodity Prices */}
      {prices.length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold mb-4">Commodity Prices</h2>
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
                <p className="text-[10px] text-gray-500 uppercase font-medium">
                  {COMMODITY_LABELS[p.name] ?? p.name}
                </p>
                <p className="text-lg font-bold mt-0.5">
                  {p.price.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
                <p className="text-[10px] text-gray-600">
                  {COMMODITY_UNITS[p.name] ?? "USD"}
                </p>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-600 mt-3">
            Source: Yahoo Finance
            {prices[0] &&
              ` | ${new Date(prices[0].timestamp).toLocaleString()}`}
          </p>
        </div>
      )}

      {/* Business Context + Pipeline */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold mb-3">Business Context</h2>
          <p className="text-gray-400 text-sm leading-relaxed">
            BOC1 is the front-month soybean oil futures contract on the CBOT —
            the global benchmark for vegetable oil. Traders use the{" "}
            <strong>crush spread</strong> (processing margin) and the{" "}
            <strong>BOC1/CPO spread</strong> (substitution signal) to time
            hedging decisions.
          </p>
          <p className="text-gray-400 text-sm leading-relaxed mt-2">
            A 1% improvement in hedge timing on a 10,000 MT position ={" "}
            <strong className="text-brand-400">~$50,000</strong>.
          </p>
        </div>
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold mb-3">Pipeline</h2>
          <div className="text-sm text-gray-400 space-y-1.5 font-mono">
            <p><span className="text-brand-400">INGEST</span> yfinance &middot; USDA &middot; FRED</p>
            <p><span className="text-gold-400">BRONZE</span> Raw Parquet (append-only)</p>
            <p><span className="text-blue-400">SILVER</span> dbt-core + DuckDB</p>
            <p><span className="text-purple-400">GOLD</span> Feature engineering (spreads, vol)</p>
            <p><span className="text-red-400">MODEL</span> XGBoost &middot; Ridge &middot; statsforecast</p>
            <p><span className="text-cyan-400">SERVE</span> FastAPI + Render</p>
          </div>
        </div>
      </div>

      {/* Models Table */}
      {models.length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold mb-4">Trained Models</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#262626] text-gray-400">
                  <th className="text-left py-2 px-3">Model</th>
                  <th className="text-left py-2 px-3">Type</th>
                  <th className="text-right py-2 px-3">R2</th>
                  <th className="text-right py-2 px-3">MAE</th>
                  <th className="text-right py-2 px-3">RMSE</th>
                  <th className="text-right py-2 px-3">Dir. Acc.</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr
                    key={m.name}
                    className="border-b border-[#262626]/50 hover:bg-white/5"
                  >
                    <td className="py-2 px-3 font-medium">{m.name}</td>
                    <td className="py-2 px-3 text-gray-400">{m.model_type}</td>
                    <td className="py-2 px-3 text-right">{m.metrics.r2?.toFixed(4) ?? "-"}</td>
                    <td className="py-2 px-3 text-right text-gray-400">{m.metrics.mae?.toFixed(2) ?? "-"}</td>
                    <td className="py-2 px-3 text-right text-gray-400">{m.metrics.rmse?.toFixed(2) ?? "-"}</td>
                    <td className="py-2 px-3 text-right text-gray-400">
                      {m.metrics.directional_accuracy
                        ? `${(m.metrics.directional_accuracy * 100).toFixed(0)}%`
                        : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
