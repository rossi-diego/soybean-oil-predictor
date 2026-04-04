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

const LABELS: Record<string, string> = {
  boc1: "Soybean Oil",
  sc1: "Soybeans",
  smc1: "Soybean Meal",
  lcoc1: "Brent Crude",
  hoc1: "Heating Oil",
  fcpoc1: "Palm Oil",
  rsc1: "Wheat",
  zc1: "Corn",
};

const UNITS: Record<string, string> = {
  boc1: "c/lb",
  sc1: "c/bu",
  smc1: "$/ton",
  lcoc1: "$/bbl",
  hoc1: "$/gal",
  fcpoc1: "GBp",
  rsc1: "c/bu",
  zc1: "c/bu",
};

function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} />;
}

function computeSpreads(prices: LivePrice[]) {
  const p: Record<string, number> = {};
  prices.forEach((pr) => (p[pr.name] = pr.price));
  return {
    crush: p.boc1 && p.smc1 && p.sc1 ? 11 * p.boc1 + p.smc1 - p.sc1 : null,
    oilPalm: p.boc1 && p.fcpoc1 ? p.boc1 - p.fcpoc1 / 100 : null,
  };
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
      api.listModels().then((d) => setModels(d.models)),
      api.livePrices().then((d) => setPrices(d.prices)),
      api.predictLive().then(setLivePred),
      api.priceHistory(90).then((d) => setHistory(d.history)),
    ]).finally(() => setLoaded(true));
  }, []);

  const demo = loaded && isDemoMode();
  const boc1 = prices.find((p) => p.name === "boc1")?.price;
  const spreads = computeSpreads(prices);
  const diff = livePred && boc1 ? livePred.predicted_price - boc1 : null;

  return (
    <div className="space-y-6">
      {demo && <DemoBanner />}

      {/* Hero */}
      <section>
        <h1 className="text-2xl font-bold tracking-tight">
          BOC1 Soybean Oil Futures
        </h1>
        <p className="text-sm text-zinc-500 mt-1">
          Front-month contract forecast using commodity cross-correlations,
          crush economics, and XGBoost.
        </p>
      </section>

      {/* Key metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="glass-card p-4">
          <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">
            BOC1 Last
          </p>
          {!loaded ? (
            <Skeleton className="h-8 w-24 mt-1.5" />
          ) : (
            <p className="text-2xl font-bold mt-1 text-green-400">
              {boc1?.toFixed(2) ?? "N/A"}
            </p>
          )}
          <p className="text-[10px] text-zinc-600 mt-0.5">cents/lb, CBOT</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">
            Model Forecast
          </p>
          {!loaded ? (
            <Skeleton className="h-8 w-24 mt-1.5" />
          ) : (
            <div className="flex items-baseline gap-2 mt-1">
              <p className="text-2xl font-bold text-blue-400">
                {livePred?.predicted_price.toFixed(2) ?? "N/A"}
              </p>
              {diff !== null && (
                <span
                  className={`text-xs font-medium ${
                    diff > 0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {diff > 0 ? "+" : ""}
                  {diff.toFixed(2)}
                </span>
              )}
            </div>
          )}
          <p className="text-[10px] text-zinc-600 mt-0.5">
            {livePred?.model_name ?? "loading"}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">
            Crush Spread
          </p>
          {!loaded ? (
            <Skeleton className="h-8 w-20 mt-1.5" />
          ) : (
            <p className="text-2xl font-bold mt-1">
              {spreads.crush?.toFixed(1) ?? "N/A"}
            </p>
          )}
          <p className="text-[10px] text-zinc-600 mt-0.5">
            11 x oil + meal &minus; beans
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">
            Oil / Palm Spread
          </p>
          {!loaded ? (
            <Skeleton className="h-8 w-20 mt-1.5" />
          ) : (
            <p className="text-2xl font-bold mt-1">
              {spreads.oilPalm?.toFixed(2) ?? "N/A"}
            </p>
          )}
          <p className="text-[10px] text-zinc-600 mt-0.5">
            BOC1 &minus; CPO/100
          </p>
        </div>
      </div>

      {/* Chart */}
      {(history.length > 0 || !loaded) && (
        <div className="glass-card p-5">
          <div className="flex items-baseline justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold">
                Actual vs Predicted
              </h2>
              <p className="text-[11px] text-zinc-500 mt-0.5">
                {history.length} trading days, walk-forward out-of-sample
              </p>
            </div>
            {loaded && (
              <span className="text-[10px] text-zinc-600">
                Model: {livePred?.model_name}
              </span>
            )}
          </div>
          {!loaded ? (
            <Skeleton className="h-64 w-full" />
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                  <XAxis
                    dataKey="date"
                    stroke="#3f3f46"
                    fontSize={10}
                    tickFormatter={(d: string) => d.slice(5)}
                  />
                  <YAxis
                    stroke="#3f3f46"
                    fontSize={10}
                    domain={["auto", "auto"]}
                    tickFormatter={(v: number) => v.toFixed(0)}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#111113",
                      border: "1px solid #1e1e22",
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                    formatter={(v: number) => v?.toFixed(2)}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 11, color: "#71717a" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#22c55e"
                    strokeWidth={1.5}
                    dot={false}
                    name="Actual"
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#3b82f6"
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    dot={false}
                    name="Predicted"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Commodity prices */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-semibold mb-3">Market Prices</h2>
        {!loaded ? (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2">
            {Array.from({ length: 7 }).map((_, i) => (
              <Skeleton key={i} className="h-16" />
            ))}
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2">
              {prices.map((p) => (
                <div
                  key={p.name}
                  className={`p-2.5 rounded-lg ${
                    p.name === "boc1"
                      ? "bg-green-500/8 border border-green-500/20"
                      : "bg-white/[0.02]"
                  }`}
                >
                  <p className="text-[10px] text-zinc-500 font-medium uppercase">
                    {LABELS[p.name] ?? p.name}
                  </p>
                  <p className="text-base font-semibold mt-0.5 tabular-nums">
                    {p.price.toLocaleString(undefined, {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </p>
                  <p className="text-[9px] text-zinc-600">
                    {UNITS[p.name] ?? "USD"}
                  </p>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-zinc-600 mt-2">
              Yahoo Finance
              {prices[0] &&
                ` \u00B7 ${new Date(prices[0].timestamp).toLocaleString()}`}
            </p>
          </>
        )}
      </div>

      {/* Two columns: context + models */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold mb-2">Why This Matters</h2>
          <p className="text-[13px] text-zinc-400 leading-relaxed">
            Soybean oil (BOC1) is the global vegetable oil benchmark.
            The <strong className="text-zinc-300">crush spread</strong>{" "}
            measures processor margins. The{" "}
            <strong className="text-zinc-300">oil/palm spread</strong>{" "}
            captures substitution dynamics between the two most traded
            vegetable oils. Traders use these signals to time hedging
            decisions on physical positions.
          </p>
          <p className="text-[13px] text-zinc-400 leading-relaxed mt-2">
            A 1% improvement in hedge timing on 10,000 MT ={" "}
            <strong className="text-green-400">~$50,000</strong> in value.
          </p>
        </div>

        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold mb-2">Data Pipeline</h2>
          <div className="space-y-1.5 text-[13px]">
            {[
              ["Ingest", "yfinance, USDA WASDE, FRED", "text-green-400"],
              ["Bronze", "Raw Parquet, append-only", "text-yellow-500"],
              ["Silver", "dbt-core + DuckDB, cleaned", "text-blue-400"],
              ["Gold", "Feature engineering", "text-purple-400"],
              ["Model", "XGBoost, Ridge, statsforecast", "text-red-400"],
              ["Serve", "FastAPI on Render", "text-cyan-400"],
            ].map(([stage, desc, color]) => (
              <div key={stage} className="flex items-center gap-3">
                <span
                  className={`text-[11px] font-mono font-semibold w-14 ${color}`}
                >
                  {stage}
                </span>
                <span className="text-zinc-500">{desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model table */}
      {models.length > 0 && (
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold mb-3">Model Performance</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b border-[#1e1e22] text-zinc-500">
                  <th className="text-left py-2 px-3 font-medium">Model</th>
                  <th className="text-right py-2 px-3 font-medium">R&sup2;</th>
                  <th className="text-right py-2 px-3 font-medium">MAE</th>
                  <th className="text-right py-2 px-3 font-medium">RMSE</th>
                  <th className="text-right py-2 px-3 font-medium">
                    Dir. Acc.
                  </th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr
                    key={m.name}
                    className="border-b border-[#1e1e22]/50 hover:bg-white/[0.02]"
                  >
                    <td className="py-2 px-3 font-medium">
                      {m.name}
                      {i === 0 && (
                        <span className="ml-2 text-[10px] text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded">
                          active
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-right tabular-nums">
                      {m.metrics.r2?.toFixed(4) ?? "-"}
                    </td>
                    <td className="py-2 px-3 text-right tabular-nums text-zinc-400">
                      {m.metrics.mae?.toFixed(2) ?? "-"}
                    </td>
                    <td className="py-2 px-3 text-right tabular-nums text-zinc-400">
                      {m.metrics.rmse?.toFixed(2) ?? "-"}
                    </td>
                    <td className="py-2 px-3 text-right tabular-nums text-zinc-400">
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
