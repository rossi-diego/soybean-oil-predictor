"use client";

import { useEffect, useState } from "react";
import {
  api,
  isDemoMode,
  PredictionResponse,
  ModelMetadata,
} from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";

const FIELDS: { key: string; label: string; unit: string; why: string }[] = [
  { key: "smc1", label: "Soybean Meal (CBOT)", unit: "$/ton", why: "Crush economics — meal is the co-product of soybean oil production." },
  { key: "sc1", label: "Soybeans (CBOT)", unit: "c/bu", why: "Feedstock cost — soybeans are crushed into oil and meal." },
  { key: "lcoc1", label: "Brent Crude (ICE)", unit: "$/bbl", why: "Biodiesel demand — soybean oil competes as a biodiesel feedstock." },
  { key: "hoc1", label: "Heating Oil (NYMEX)", unit: "$/gal", why: "Energy correlation — heating oil tracks energy-vegetable oil substitution." },
  { key: "rsc1", label: "Wheat (CBOT)", unit: "c/bu", why: "Grain complex — wheat reflects broader agricultural market sentiment." },
];

const FEATURE_LABELS: Record<string, string> = Object.fromEntries(
  FIELDS.map((f) => [f.key, f.label.split(" (")[0]])
);

const SCENARIOS: { label: string; desc: string; adjustments: Record<string, number> }[] = [
  { label: "Crude Spike +20%", desc: "Oil supply shock", adjustments: { lcoc1: 1.2, hoc1: 1.15 } },
  { label: "Demand Destruction -10%", desc: "Global slowdown", adjustments: { smc1: 0.9, sc1: 0.9, lcoc1: 0.9, hoc1: 0.9, rsc1: 0.9 } },
  { label: "Grain Rally +15%", desc: "Drought or export ban", adjustments: { sc1: 1.15, smc1: 1.1, rsc1: 1.15 } },
];

export default function PredictPage() {
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [filling, setFilling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demo, setDemo] = useState(false);
  const [boc1, setBoc1] = useState<number | null>(null);
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [livePriceMap, setLivePriceMap] = useState<Record<string, number>>({});
  const [tooltipOpen, setTooltipOpen] = useState<string | null>(null);
  const [modelCardOpen, setModelCardOpen] = useState(false);

  useEffect(() => {
    api.livePrices().then((d) => {
      const b = d.prices.find((p) => p.name === "boc1");
      if (b) setBoc1(b.price);
      const map: Record<string, number> = {};
      d.prices.forEach((p) => { map[p.name] = p.price; });
      setLivePriceMap(map);
    }).catch(() => {});
    api.modelInfo().then(setMeta).catch(() => {});
  }, []);

  const fillLive = async () => {
    setFilling(true);
    try {
      const d = await api.livePrices();
      const v: Record<string, string> = {};
      for (const { key } of FIELDS) {
        const m = d.prices.find((p) => p.name === key);
        if (m) v[key] = m.price.toString();
      }
      setValues(v);
      setDemo(isDemoMode());
      const b = d.prices.find((p) => p.name === "boc1");
      if (b) setBoc1(b.price);
      const map: Record<string, number> = {};
      d.prices.forEach((p) => { map[p.name] = p.price; });
      setLivePriceMap(map);
    } catch {
      setError("Could not fetch live prices");
    } finally {
      setFilling(false);
    }
  };

  const resetToLive = () => {
    const v: Record<string, string> = {};
    for (const { key } of FIELDS) {
      if (livePriceMap[key]) v[key] = livePriceMap[key].toString();
    }
    setValues(v);
  };

  const applyScenario = (adjustments: Record<string, number>) => {
    const base: Record<string, string> = {};
    for (const { key } of FIELDS) {
      const live = livePriceMap[key] || parseFloat(values[key] || "0");
      const mult = adjustments[key] ?? 1;
      base[key] = (live * mult).toFixed(2);
    }
    setValues(base);
  };

  const allFilled = FIELDS.every(({ key }) => {
    const v = parseFloat(values[key] || "");
    return !isNaN(v) && v > 0;
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!allFilled) return;
    setLoading(true);
    setError(null);
    const payload = Object.fromEntries(
      FIELDS.map(({ key }) => [key, parseFloat(values[key])])
    ) as any;
    payload.month = new Date().getMonth() + 1;
    try {
      const data = await api.predict(payload);
      setResult(data);
      setDemo(isDemoMode());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const diff = result && boc1 ? result.predicted_price - boc1 : null;
  const bm = meta?.backtest_metrics;

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-bold tracking-tight">Price Forecast</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Load live prices, run the model, and explore scenarios.
        </p>
      </section>

      {demo && <DemoBanner />}

      {/* Section 1: Quick Forecast */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold">Quick Forecast</h2>
          <div className="flex gap-2">
            <button
              onClick={fillLive}
              disabled={filling}
              className="text-[12px] bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-3 py-1.5 rounded-md transition-colors font-medium"
            >
              {filling ? "Loading..." : "Load Live Prices"}
            </button>
            {allFilled && (
              <button
                onClick={handleSubmit as any}
                disabled={loading}
                className="text-[12px] bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white px-3 py-1.5 rounded-md transition-colors font-medium"
              >
                {loading ? "Running..." : "Forecast"}
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="p-2 rounded bg-red-500/5 border border-red-500/20 mb-3">
            <p className="text-red-400 text-[12px]">{error}</p>
          </div>
        )}

        {result ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-green-500/5 border border-green-500/20">
              {demo && <p className="text-[9px] text-yellow-500 mb-1">Sample data</p>}
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Predicted BOC1</p>
              <div className="flex items-baseline gap-2 mt-1">
                <p className="text-4xl font-bold text-green-400 tabular-nums">{result.predicted_price.toFixed(2)}</p>
                <span className="text-sm text-zinc-500">c/lb</span>
                {diff !== null && (
                  <span className={`text-sm font-medium ${diff > 0 ? "text-green-400" : "text-red-400"}`}>
                    {diff > 0 ? "\u25B2+" : "\u25BC"}{diff.toFixed(2)}
                  </span>
                )}
              </div>
              {result.confidence.lower != null && result.confidence.upper != null && (
                <p className="text-[11px] text-zinc-500 mt-1">
                  95% CI: {result.confidence.lower.toFixed(2)} &ndash; {result.confidence.upper.toFixed(2)}
                </p>
              )}
              {diff !== null && boc1 != null && (
                <div className="mt-2 pt-2 border-t border-[#1e1e22]">
                  <div className="flex items-center gap-2 text-[13px]">
                    <span className={`font-semibold ${diff > 0 ? "text-green-400" : "text-red-400"}`}>
                      {diff > 0 ? "\u25B2 UP" : "\u25BC DOWN"} ({diff > 0 ? "+" : ""}{((diff / boc1) * 100).toFixed(1)}%)
                    </span>
                  </div>
                  <p className="text-[11px] text-zinc-500 mt-1">
                    vs current {boc1.toFixed(2)} c/lb ({diff > 0 ? "+" : ""}{diff.toFixed(2)} c/lb)
                  </p>
                </div>
              )}
              <p className="text-[9px] text-zinc-600 mt-2">Model: {result.model_name}</p>
            </div>

            {/* Feature Contributions */}
            {result.feature_contributions.length > 0 && (
              <div>
                <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Why this price?</p>
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.feature_contributions} layout="vertical" margin={{ left: 70, right: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                      <XAxis type="number" stroke="#3f3f46" fontSize={9} />
                      <YAxis type="category" dataKey="feature" stroke="#3f3f46" fontSize={9} tickFormatter={(f: string) => FEATURE_LABELS[f] ?? f} />
                      <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} formatter={(v: number) => [`${v > 0 ? "+" : ""}${v.toFixed(2)} c/lb`, "Impact"]} />
                      <ReferenceLine x={0} stroke="#3f3f46" />
                      <Bar dataKey="contribution" radius={[0, 3, 3, 0]}>
                        {result.feature_contributions.map((c, i) => (
                          <Cell key={i} fill={c.contribution > 0 ? "#22c55e" : "#ef4444"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className="text-zinc-600 text-[13px] py-4 text-center">
            Click <strong className="text-zinc-400">Load Live Prices</strong> then <strong className="text-zinc-400">Forecast</strong> to see the prediction.
          </p>
        )}
      </div>

      {/* Section 2: Model Card (collapsible) */}
      <div className="glass-card">
        <button
          onClick={() => setModelCardOpen(!modelCardOpen)}
          className="w-full p-4 flex items-center justify-between text-left"
        >
          <h2 className="text-sm font-semibold">Model Card</h2>
          <span className="text-zinc-500 text-xs">{modelCardOpen ? "Hide" : "Show"}</span>
        </button>
        {modelCardOpen && (
          <div className="px-5 pb-5 grid grid-cols-1 md:grid-cols-3 gap-5 border-t border-[#1e1e22]  pt-4">
            <div>
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Model</h3>
              <div className="space-y-1.5 text-[12px]">
                <div className="flex justify-between"><span className="text-zinc-500">Algorithm</span><span>{meta?.algorithm ?? "Ridge"}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">Champion</span><span>{(meta as any)?.champion ?? "ridge"}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">Training samples</span><span className="tabular-nums">{meta?.n_training_samples?.toLocaleString() ?? "2,016"}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">Period</span><span className="text-zinc-400">{(meta as any)?.train_start ?? "2013"} &ndash; {(meta as any)?.test_end ?? "2025"}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">Last trained</span><span className="text-zinc-400">{meta?.trained_at ? new Date(meta.trained_at).toLocaleDateString() : "\u2014"}</span></div>
              </div>
            </div>
            <div>
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Features</h3>
              <div className="space-y-1 text-[11px]">
                {FIELDS.map(({ key, label, why }) => (
                  <div key={key}>
                    <span className="text-zinc-300">{label.split(" (")[0]}</span>
                    <span className="text-zinc-600"> &mdash; {why.split(" — ")[1]?.slice(0, 40) ?? why.slice(0, 40)}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Walk-Forward Validation</h3>
              <div className="space-y-1.5 text-[12px]">
                <div className="flex justify-between"><span className="text-zinc-500">MAE</span><span className="tabular-nums">{bm?.mae?.toFixed(2) ?? "2.69"} c/lb</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">RMSE</span><span className="tabular-nums">{bm?.rmse?.toFixed(2) ?? "3.53"} c/lb</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">R&sup2;</span><span className="tabular-nums">{bm?.r2?.toFixed(4) ?? "0.8138"}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">Dir. Accuracy</span><span className="tabular-nums">{bm?.directional_accuracy ? `${(bm.directional_accuracy * 100).toFixed(1)}%` : "76.3%"}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">OOS predictions</span><span className="tabular-nums">{bm?.n_predictions?.toLocaleString() ?? "2,100"}</span></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Section 3: Scenario Analysis */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold">Scenario Analysis</h2>
          <div className="flex gap-1.5">
            {Object.keys(livePriceMap).length > 0 && (
              <button onClick={resetToLive} className="text-[10px] text-zinc-500 hover:text-zinc-300 px-2 py-1 rounded border border-[#262626] transition-colors">
                Reset to Live
              </button>
            )}
          </div>
        </div>

        {/* Scenario presets */}
        <div className="flex flex-wrap gap-2 mb-4">
          {SCENARIOS.map((s) => (
            <button
              key={s.label}
              onClick={() => applyScenario(s.adjustments)}
              className="text-[11px] px-2.5 py-1.5 rounded-md bg-white/[0.03] border border-[#262626] text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 transition-colors"
              title={s.desc}
            >
              {s.label}
            </button>
          ))}
        </div>

        {/* Input fields */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {FIELDS.map(({ key, label, unit, why }) => {
              const range = meta?.feature_ranges?.[key];
              const livePrice = livePriceMap[key];
              const val = parseFloat(values[key] || "");
              const outOfRange = range && !isNaN(val) && val > 0 && (val < range.min || val > range.max);
              return (
                <div key={key}>
                  <div className="flex items-center gap-1 mb-1">
                    <label className="text-[10px] text-zinc-500 font-medium">{label.split(" (")[0]}</label>
                    <button type="button" onClick={() => setTooltipOpen(tooltipOpen === key ? null : key)} className="text-zinc-600 hover:text-zinc-400" aria-label="Info">
                      <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" /></svg>
                    </button>
                  </div>
                  {tooltipOpen === key && <p className="text-[9px] text-zinc-400 mb-1">{why}</p>}
                  <input
                    type="number" step="any" min="0"
                    placeholder={livePrice ? `${livePrice.toFixed(2)}` : range?.mean?.toFixed(1) ?? ""}
                    value={values[key] || ""}
                    onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                    className={`w-full bg-black/30 border rounded-md px-2 py-1.5 text-[13px] tabular-nums focus:outline-none transition-colors placeholder:text-zinc-700 ${outOfRange ? "border-yellow-600/50" : "border-[#262626] focus:border-zinc-500"}`}
                  />
                  {range && <p className="text-[8px] text-zinc-600 mt-0.5">{range.min}\u2013{range.max} {unit}</p>}
                  {outOfRange && <p className="text-[9px] text-yellow-500/80 mt-0.5">Outside training range</p>}
                </div>
              );
            })}
          </div>
          <button
            type="submit" disabled={loading || !allFilled}
            className="bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-[12px] font-medium py-2 px-6 rounded-md transition-colors"
          >
            {loading ? "Running model..." : "Run Forecast"}
          </button>
        </form>
      </div>
    </div>
  );
}
