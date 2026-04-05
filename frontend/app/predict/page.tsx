"use client";

import { useEffect, useState } from "react";
import {
  api,
  isDemoMode,
  LivePrice,
  PredictionResponse,
  ModelMetadata,
  FeatureContribution,
} from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

const FIELDS: {
  key: string;
  label: string;
  unit: string;
  why: string;
}[] = [
  { key: "smc1", label: "Soybean Meal (CBOT)", unit: "$/ton", why: "Crush economics — meal is the co-product of soybean oil production. Higher meal prices support oil margins." },
  { key: "sc1", label: "Soybeans (CBOT)", unit: "c/bu", why: "Feedstock cost — soybeans are crushed into oil and meal. Higher bean prices increase production costs." },
  { key: "lcoc1", label: "Brent Crude (ICE)", unit: "$/bbl", why: "Biodiesel demand — soybean oil competes as a biodiesel feedstock. Crude prices drive renewable fuel economics." },
  { key: "hoc1", label: "Heating Oil (NYMEX)", unit: "$/gal", why: "Energy correlation — heating oil tracks energy-vegetable oil substitution and refining margins." },
  { key: "rsc1", label: "Wheat (CBOT)", unit: "c/bu", why: "Grain complex — wheat reflects broader agricultural market sentiment and planting competition." },
];

const FEATURE_CONTEXT: Record<string, { label: string; why: string }> = Object.fromEntries(
  FIELDS.map((f) => [f.key, { label: f.label, why: f.why.split(" — ")[0] }])
);

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
    } catch {
      setError("Could not fetch live prices");
    } finally {
      setFilling(false);
    }
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
          Enter commodity prices or load live market data, then forecast
          the BOC1 soybean oil price. Adjust inputs for scenario analysis.
        </p>
      </section>

      {demo && <DemoBanner />}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Form */}
        <form onSubmit={handleSubmit} className="lg:col-span-3 glass-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Input Features</h2>
            <button
              type="button"
              onClick={fillLive}
              disabled={filling}
              className="text-[12px] bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-3 py-1.5 rounded-md transition-colors font-medium"
            >
              {filling ? "Loading..." : "Load Live Prices"}
            </button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {FIELDS.map(({ key, label, unit, why }) => {
              const range = meta?.feature_ranges?.[key];
              const livePrice = livePriceMap[key];
              const val = parseFloat(values[key] || "");
              const outOfRange = range && !isNaN(val) && val > 0 && (val < range.min || val > range.max);

              return (
                <div key={key}>
                  <div className="flex items-center gap-1 mb-1">
                    <label className="text-[11px] text-zinc-500 font-medium">
                      {label} <span className="text-zinc-600">({unit})</span>
                    </label>
                    <button
                      type="button"
                      onClick={() => setTooltipOpen(tooltipOpen === key ? null : key)}
                      className="text-zinc-600 hover:text-zinc-400 transition-colors"
                      aria-label={`Info about ${label}`}
                    >
                      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                  {tooltipOpen === key && (
                    <p className="text-[10px] text-zinc-400 mb-1 leading-relaxed">{why}</p>
                  )}
                  <input
                    type="number"
                    step="any"
                    min="0"
                    placeholder={livePrice ? `${livePrice.toFixed(2)} (live)` : range?.mean?.toFixed(1) ?? ""}
                    value={values[key] || ""}
                    onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                    className={`w-full bg-black/30 border rounded-md px-2.5 py-2 text-sm tabular-nums focus:outline-none transition-colors placeholder:text-zinc-700 ${
                      outOfRange ? "border-yellow-600/50" : "border-[#262626] focus:border-zinc-500"
                    }`}
                  />
                  {range && (
                    <p className="text-[9px] text-zinc-600 mt-0.5">
                      Range: {range.min}–{range.max} {unit}
                    </p>
                  )}
                  {outOfRange && (
                    <p className="text-[10px] text-yellow-500/80 mt-0.5">
                      Outside training distribution — prediction may be less reliable
                    </p>
                  )}
                </div>
              );
            })}
          </div>
          <button
            type="submit"
            disabled={loading || !allFilled}
            className="w-full bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium py-2.5 rounded-md transition-colors"
          >
            {loading ? "Running model..." : "Forecast BOC1 Price"}
          </button>
          {!allFilled && Object.keys(values).length > 0 && (
            <p className="text-[11px] text-zinc-600 text-center">All fields must be greater than zero</p>
          )}
        </form>

        {/* Result + Model Card */}
        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="glass-card p-3 border-red-500/20 bg-red-500/5">
              <p className="text-red-400 text-[13px]">{error}</p>
            </div>
          )}

          {result ? (
            <div className="glass-card p-5 border-green-500/20 bg-green-500/5">
              {demo && <p className="text-[10px] text-yellow-500 mb-2">Sample data</p>}
              <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">Predicted BOC1</p>
              <div className="flex items-baseline gap-2 mt-1">
                <p className="text-4xl font-bold text-green-400 tabular-nums">{result.predicted_price.toFixed(2)}</p>
                <span className="text-sm text-zinc-500">c/lb</span>
              </div>

              {result.confidence.lower != null && result.confidence.upper != null && (
                <p className="text-[11px] text-zinc-500 mt-1">
                  95% CI: {result.confidence.lower.toFixed(2)} &ndash; {result.confidence.upper.toFixed(2)} c/lb
                </p>
              )}

              {diff !== null && (
                <div className="mt-3 pt-3 border-t border-[#1e1e22] grid grid-cols-2 gap-3 text-[13px]">
                  <div>
                    <p className="text-zinc-500">Current BOC1</p>
                    <p className="font-medium tabular-nums">{boc1?.toFixed(2)} c/lb</p>
                  </div>
                  <div>
                    <p className="text-zinc-500">Signal</p>
                    <p className={`font-medium ${diff > 0 ? "text-green-400" : "text-red-400"}`}>
                      {diff > 0 ? "Upside" : "Downside"} ({diff > 0 ? "+" : ""}{diff.toFixed(2)})
                    </p>
                  </div>
                </div>
              )}
              <p className="text-[10px] text-zinc-600 mt-2">Model: {result.model_name}</p>
            </div>
          ) : (
            <div className="glass-card p-5 text-center">
              <p className="text-zinc-600 text-sm">Load live prices and run the forecast to see results.</p>
            </div>
          )}

          {/* Feature Contributions chart */}
          {result && result.feature_contributions.length > 0 && (
            <div className="glass-card p-4">
              <h3 className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-3">
                Feature Contributions
              </h3>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={result.feature_contributions}
                    layout="vertical"
                    margin={{ left: 50, right: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                    <XAxis type="number" stroke="#3f3f46" fontSize={10} />
                    <YAxis
                      type="category"
                      dataKey="feature"
                      stroke="#3f3f46"
                      fontSize={10}
                      tickFormatter={(f: string) => FEATURE_CONTEXT[f]?.label ?? f}
                    />
                    <Tooltip
                      contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }}
                      formatter={(v: number) => [`${v > 0 ? "+" : ""}${v.toFixed(2)} c/lb`, "Impact"]}
                    />
                    <ReferenceLine x={0} stroke="#3f3f46" />
                    <Bar dataKey="contribution" radius={[0, 3, 3, 0]}>
                      {result.feature_contributions.map((c, i) => (
                        <Cell key={i} fill={c.contribution > 0 ? "#22c55e" : "#ef4444"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[10px] text-zinc-600 mt-1">
                How each feature pushes the prediction above or below the base value
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Model Card */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-semibold mb-4">Model Card</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Model Info */}
          <div>
            <h3 className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Model Info</h3>
            <div className="space-y-1.5 text-[13px]">
              <div className="flex justify-between">
                <span className="text-zinc-500">Algorithm</span>
                <span className="font-medium">{meta?.algorithm ?? "XGBRegressor"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Training samples</span>
                <span className="tabular-nums">{meta?.n_training_samples?.toLocaleString() ?? "2,016"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Features</span>
                <span className="tabular-nums">{meta?.n_features ?? 8}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Last trained</span>
                <span className="text-zinc-400">
                  {meta?.trained_at ? new Date(meta.trained_at).toLocaleDateString() : "—"}
                </span>
              </div>
            </div>
          </div>

          {/* Features Used */}
          <div>
            <h3 className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Features Used</h3>
            <div className="space-y-1 text-[12px]">
              {Object.entries(FEATURE_CONTEXT).map(([key, { label, why }]) => (
                <div key={key} className="flex items-baseline gap-1.5">
                  <span className="text-zinc-300 font-medium">{label}</span>
                  <span className="text-zinc-600">&mdash;</span>
                  <span className="text-zinc-500">{why}</span>
                </div>
              ))}
              <div className="flex items-baseline gap-1.5">
                <span className="text-zinc-300 font-medium">Month</span>
                <span className="text-zinc-600">&mdash;</span>
                <span className="text-zinc-500">Cyclical encoded seasonality</span>
              </div>
            </div>
          </div>

          {/* Validation Metrics */}
          <div>
            <h3 className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-2">
              Walk-Forward Validation
            </h3>
            <div className="space-y-1.5 text-[13px]">
              <div className="flex justify-between">
                <span className="text-zinc-500">MAE</span>
                <span className="tabular-nums">{bm?.mae?.toFixed(2) ?? "3.83"} c/lb</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">RMSE</span>
                <span className="tabular-nums">{bm?.rmse?.toFixed(2) ?? "6.16"} c/lb</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">R&sup2;</span>
                <span className="tabular-nums">{bm?.r2?.toFixed(4) ?? "0.8266"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Dir. Accuracy</span>
                <span className="tabular-nums">
                  {bm?.directional_accuracy ? `${(bm.directional_accuracy * 100).toFixed(1)}%` : "61.9%"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">OOS predictions</span>
                <span className="tabular-nums">{bm?.n_predictions?.toLocaleString() ?? "2,100"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-500">Folds</span>
                <span className="tabular-nums">{bm?.n_folds ?? 5} (TimeSeriesSplit)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
