"use client";

import { useEffect, useState } from "react";
import { api, isDemoMode, LivePrice, PredictionResponse } from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";

const FIELDS = [
  { key: "smc1", label: "Soybean Meal", ticker: "ZM", unit: "$/ton", placeholder: "350" },
  { key: "sc1", label: "Soybeans", ticker: "ZS", unit: "c/bu", placeholder: "1100" },
  { key: "lcoc1", label: "Brent Crude", ticker: "BZ", unit: "$/bbl", placeholder: "75" },
  { key: "hoc1", label: "Heating Oil", ticker: "HO", unit: "$/gal", placeholder: "2.20" },
  { key: "fcpoc1", label: "Palm Oil", ticker: "PALM", unit: "GBp", placeholder: "78" },
  { key: "rsc1", label: "Wheat", ticker: "ZW", unit: "c/bu", placeholder: "600" },
] as const;

export default function PredictPage() {
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [filling, setFilling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demo, setDemo] = useState(false);
  const [boc1, setBoc1] = useState<number | null>(null);

  useEffect(() => {
    api.livePrices().then((d) => {
      const b = d.prices.find((p) => p.name === "boc1");
      if (b) setBoc1(b.price);
    }).catch(() => {});
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

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-bold tracking-tight">
          Price Forecast
        </h1>
        <p className="text-sm text-zinc-500 mt-1">
          Enter commodity prices or load live market data, then forecast
          the BOC1 soybean oil price. Adjust inputs for scenario analysis.
        </p>
      </section>

      {demo && <DemoBanner />}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Form — 3 columns */}
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
            {FIELDS.map(({ key, label, ticker, unit, placeholder }) => (
              <div key={key}>
                <label className="block text-[11px] text-zinc-500 mb-1 font-medium">
                  {label}
                  <span className="text-zinc-600 ml-1">({unit})</span>
                </label>
                <input
                  type="number"
                  step="any"
                  min="0"
                  placeholder={placeholder}
                  value={values[key] || ""}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [key]: e.target.value }))
                  }
                  className="w-full bg-black/30 border border-[#262626] rounded-md px-2.5 py-2 text-sm tabular-nums focus:outline-none focus:border-zinc-500 transition-colors placeholder:text-zinc-700"
                />
              </div>
            ))}
          </div>

          <button
            type="submit"
            disabled={loading || !allFilled}
            className="w-full bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium py-2.5 rounded-md transition-colors"
          >
            {loading ? "Running model..." : "Forecast BOC1 Price"}
          </button>

          {!allFilled && Object.keys(values).length > 0 && (
            <p className="text-[11px] text-zinc-600 text-center">
              All fields must be greater than zero
            </p>
          )}
        </form>

        {/* Result — 2 columns */}
        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="glass-card p-3 border-red-500/20 bg-red-500/5">
              <p className="text-red-400 text-[13px]">{error}</p>
            </div>
          )}

          {result ? (
            <div className="glass-card p-5 border-green-500/20 bg-green-500/5">
              {demo && (
                <p className="text-[10px] text-yellow-500 mb-2">
                  Sample data &mdash; simplified formula
                </p>
              )}
              <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">
                Predicted BOC1
              </p>
              <div className="flex items-baseline gap-2 mt-1">
                <p className="text-4xl font-bold text-green-400 tabular-nums">
                  {result.predicted_price.toFixed(2)}
                </p>
                <span className="text-sm text-zinc-500">c/lb</span>
              </div>

              {diff !== null && (
                <div className="mt-3 pt-3 border-t border-[#1e1e22] grid grid-cols-2 gap-3 text-[13px]">
                  <div>
                    <p className="text-zinc-500">Current BOC1</p>
                    <p className="font-medium tabular-nums">
                      {boc1?.toFixed(2)} c/lb
                    </p>
                  </div>
                  <div>
                    <p className="text-zinc-500">Signal</p>
                    <p
                      className={`font-medium ${
                        diff > 0 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {diff > 0 ? "Upside" : "Downside"}{" "}
                      ({diff > 0 ? "+" : ""}
                      {diff.toFixed(2)})
                    </p>
                  </div>
                </div>
              )}

              <p className="text-[10px] text-zinc-600 mt-3">
                Model: {result.model_name}
              </p>
            </div>
          ) : (
            <div className="glass-card p-5 text-center">
              <p className="text-zinc-600 text-sm">
                Load live prices and run the forecast to see results.
              </p>
            </div>
          )}

          <div className="glass-card p-4">
            <h3 className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-2">
              How to use
            </h3>
            <ol className="text-[13px] text-zinc-500 space-y-1.5 list-decimal list-inside">
              <li>
                Click <strong className="text-zinc-300">Load Live Prices</strong>{" "}
                to pull current market data
              </li>
              <li>
                Adjust any value for scenario analysis
              </li>
              <li>
                Click <strong className="text-zinc-300">Forecast</strong>{" "}
                to see the predicted price vs current
              </li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
