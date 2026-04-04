"use client";

import { useEffect, useState } from "react";
import { api, isDemoMode, LivePrice, PredictionResponse } from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";

const FIELDS = [
  { key: "smc1", label: "Soybean Meal (ZM)", unit: "$/ton", placeholder: "350" },
  { key: "sc1", label: "Soybeans (ZS)", unit: "c/bu", placeholder: "1100" },
  { key: "lcoc1", label: "Brent Crude (BZ)", unit: "$/bbl", placeholder: "75" },
  { key: "hoc1", label: "Heating Oil (HO)", unit: "$/gal", placeholder: "2.20" },
  { key: "fcpoc1", label: "Palm Oil (PALM)", unit: "GBp", placeholder: "78" },
  { key: "rsc1", label: "Wheat (ZW)", unit: "c/bu", placeholder: "600" },
] as const;

export default function PredictPage() {
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingPrices, setLoadingPrices] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demo, setDemo] = useState(false);
  const [boc1Current, setBoc1Current] = useState<number | null>(null);
  const [livePrices, setLivePrices] = useState<LivePrice[]>([]);

  useEffect(() => {
    api.livePrices().then((data) => {
      setLivePrices(data.prices);
      const boc1 = data.prices.find((p) => p.name === "boc1");
      if (boc1) setBoc1Current(boc1.price);
    }).catch(() => {});
  }, []);

  const fillLivePrices = async () => {
    setLoadingPrices(true);
    try {
      const data = await api.livePrices();
      setLivePrices(data.prices);
      const newValues: Record<string, string> = {};
      for (const { key } of FIELDS) {
        const match = data.prices.find((p) => p.name === key);
        if (match) newValues[key] = match.price.toString();
      }
      setValues(newValues);
      setDemo(isDemoMode());
      const boc1 = data.prices.find((p) => p.name === "boc1");
      if (boc1) setBoc1Current(boc1.price);
    } catch {
      setError("Could not fetch live prices");
    } finally {
      setLoadingPrices(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const payload = Object.fromEntries(
        FIELDS.map(({ key }) => [key, parseFloat(values[key] || "0")])
      ) as any;
      payload.month = new Date().getMonth() + 1;

      const data = await api.predict(payload);
      setResult(data);
      setDemo(isDemoMode());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const predDiff =
    result && boc1Current
      ? result.predicted_price - boc1Current
      : null;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold gradient-text">Price Prediction</h1>
        <p className="text-gray-400 mt-2">
          Forecast the BOC1 soybean oil futures price. Use live market prices
          as a starting point, then adjust for scenario analysis.
        </p>
      </div>

      {demo && <DemoBanner />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <form onSubmit={handleSubmit} className="glass-card p-6 space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Input Features</h2>
            <button
              type="button"
              onClick={fillLivePrices}
              disabled={loadingPrices}
              className="text-sm bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-3 py-1.5 rounded-lg transition-colors"
            >
              {loadingPrices ? "Loading..." : "Use Live Prices"}
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {FIELDS.map(({ key, label, unit, placeholder }) => (
              <div key={key}>
                <label className="block text-sm text-gray-400 mb-1">
                  {label}{" "}
                  <span className="text-gray-600">({unit})</span>
                </label>
                <input
                  type="number"
                  step="any"
                  placeholder={placeholder}
                  value={values[key] || ""}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [key]: e.target.value }))
                  }
                  className="w-full bg-black/30 border border-[#333] rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-brand-500 transition-colors"
                  required
                />
              </div>
            ))}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-brand-600 hover:bg-brand-500 disabled:opacity-50 text-white font-medium py-2.5 rounded-lg transition-colors"
          >
            {loading ? "Predicting..." : "Predict BOC1 Price"}
          </button>
        </form>

        <div className="space-y-4">
          {error && (
            <div className="glass-card p-4 border-red-500/30 bg-red-500/5">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {result && (
            <div className="glass-card p-6 border-brand-500/30 bg-brand-500/5">
              {demo && (
                <p className="text-xs text-gold-400 mb-3">
                  Demo mode — simplified formula, not the trained model
                </p>
              )}
              <p className="text-sm text-gray-400 mb-2">Predicted BOC1 Price</p>
              <div className="flex items-end gap-3">
                <p className="text-5xl font-bold text-brand-400">
                  {result.predicted_price.toFixed(2)}
                </p>
                {predDiff !== null && (
                  <span
                    className={`text-sm font-medium px-2 py-1 rounded-full mb-2 ${
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
              <p className="text-sm text-gray-500 mt-1">cents per pound</p>

              {boc1Current && (
                <div className="mt-4 pt-4 border-t border-[#262626] grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-gray-500">Current BOC1</p>
                    <p className="font-medium">{boc1Current.toFixed(2)} c/lb</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Signal</p>
                    <p className={`font-medium ${
                      predDiff && predDiff > 0 ? "text-green-400" : "text-red-400"
                    }`}>
                      {predDiff && predDiff > 0
                        ? "Model suggests upside"
                        : "Model suggests downside"}
                    </p>
                  </div>
                </div>
              )}

              <div className="mt-3 pt-3 border-t border-[#262626]">
                <p className="text-xs text-gray-500">
                  Model: {result.model_name} | Features: {result.features_used.join(", ")}
                </p>
              </div>
            </div>
          )}

          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-gray-400 mb-3">
              How to use
            </h3>
            <ol className="text-sm text-gray-500 leading-relaxed space-y-2 list-decimal list-inside">
              <li>
                Click <strong>Use Live Prices</strong> to auto-fill with current
                market data from Yahoo Finance.
              </li>
              <li>
                Adjust any price to run a{" "}
                <strong>scenario analysis</strong> — e.g., &ldquo;what if crude oil
                jumps 10%?&rdquo;
              </li>
              <li>
                Click <strong>Predict</strong> to see the forecasted BOC1 price
                and whether the model sees upside or downside vs current.
              </li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
