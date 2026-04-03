"use client";

import { useState } from "react";
import { api, PredictionResponse } from "@/lib/api";

const FIELDS = [
  { key: "smc1", label: "Soybean Meal (ZM)", unit: "$/ton", placeholder: "350" },
  { key: "sc1", label: "Soybeans (ZS)", unit: "c/bu", placeholder: "1100" },
  { key: "lcoc1", label: "Brent Crude (BZ)", unit: "$/bbl", placeholder: "75" },
  { key: "hoc1", label: "Heating Oil (HO)", unit: "$/gal", placeholder: "2.20" },
  { key: "fcpoc1", label: "Palm Oil (FCPO)", unit: "MYR/t", placeholder: "3200" },
  { key: "rsc1", label: "Canola (RS)", unit: "CAD/t", placeholder: "600" },
] as const;

export default function PredictPage() {
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold gradient-text">Price Prediction</h1>
        <p className="text-gray-400 mt-2">
          Enter current commodity prices to forecast the BOC1 (soybean oil
          futures) price. The model uses correlated commodity prices as
          predictive features.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <form onSubmit={handleSubmit} className="glass-card p-6 space-y-5">
          <h2 className="text-lg font-semibold">Input Features</h2>

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
              <p className="text-sm text-gray-400 mb-2">Predicted BOC1 Price</p>
              <p className="text-5xl font-bold text-brand-400">
                {result.predicted_price.toFixed(2)}
              </p>
              <p className="text-sm text-gray-500 mt-1">cents per pound</p>
              <div className="mt-4 pt-4 border-t border-[#262626]">
                <p className="text-xs text-gray-500">
                  Model: <span className="text-gray-300">{result.model_name}</span>
                </p>
                <p className="text-xs text-gray-500">
                  Features used: {result.features_used.join(", ")}
                </p>
              </div>
            </div>
          )}

          <div className="glass-card p-6">
            <h3 className="text-sm font-semibold text-gray-400 mb-3">
              What is BOC1?
            </h3>
            <p className="text-sm text-gray-500 leading-relaxed">
              BOC1 is the front-month soybean oil futures contract traded on
              the Chicago Board of Trade (CBOT). It is priced in US cents per
              pound and is the global benchmark for vegetable oil pricing.
              Soybean oil is used in food production, biodiesel, and industrial
              applications.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
