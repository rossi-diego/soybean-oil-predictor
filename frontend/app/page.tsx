"use client";

import { useEffect, useState } from "react";
import { api, HealthResponse, ModelInfo } from "@/lib/api";

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

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e) => setError(e.message));

    api
      .listModels()
      .then((data) => setModels(data.models))
      .catch(() => {});
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold gradient-text">Dashboard</h1>
        <p className="text-gray-400 mt-2">
          BOC1 (Soybean Oil Futures) price forecasting with domain-driven ML.
          Built for commodity traders and risk managers.
        </p>
      </div>

      {error && (
        <div className="glass-card p-4 border-gold-500/30 bg-gold-500/5">
          <p className="text-gold-400 text-sm">
            API not reachable: {error}. Start the backend with{" "}
            <code className="bg-black/30 px-1 rounded">
              uvicorn src.serving.app:app
            </code>
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatusCard
          title="API Status"
          value={health?.status === "healthy" ? "Healthy" : "Offline"}
          subtitle={`v${health?.version || "?"}`}
          status={health?.status === "healthy" ? "ok" : "error"}
        />
        <StatusCard
          title="Models Loaded"
          value={health?.models_loaded ? "Yes" : "No"}
          subtitle={`${models.length} model(s) available`}
          status={health?.models_loaded ? "ok" : "warn"}
        />
        <StatusCard
          title="Data Fresh"
          value={health?.data_fresh ? "Yes" : "No"}
          subtitle="Training data available"
          status={health?.data_fresh ? "ok" : "warn"}
        />
        <StatusCard
          title="Active Model"
          value={models.length > 0 ? models[0].name : "None"}
          subtitle={models.length > 0 ? models[0].model_type : "Train a model first"}
          status={models.length > 0 ? "ok" : "warn"}
        />
      </div>

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
            <p>
              <span className="text-brand-400">INGEST</span> &rarr; yfinance
              &middot; USDA &middot; FRED
            </p>
            <p>
              <span className="text-gold-400">BRONZE</span> &rarr; Raw
              Parquet (append-only)
            </p>
            <p>
              <span className="text-blue-400">SILVER</span> &rarr; dbt-core
              + DuckDB (cleaned)
            </p>
            <p>
              <span className="text-purple-400">GOLD</span> &rarr; dbt-core
              + DuckDB (features)
            </p>
            <p>
              <span className="text-red-400">MODEL</span> &rarr; XGBoost
              &middot; Ridge &middot; statsforecast
            </p>
            <p>
              <span className="text-cyan-400">SERVE</span> &rarr; FastAPI +
              MLflow
            </p>
          </div>
        </div>
      </div>

      <div className="glass-card p-6">
        <h2 className="text-xl font-semibold mb-4">Available Models</h2>
        {models.length === 0 ? (
          <p className="text-gray-500 text-sm">
            No models loaded. Run the training pipeline first.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#262626] text-gray-400">
                  <th className="text-left py-2 px-3">Name</th>
                  <th className="text-left py-2 px-3">Type</th>
                  <th className="text-left py-2 px-3">Features</th>
                  <th className="text-left py-2 px-3">Trained At</th>
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
                    <td className="py-2 px-3 text-gray-400">{m.n_features}</td>
                    <td className="py-2 px-3 text-gray-400">
                      {m.trained_at
                        ? new Date(m.trained_at).toLocaleDateString()
                        : "Unknown"}
                    </td>
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
