"use client";

import { useEffect, useState } from "react";
import { api, HealthResponse } from "@/lib/api";

export default function MonitoringPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold gradient-text">Monitoring</h1>
        <p className="text-gray-400 mt-2">
          Data drift detection and model performance monitoring powered by
          Evidently AI. Ensures prediction quality stays within acceptable
          bounds.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-card p-6">
          <p className="text-sm text-gray-400 mb-1">API Health</p>
          <p
            className={`text-2xl font-bold ${
              health?.status === "healthy"
                ? "text-brand-400"
                : "text-red-400"
            }`}
          >
            {health?.status === "healthy" ? "Healthy" : "Checking..."}
          </p>
        </div>
        <div className="glass-card p-6">
          <p className="text-sm text-gray-400 mb-1">Models Status</p>
          <p
            className={`text-2xl font-bold ${
              health?.models_loaded ? "text-brand-400" : "text-gold-400"
            }`}
          >
            {health?.models_loaded ? "Loaded" : "Not loaded"}
          </p>
        </div>
        <div className="glass-card p-6">
          <p className="text-sm text-gray-400 mb-1">Data Status</p>
          <p
            className={`text-2xl font-bold ${
              health?.data_fresh ? "text-brand-400" : "text-gold-400"
            }`}
          >
            {health?.data_fresh ? "Fresh" : "Stale"}
          </p>
        </div>
      </div>

      <div className="glass-card p-6">
        <h2 className="text-xl font-semibold mb-4">Drift Detection</h2>
        <p className="text-gray-400 text-sm mb-4">
          Data drift occurs when the statistical properties of incoming data
          diverge from the training data distribution. This is an early warning
          sign that model predictions may degrade — catching drift before it
          impacts trading decisions saves money.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-black/20 rounded-lg">
            <h3 className="text-sm font-semibold text-gray-200 mb-2">
              Monitored Features
            </h3>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>Soybean Meal (SMC1) distribution</li>
              <li>Soybeans (SC1) distribution</li>
              <li>Brent Crude (LCOC1) distribution</li>
              <li>Heating Oil (HOC1) distribution</li>
              <li>Palm Oil (FCPOC1) distribution</li>
              <li>Canola (RSC1) distribution</li>
            </ul>
          </div>
          <div className="p-4 bg-black/20 rounded-lg">
            <h3 className="text-sm font-semibold text-gray-200 mb-2">
              Alert Thresholds
            </h3>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>MAE &gt; 5.0 cents/lb</li>
              <li>RMSE &gt; 7.0 cents/lb</li>
              <li>R2 &lt; 0.50</li>
              <li>Retrain if 2+ thresholds breached</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="glass-card p-6">
        <h2 className="text-xl font-semibold mb-4">
          Retraining Pipeline
        </h2>
        <div className="text-sm text-gray-400 font-mono space-y-2">
          <p>
            <span className="text-brand-400">SCHEDULE</span> &rarr; Airflow
            DAG runs every Sunday 20:00 UTC
          </p>
          <p>
            <span className="text-blue-400">INGEST</span> &rarr; Fetch latest
            futures prices (incremental append)
          </p>
          <p>
            <span className="text-purple-400">TRANSFORM</span> &rarr; dbt run
            (bronze &rarr; silver &rarr; gold)
          </p>
          <p>
            <span className="text-gold-400">TRAIN</span> &rarr; XGBoost +
            walk-forward validation
          </p>
          <p>
            <span className="text-cyan-400">LOG</span> &rarr; MLflow
            experiment tracking
          </p>
          <p>
            <span className="text-red-400">MONITOR</span> &rarr; Evidently
            drift report
          </p>
        </div>
      </div>
    </div>
  );
}
