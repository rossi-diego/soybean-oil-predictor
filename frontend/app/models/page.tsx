"use client";

import { useEffect, useState } from "react";
import { api, ModelInfo } from "@/lib/api";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeModel, setActiveModel] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listModels()
      .then((data) => {
        setModels(data.models);
        setActiveModel(data.active_model);
      })
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold gradient-text">Model Comparison</h1>
        <p className="text-gray-400 mt-2">
          Compare trained models side-by-side. Per quant best practices,
          we always start with an XGBoost baseline and compare against
          linear and time-series alternatives.
        </p>
      </div>

      {error && (
        <div className="glass-card p-4 border-gold-500/30 bg-gold-500/5">
          <p className="text-gold-400 text-sm">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {models.map((model) => (
          <div
            key={model.name}
            className={`glass-card p-6 ${
              model.name === activeModel
                ? "border-brand-500/40 bg-brand-500/5"
                : ""
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold">{model.name}</h3>
              {model.name === activeModel && (
                <span className="text-xs bg-brand-500/20 text-brand-400 px-2 py-0.5 rounded-full">
                  Active
                </span>
              )}
            </div>
            <p className="text-sm text-gray-400 mb-4">{model.model_type}</p>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Features</span>
                <span>{model.n_features}</span>
              </div>
              {Object.entries(model.metrics).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="text-gray-500">{key.toUpperCase()}</span>
                  <span>{typeof value === "number" ? value.toFixed(4) : value}</span>
                </div>
              ))}
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Trained</span>
                <span className="text-gray-400">
                  {model.trained_at
                    ? new Date(model.trained_at).toLocaleDateString()
                    : "Unknown"}
                </span>
              </div>
            </div>
          </div>
        ))}

        {models.length === 0 && !error && (
          <div className="glass-card p-8 col-span-full text-center">
            <p className="text-gray-500">
              No models available. Run the training pipeline to generate models.
            </p>
          </div>
        )}
      </div>

      <div className="glass-card p-6">
        <h2 className="text-xl font-semibold mb-4">Evaluation Methodology</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-gray-400">
          <div>
            <h3 className="text-gray-200 font-medium mb-2">
              Metrics Reported
            </h3>
            <ul className="space-y-1">
              <li>
                <strong>MAE</strong> — Mean Absolute Error (interpretable, in
                cents/lb)
              </li>
              <li>
                <strong>RMSE</strong> — Root Mean Squared Error (penalizes
                large errors)
              </li>
              <li>
                <strong>R2</strong> — Coefficient of determination (explained
                variance)
              </li>
              <li>
                <strong>Directional Accuracy</strong> — % of correct up/down
                predictions (key for trading)
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-gray-200 font-medium mb-2">Validation</h3>
            <ul className="space-y-1">
              <li>
                Walk-forward validation (no look-ahead bias)
              </li>
              <li>
                TimeSeriesSplit for temporal ordering
              </li>
              <li>
                All metrics on out-of-sample data only
              </li>
              <li>
                SHAP explainability for feature contribution
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
