/**
 * FastAPI client for the Soybean Oil Predictor backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PredictionRequest {
  smc1: number;
  sc1: number;
  lcoc1: number;
  hoc1: number;
  fcpoc1: number;
  rsc1: number;
  month?: number;
}

export interface PredictionResponse {
  predicted_price: number;
  model_name: string;
  confidence: Record<string, number>;
  features_used: string[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelInfo {
  name: string;
  model_type: string;
  metrics: Record<string, number>;
  n_features: number;
  trained_at: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: boolean;
  data_fresh: boolean;
}

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }

  return res.json();
}

export const api = {
  health: () => fetchApi<HealthResponse>("/health"),

  predict: (data: PredictionRequest) =>
    fetchApi<PredictionResponse>("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  featureStats: () =>
    fetchApi<{ features: Record<string, Record<string, number>> }>(
      "/api/v1/features/stats"
    ),

  featureImportance: () =>
    fetchApi<{ model_name: string; importances: FeatureImportance[] }>(
      "/api/v1/features/importance"
    ),

  listModels: () =>
    fetchApi<{ models: ModelInfo[]; active_model: string }>("/api/v1/models"),
};
