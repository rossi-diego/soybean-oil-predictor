/**
 * FastAPI client for the Soybean Oil Predictor backend.
 * Falls back to mock data when the API is unreachable (demo mode).
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

export interface LivePrice {
  name: string;
  ticker: string;
  price: number;
  currency: string;
  timestamp: string;
}

export interface LivePricesResponse {
  prices: LivePrice[];
  cached: boolean;
  fetched_at: string;
}

export interface LivePredictionResponse {
  predicted_price: number;
  model_name: string;
  input_prices: Record<string, number>;
  cached: boolean;
  fetched_at: string;
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

// ---------------------------------------------------------------------------
// Mock data for demo mode (when API is unreachable)
// ---------------------------------------------------------------------------

const MOCK_HEALTH: HealthResponse = {
  status: "demo",
  version: "0.2.0",
  models_loaded: true,
  data_fresh: true,
};

const MOCK_MODELS: ModelInfo[] = [
  {
    name: "xgboost_baseline",
    model_type: "XGBRegressor",
    metrics: { r2: 0.9412, mae: 1.83, rmse: 2.47, directional_accuracy: 0.72 },
    n_features: 12,
    trained_at: "2025-03-15T14:30:00",
  },
  {
    name: "ridge_regression",
    model_type: "Ridge",
    metrics: { r2: 0.8876, mae: 2.51, rmse: 3.18, directional_accuracy: 0.65 },
    n_features: 12,
    trained_at: "2025-03-15T14:28:00",
  },
  {
    name: "elasticnet",
    model_type: "ElasticNet",
    metrics: { r2: 0.8734, mae: 2.67, rmse: 3.38, directional_accuracy: 0.63 },
    n_features: 12,
    trained_at: "2025-03-15T14:29:00",
  },
];

const MOCK_LIVE_PRICES: LivePricesResponse = {
  prices: [
    { name: "boc1", ticker: "BO=F", price: 42.35, currency: "USD", timestamp: new Date().toISOString() },
    { name: "sc1", ticker: "ZS=F", price: 1142.50, currency: "USD", timestamp: new Date().toISOString() },
    { name: "smc1", ticker: "ZM=F", price: 362.80, currency: "USD", timestamp: new Date().toISOString() },
    { name: "lcoc1", ticker: "BZ=F", price: 72.15, currency: "USD", timestamp: new Date().toISOString() },
    { name: "hoc1", ticker: "HO=F", price: 2.18, currency: "USD", timestamp: new Date().toISOString() },
    { name: "fcpoc1", ticker: "FCPO=F", price: 3420.00, currency: "USD", timestamp: new Date().toISOString() },
    { name: "rsc1", ticker: "RS=F", price: 615.20, currency: "USD", timestamp: new Date().toISOString() },
  ],
  cached: true,
  fetched_at: new Date().toISOString(),
};

const MOCK_LIVE_PREDICTION: LivePredictionResponse = {
  predicted_price: 43.12,
  model_name: "xgboost_baseline (demo)",
  input_prices: { smc1: 362.80, sc1: 1142.50, lcoc1: 72.15, hoc1: 2.18, fcpoc1: 3420.00, rsc1: 615.20 },
  cached: true,
  fetched_at: new Date().toISOString(),
};

const MOCK_FEATURE_STATS = {
  features: {
    boc1:   { mean: 41.00, std: 13.86, min: 24.99, max: 90.60 },
    smc1:   { mean: 356.58, std: 61.35, min: 257.20, max: 521.90 },
    sc1:    { mean: 1119.44, std: 237.51, min: 791.00, max: 1769.00 },
    lcoc1:  { mean: 69.91, std: 20.72, min: 19.99, max: 127.98 },
    hoc1:   { mean: 2.15, std: 0.70, min: 0.61, max: 5.14 },
    fcpoc1: { mean: 3166.84, std: 1105.07, min: 1759.00, max: 7821.00 },
    rsc1:   { mean: 591.19, std: 178.30, min: 394.00, max: 1226.00 },
  },
};

const MOCK_IMPORTANCES: FeatureImportance[] = [
  { feature: "smc1", importance: 0.284 },
  { feature: "hoc1", importance: 0.197 },
  { feature: "lcoc1", importance: 0.143 },
  { feature: "fcpoc1", importance: 0.121 },
  { feature: "sc1", importance: 0.098 },
  { feature: "rsc1", importance: 0.067 },
  { feature: "crush_spread", importance: 0.042 },
  { feature: "oil_share", importance: 0.028 },
  { feature: "month_sin", importance: 0.012 },
  { feature: "month_cos", importance: 0.008 },
];

// ---------------------------------------------------------------------------
// API client with demo fallback
// ---------------------------------------------------------------------------

let _isDemoMode = false;

export function isDemoMode(): boolean {
  return _isDemoMode;
}

async function fetchApi<T>(
  path: string,
  options?: RequestInit,
  mockFallback?: T,
): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(error.detail || `API error: ${res.status}`);
    }

    _isDemoMode = false;
    return res.json();
  } catch {
    _isDemoMode = true;
    if (mockFallback !== undefined) {
      return mockFallback;
    }
    throw new Error("API unreachable — showing demo data");
  }
}

function mockPredict(data: PredictionRequest): PredictionResponse {
  const base = 0.08 * data.smc1 + 0.005 * data.sc1 + 0.12 * data.lcoc1
    + 8.5 * data.hoc1 + 0.002 * data.fcpoc1 + 0.01 * data.rsc1;
  const noise = (Math.sin(data.month || 1) * 2);
  return {
    predicted_price: Math.round((base / 6 + noise) * 100) / 100,
    model_name: "xgboost_baseline (demo)",
    confidence: {},
    features_used: ["smc1", "sc1", "lcoc1", "hoc1", "fcpoc1", "rsc1"],
  };
}

export const api = {
  health: () => fetchApi<HealthResponse>("/health", undefined, MOCK_HEALTH),

  predict: (data: PredictionRequest) =>
    fetchApi<PredictionResponse>(
      "/api/v1/predict",
      { method: "POST", body: JSON.stringify(data) },
      mockPredict(data),
    ),

  featureStats: () =>
    fetchApi<{ features: Record<string, Record<string, number>> }>(
      "/api/v1/features/stats",
      undefined,
      MOCK_FEATURE_STATS,
    ),

  featureImportance: () =>
    fetchApi<{ model_name: string; importances: FeatureImportance[] }>(
      "/api/v1/features/importance",
      undefined,
      { model_name: "xgboost_baseline (demo)", importances: MOCK_IMPORTANCES },
    ),

  listModels: () =>
    fetchApi<{ models: ModelInfo[]; active_model: string }>(
      "/api/v1/models",
      undefined,
      { models: MOCK_MODELS, active_model: "xgboost_baseline" },
    ),

  livePrices: () =>
    fetchApi<LivePricesResponse>(
      "/api/v1/prices/latest",
      undefined,
      MOCK_LIVE_PRICES,
    ),

  predictLive: () =>
    fetchApi<LivePredictionResponse>(
      "/api/v1/predict/live",
      undefined,
      MOCK_LIVE_PREDICTION,
    ),
};
