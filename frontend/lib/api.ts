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
  rsc1: number;
  month?: number;
}

export interface FeatureContribution {
  feature: string;
  contribution: number;
}

export interface PredictionResponse {
  predicted_price: number;
  model_name: string;
  confidence: { lower?: number; upper?: number };
  features_used: string[];
  feature_contributions: FeatureContribution[];
}

export interface ModelMetadata {
  algorithm: string;
  n_training_samples: number;
  n_total_samples: number;
  n_features: number;
  feature_names: string[];
  residual_std: number;
  trained_at: string;
  backtest_metrics: {
    mae: number;
    rmse: number;
    r2: number;
    directional_accuracy: number;
    n_predictions: number;
    n_folds: number;
  };
  hyperparameters: Record<string, any>;
  feature_ranges: Record<string, { min: number; max: number; mean: number; p5: number; p95: number }>;
}

export interface LivePrice {
  name: string;
  ticker: string;
  price: number;
  prev_close: number;
  change: number;
  change_pct: number;
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

export interface SpreadSignal {
  name: string;
  label: string;
  value: number;
  unit: string;
  ma30: number;
  deviation_pct: number;
  trend: "up" | "down" | "flat";
  interpretation: string;
  signal: "bullish" | "bearish" | "neutral";
}

export interface SpreadsResponse {
  spreads: SpreadSignal[];
}

export interface BacktestPoint {
  date: string;
  actual: number;
  predicted: number;
  residual: number;
  lower: number;
  upper: number;
  fold: number;
}

export interface FoldBoundary {
  fold: number;
  start_date: string;
}

// --- EDA types ---
export interface EdaPricesResponse {
  dates: string[];
  series: Record<string, (number | null)[]>;
  normalized: Record<string, (number | null)[]>;
  n_points: number;
  labels: Record<string, string>;
}

export interface EdaCorrelationsResponse {
  features: string[];
  matrix: number[][];
  method: string;
  labels: Record<string, string>;
}

export interface EdaDistributionBin { x: number; x_end: number; count: number; recent_count: number }
export interface EdaQuartiles { min: number; q1: number; median: number; q3: number; max: number }
export interface EdaFeatureDistribution {
  bins: EdaDistributionBin[];
  quartiles: EdaQuartiles;
  mean: number; std: number; n: number; recent_n: number;
}
export interface EdaDistributionsResponse {
  features: Record<string, EdaFeatureDistribution>;
  labels: Record<string, string>;
}

export interface EdaSpreadSeries {
  values: (number | null)[];
  label: string; unit: string;
  p10: number; p90: number; mean: number;
}
export interface EdaSpreadsResponse {
  dates: string[];
  spreads: Record<string, EdaSpreadSeries>;
}

export interface EdaMonthStats {
  month: number; name: string;
  min: number; q1: number; median: number; q3: number; max: number;
  mean: number; n: number;
}
export interface EdaSeasonalityResponse { months: EdaMonthStats[] }

export interface EdaAdfResult {
  statistic: number; p_value: number; lags: number;
  stationary: boolean; critical_values: Record<string, number>;
}
export interface EdaStationarityResponse {
  adf_tests: Record<string, EdaAdfResult>;
  returns_hist: { x: number; x_end: number; count: number }[];
  returns_stats: { mean: number; std: number; skew: number; kurtosis: number };
  acf: { lag: number; value: number }[];
  pacf: { lag: number; value: number }[];
  confidence_interval: number;
  labels: Record<string, string>;
}

// --- Models page types ---
export interface ModelComparisonEntry {
  model: string; mae: number; rmse: number; r2: number;
  directional_accuracy: number; train_time_s: number;
}
export interface ModelComparisonResponse { models: ModelComparisonEntry[]; champion: string }

export interface DiagnosticPoint { actual: number; predicted: number; residual: number; index: number }
export interface ChampionDiagnosticsResponse {
  model: string;
  points: DiagnosticPoint[];
  residual_stats: { mean: number; std: number; skew: number; kurtosis: number };
  residual_histogram: { x: number; count: number }[];
}

export interface FeatureImportanceEntry { feature: string; importance: number; coefficient?: number }
export interface FeatureImportanceResponse { models: Record<string, FeatureImportanceEntry[]> }

export interface LearningCurveModel { train_sizes: number[]; train_mae: number[]; val_mae: number[] }
export interface LearningCurvesResponse { models: Record<string, LearningCurveModel> }

export interface WalkForwardFold { fold: number; n_train: number; n_test: number; [key: string]: number }
export interface WalkForwardResponse { folds: WalkForwardFold[]; champion: string; models: string[] }

export interface BacktestResponse {
  points: BacktestPoint[];
  fold_boundaries: FoldBoundary[];
  metrics: {
    mae: number;
    rmse: number;
    r2: number;
    directional_accuracy: number;
  };
  n_points: number;
  n_folds: number;
  model: string;
}

export interface PriceHistoryPoint {
  date: string;
  actual: number;
  predicted: number | null;
}

export interface PriceHistoryResponse {
  history: PriceHistoryPoint[];
  model_name: string;
  days: number;
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

const MOCK_MODEL_METADATA: ModelMetadata = {
  algorithm: "XGBRegressor",
  n_training_samples: 2016,
  n_total_samples: 2520,
  n_features: 8,
  feature_names: ["smc1", "sc1", "lcoc1", "hoc1", "rsc1"],
  residual_std: 5.87,
  trained_at: "2025-03-15T14:30:00",
  backtest_metrics: {
    mae: 3.83,
    rmse: 6.16,
    r2: 0.8266,
    directional_accuracy: 0.619,
    n_predictions: 2100,
    n_folds: 5,
  },
  hyperparameters: {
    n_estimators: 500,
    max_depth: 6,
    learning_rate: 0.05,
    subsample: 0.8,
  },
  feature_ranges: {
    smc1: { min: 257.2, max: 521.9, mean: 356.58, p5: 285.6, p95: 471.21 },
    sc1: { min: 791.0, max: 1769.0, mean: 1119.44, p5: 856.99, p95: 1544.11 },
    lcoc1: { min: 19.99, max: 127.98, mean: 69.91, p5: 39.84, p95: 108.84 },
    hoc1: { min: 0.61, max: 5.14, mean: 2.15, p5: 1.14, p95: 3.41 },
    rsc1: { min: 394.0, max: 1226.0, mean: 591.19, p5: 439.09, p95: 996.93 },
  },
};

const MOCK_SPREADS: SpreadsResponse = {
  spreads: [
    {
      name: "crush_spread",
      label: "Crush Spread",
      value: 142.30,
      unit: "$/ton",
      ma30: 128.50,
      deviation_pct: 10.7,
      trend: "up",
      interpretation: "Crush margin 11% above 30d avg — bullish for processor demand",
      signal: "bullish",
    },
    {
      name: "oil_palm_spread",
      label: "Oil / Palm Spread",
      value: -35.15,
      unit: "c/lb",
      ma30: -32.80,
      deviation_pct: -7.2,
      trend: "down",
      interpretation: "Spread narrowing — soy oil gaining competitive advantage",
      signal: "bearish",
    },
  ],
};

function generateMockBacktest(): BacktestResponse {
  const points: BacktestPoint[] = [];
  let price = 38;
  const start = new Date("2016-01-01");
  for (let i = 0; i < 200; i++) {
    const d = new Date(start);
    d.setDate(d.getDate() + i * 5);
    price += (Math.random() - 0.48) * 1.5;
    price = Math.max(28, Math.min(65, price));
    const predicted = price + (Math.random() - 0.5) * 8;
    const std = 4.5;
    const fold = Math.floor(i / 40);
    points.push({
      date: d.toISOString().slice(0, 10),
      actual: Math.round(price * 100) / 100,
      predicted: Math.round(predicted * 100) / 100,
      residual: Math.round((price - predicted) * 100) / 100,
      lower: Math.round((predicted - 1.96 * std) * 100) / 100,
      upper: Math.round((predicted + 1.96 * std) * 100) / 100,
      fold,
    });
  }
  return {
    points,
    fold_boundaries: [
      { fold: 0, start_date: "2016-01-01" },
      { fold: 1, start_date: "2016-07-01" },
      { fold: 2, start_date: "2017-01-01" },
      { fold: 3, start_date: "2017-07-01" },
      { fold: 4, start_date: "2018-01-01" },
    ],
    metrics: { mae: 2.69, rmse: 3.53, r2: 0.8138, directional_accuracy: 0.763 },
    n_points: points.length,
    n_folds: 5,
    model: "ridge",
  };
}

const MOCK_BACKTEST = generateMockBacktest();

const MOCK_LIVE_PRICES: LivePricesResponse = {
  prices: [
    { name: "boc1", ticker: "ZL=F", price: 42.35, prev_close: 41.80, change: 0.55, change_pct: 1.32, currency: "USD", timestamp: new Date().toISOString() },
    { name: "sc1", ticker: "ZS=F", price: 1142.50, prev_close: 1148.25, change: -5.75, change_pct: -0.50, currency: "USD", timestamp: new Date().toISOString() },
    { name: "smc1", ticker: "ZM=F", price: 362.80, prev_close: 358.40, change: 4.40, change_pct: 1.23, currency: "USD", timestamp: new Date().toISOString() },
    { name: "lcoc1", ticker: "BZ=F", price: 72.15, prev_close: 73.02, change: -0.87, change_pct: -1.19, currency: "USD", timestamp: new Date().toISOString() },
    { name: "hoc1", ticker: "HO=F", price: 2.18, prev_close: 2.15, change: 0.03, change_pct: 1.40, currency: "USD", timestamp: new Date().toISOString() },
    { name: "rsc1", ticker: "ZW=F", price: 615.20, prev_close: 612.50, change: 2.70, change_pct: 0.44, currency: "USD", timestamp: new Date().toISOString() },
  ],
  cached: true,
  fetched_at: new Date().toISOString(),
};

const MOCK_LIVE_PREDICTION: LivePredictionResponse = {
  predicted_price: 43.12,
  model_name: "xgboost_baseline (demo)",
  input_prices: { smc1: 362.80, sc1: 1142.50, lcoc1: 72.15, hoc1: 2.18, rsc1: 615.20 },
  cached: true,
  fetched_at: new Date().toISOString(),
};

function generateMockHistory(): PriceHistoryResponse {
  const history: PriceHistoryPoint[] = [];
  let price = 42;
  const today = new Date();
  for (let i = 60; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    price += (Math.random() - 0.48) * 1.2;
    price = Math.max(35, Math.min(55, price));
    const predicted = price + (Math.random() - 0.5) * 3;
    history.push({
      date: d.toISOString().slice(0, 10),
      actual: Math.round(price * 100) / 100,
      predicted: Math.round(predicted * 100) / 100,
    });
  }
  return { history, model_name: "xgboost_baseline (demo)", days: history.length };
}

const MOCK_HISTORY = generateMockHistory();

const MOCK_FEATURE_STATS = {
  features: {
    boc1:   { mean: 41.00, std: 13.86, min: 24.99, max: 90.60 },
    smc1:   { mean: 356.58, std: 61.35, min: 257.20, max: 521.90 },
    sc1:    { mean: 1119.44, std: 237.51, min: 791.00, max: 1769.00 },
    lcoc1:  { mean: 69.91, std: 20.72, min: 19.99, max: 127.98 },
    hoc1:   { mean: 2.15, std: 0.70, min: 0.61, max: 5.14 },
    rsc1:   { mean: 591.19, std: 178.30, min: 394.00, max: 1226.00 },
  },
};

const MOCK_IMPORTANCES: FeatureImportance[] = [
  { feature: "smc1", importance: 0.284 },
  { feature: "hoc1", importance: 0.197 },
  { feature: "lcoc1", importance: 0.143 },
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
    + 8.5 * data.hoc1 + 0.01 * data.rsc1;
  const noise = (Math.sin(data.month || 1) * 2);
  const price = Math.round((base / 6 + noise) * 100) / 100;
  return {
    predicted_price: price,
    model_name: "xgboost_baseline (demo)",
    confidence: { lower: Math.round((price - 11.5) * 100) / 100, upper: Math.round((price + 11.5) * 100) / 100 },
    features_used: ["smc1", "sc1", "lcoc1", "hoc1", "rsc1"],
    feature_contributions: [
      { feature: "hoc1", contribution: 4.82 },
      { feature: "smc1", contribution: 3.15 },
      { feature: "lcoc1", contribution: -2.41 },
      { feature: "sc1", contribution: -1.23 },
      { feature: "rsc1", contribution: 0.64 },
    ],
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

  priceHistory: (days = 90) =>
    fetchApi<PriceHistoryResponse>(
      `/api/v1/prices/history?days=${days}`,
      undefined,
      MOCK_HISTORY,
    ),

  spreads: () =>
    fetchApi<SpreadsResponse>(
      "/api/v1/spreads",
      undefined,
      MOCK_SPREADS,
    ),

  backtest: () =>
    fetchApi<BacktestResponse>(
      "/api/v1/backtest",
      undefined,
      MOCK_BACKTEST,
    ),

  modelInfo: () =>
    fetchApi<ModelMetadata>(
      "/api/v1/model/info",
      undefined,
      MOCK_MODEL_METADATA,
    ),

  modelsComparison: () =>
    fetchApi<ModelComparisonResponse>("/api/v1/models/comparison"),

  championDiagnostics: () =>
    fetchApi<ChampionDiagnosticsResponse>("/api/v1/models/champion/diagnostics"),

  featureImportanceAll: () =>
    fetchApi<FeatureImportanceResponse>("/api/v1/models/champion/feature-importance"),

  learningCurves: () =>
    fetchApi<LearningCurvesResponse>("/api/v1/models/learning-curves"),

  walkForward: () =>
    fetchApi<WalkForwardResponse>("/api/v1/models/walk-forward"),

  edaPrices: (period = "all") =>
    fetchApi<EdaPricesResponse>(`/api/v1/eda/prices?period=${period}`),

  edaCorrelations: (method = "pearson") =>
    fetchApi<EdaCorrelationsResponse>(`/api/v1/eda/correlations?method=${method}`),

  edaDistributions: () =>
    fetchApi<EdaDistributionsResponse>("/api/v1/eda/distributions"),

  edaSpreads: () =>
    fetchApi<EdaSpreadsResponse>("/api/v1/eda/spreads"),

  edaSeasonality: () =>
    fetchApi<EdaSeasonalityResponse>("/api/v1/eda/seasonality"),

  edaStationarity: () =>
    fetchApi<EdaStationarityResponse>("/api/v1/eda/stationarity"),
};
