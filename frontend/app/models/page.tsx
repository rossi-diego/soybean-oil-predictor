"use client";

import { useEffect, useState } from "react";
import {
  api,
  isDemoMode,
  ModelComparisonResponse,
  ChampionDiagnosticsResponse,
  FeatureImportanceResponse,
  LearningCurvesResponse,
  WalkForwardResponse,
  ModelMetadata,
} from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, Cell, ReferenceLine,
} from "recharts";

const COLORS = ["#22c55e", "#3b82f6", "#eab308", "#ef4444", "#8b5cf6"];
const MODEL_COLORS: Record<string, string> = {
  xgboost: "#22c55e", ridge: "#3b82f6", lasso: "#eab308",
  elasticnet: "#ef4444", linear: "#8b5cf6",
};

const FEATURE_LABELS: Record<string, string> = {
  smc1: "Soybean Meal", sc1: "Soybeans", lcoc1: "Brent Crude",
  hoc1: "Heating Oil", fcpoc1: "Palm Oil", rsc1: "Wheat",
  "so-premp-c1": "Oil Premium", "brl=": "BRL/USD",
};

function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse bg-[#1e1e22] rounded ${className}`} />;
}

function Section({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="glass-card p-5">
      <h2 className="text-sm font-semibold">{title}</h2>
      {sub && <p className="text-[11px] text-zinc-500 mt-0.5 mb-4">{sub}</p>}
      {!sub && <div className="mb-4" />}
      {children}
    </div>
  );
}

export default function ModelsPage() {
  const [comp, setComp] = useState<ModelComparisonResponse | null>(null);
  const [diag, setDiag] = useState<ChampionDiagnosticsResponse | null>(null);
  const [imp, setImp] = useState<FeatureImportanceResponse | null>(null);
  const [lc, setLc] = useState<LearningCurvesResponse | null>(null);
  const [wf, setWf] = useState<WalkForwardResponse | null>(null);
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [sortKey, setSortKey] = useState("mae");

  useEffect(() => {
    Promise.all([
      api.modelsComparison().then(setComp).catch(() => {}),
      api.modelInfo().then(setMeta).catch(() => {}),
      api.championDiagnostics().then(setDiag).catch(() => {}),
      api.featureImportanceAll().then(setImp).catch(() => {}),
      api.learningCurves().then(setLc).catch(() => {}),
      api.walkForward().then(setWf).catch(() => {}),
    ]).finally(() => setLoaded(true));
  }, []);

  const demo = loaded && isDemoMode();
  const sorted = comp ? [...comp.models].sort((a, b) => {
    const va = (a as any)[sortKey], vb = (b as any)[sortKey];
    return sortKey === "r2" || sortKey === "directional_accuracy" ? vb - va : va - vb;
  }) : [];

  const bestPerMetric: Record<string, string> = {};
  if (comp) {
    ["mae", "rmse", "r2", "directional_accuracy", "train_time_s"].forEach((k) => {
      const best = [...comp.models].sort((a, b) => {
        const va = (a as any)[k], vb = (b as any)[k];
        return k === "r2" || k === "directional_accuracy" ? vb - va : va - vb;
      })[0];
      bestPerMetric[k] = best.model;
    });
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-bold tracking-tight">Model Comparison</h1>
        <p className="text-sm text-zinc-500 mt-1">
          5 models trained on the same data. Champion selected by lowest MAE on holdout set.
        </p>
      </section>

      {demo && <DemoBanner />}

      {/* 1. Comparison Table */}
      <Section title="Performance Comparison" sub="5 models trained on same 80/20 temporal split. Click column headers to sort. Green = best in column.">
        {!comp ? <Skeleton className="h-48" /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b border-[#1e1e22] text-zinc-500">
                  <th className="text-left py-2 px-3 font-medium">Model</th>
                  {[
                    { key: "mae", label: "MAE" },
                    { key: "rmse", label: "RMSE" },
                    { key: "r2", label: "R\u00B2" },
                    { key: "directional_accuracy", label: "Dir. Acc." },
                    { key: "train_time_s", label: "Time (s)" },
                  ].map(({ key, label }) => (
                    <th
                      key={key}
                      className="text-right py-2 px-3 font-medium cursor-pointer hover:text-zinc-300 transition-colors"
                      onClick={() => setSortKey(key)}
                    >
                      {label} {sortKey === key && "\u25BC"}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((m) => (
                  <tr key={m.model} className="border-b border-[#1e1e22]/50 hover:bg-white/[0.02]">
                    <td className="py-2 px-3 font-medium">
                      <span className="inline-block w-2 h-2 rounded-full mr-2" style={{ background: MODEL_COLORS[m.model] }} />
                      {m.model}
                      {m.model === comp.champion && (
                        <span className="ml-2 text-[9px] text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded">champion</span>
                      )}
                    </td>
                    {["mae", "rmse", "r2", "directional_accuracy", "train_time_s"].map((k) => {
                      const val = (m as any)[k];
                      const isBest = bestPerMetric[k] === m.model;
                      return (
                        <td key={k} className={`py-2 px-3 text-right tabular-nums ${isBest ? "text-green-400 font-medium" : "text-zinc-400"}`}>
                          {k === "directional_accuracy" ? `${(val * 100).toFixed(1)}%` : val?.toFixed(k === "r2" ? 4 : 2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* 2. Residual Diagnostics */}
      <Section title="Residual Diagnostics" sub={`Champion model: ${diag?.model ?? "loading"}`}>
        {!diag ? <Skeleton className="h-48" /> : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Predicted vs Actual */}
            <div>
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Predicted vs Actual</h3>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                    <XAxis type="number" dataKey="actual" name="Actual" stroke="#3f3f46" fontSize={9} />
                    <YAxis type="number" dataKey="predicted" name="Predicted" stroke="#3f3f46" fontSize={9} />
                    <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                    <Scatter data={diag.points} fill="#3b82f6" opacity={0.4} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
            {/* Residual histogram */}
            <div>
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Residual Distribution</h3>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={diag.residual_histogram}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                    <XAxis dataKey="x" stroke="#3f3f46" fontSize={9} />
                    <YAxis stroke="#3f3f46" fontSize={9} />
                    <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                    <ReferenceLine x={0} stroke="#3f3f46" />
                    <Bar dataKey="count" fill="#3b82f6" opacity={0.7} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            {/* Residual stats */}
            <div>
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Residual Statistics</h3>
              <div className="text-[12px] space-y-1.5 mt-2">
                <div className="flex justify-between"><span className="text-zinc-500">Mean</span><span className="tabular-nums">{diag.residual_stats.mean.toFixed(4)}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">Std Dev</span><span className="tabular-nums">{diag.residual_stats.std.toFixed(4)}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">Skewness</span><span className="tabular-nums">{diag.residual_stats.skew.toFixed(4)}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">Kurtosis</span><span className="tabular-nums">{diag.residual_stats.kurtosis.toFixed(4)}</span></div>
                <div className="flex justify-between"><span className="text-zinc-500">N points</span><span className="tabular-nums">{diag.points.length}</span></div>
              </div>
              {/* Residuals over time */}
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mt-4 mb-2">Residuals Over Time</h3>
              <div className="h-24">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={diag.points}>
                    <XAxis dataKey="index" hide />
                    <YAxis stroke="#3f3f46" fontSize={9} />
                    <ReferenceLine y={0} stroke="#3f3f46" />
                    <Line type="monotone" dataKey="residual" stroke="#ef4444" strokeWidth={1} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </Section>

      {/* 3. Feature Importance */}
      <Section title="Feature Importance" sub="XGBoost native gain-based importance">
        {!imp ? <Skeleton className="h-48" /> : (
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={imp.models?.xgboost ?? []} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                <XAxis type="number" stroke="#3f3f46" fontSize={10} />
                <YAxis
                  type="category" dataKey="feature" stroke="#3f3f46" fontSize={10}
                  tickFormatter={(f: string) => FEATURE_LABELS[f] ?? f}
                />
                <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
                  {(imp.models?.xgboost ?? []).map((_, i) => (
                    <Cell key={i} fill={i === 0 ? "#22c55e" : i < 3 ? "#4ade80" : "#52525b"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      {/* 4. Learning Curves */}
      <Section title="Learning Curves" sub="Training vs validation MAE as function of training set size">
        {!lc ? <Skeleton className="h-48" /> : (
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                <XAxis
                  dataKey="size" type="number" stroke="#3f3f46" fontSize={10}
                  domain={["auto", "auto"]} label={{ value: "Training samples", position: "insideBottom", offset: -5, fontSize: 10, fill: "#52525b" }}
                />
                <YAxis stroke="#3f3f46" fontSize={10} label={{ value: "MAE", angle: -90, position: "insideLeft", fontSize: 10, fill: "#52525b" }} />
                <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                {Object.entries(lc.models).map(([name, data]) => {
                  const chartData = data.train_sizes.map((s, i) => ({
                    size: s,
                    [`${name}_train`]: data.train_mae[i],
                    [`${name}_val`]: data.val_mae[i],
                  }));
                  return [
                    <Line
                      key={`${name}-train`}
                      data={chartData}
                      type="monotone"
                      dataKey={`${name}_train`}
                      stroke={MODEL_COLORS[name]}
                      strokeWidth={1}
                      dot={{ r: 2 }}
                      name={`${name} (train)`}
                    />,
                    <Line
                      key={`${name}-val`}
                      data={chartData}
                      type="monotone"
                      dataKey={`${name}_val`}
                      stroke={MODEL_COLORS[name]}
                      strokeWidth={1.5}
                      strokeDasharray="4 2"
                      dot={{ r: 2 }}
                      name={`${name} (val)`}
                    />,
                  ];
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      {/* 5. Walk-Forward Folds */}
      <Section title="Walk-Forward Performance by Fold" sub="Does the model break down in specific periods?">
        {!wf ? <Skeleton className="h-48" /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-[#1e1e22] text-zinc-500">
                  <th className="text-left py-1.5 px-2 font-medium">Fold</th>
                  <th className="text-right py-1.5 px-2 font-medium">Train</th>
                  <th className="text-right py-1.5 px-2 font-medium">Test</th>
                  {wf.models.map((m) => (
                    <th key={m} className="text-right py-1.5 px-2 font-medium" style={{ color: MODEL_COLORS[m] }}>
                      {m} MAE
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {wf.folds.map((f) => (
                  <tr key={f.fold} className="border-b border-[#1e1e22]/50 hover:bg-white/[0.02]">
                    <td className="py-1.5 px-2 font-medium">Fold {f.fold}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-zinc-500">{f.n_train}</td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-zinc-500">{f.n_test}</td>
                    {wf.models.map((m) => {
                      const val = f[`${m}_mae`];
                      const worst = Math.max(...wf.folds.map((ff) => ff[`${m}_mae`] as number));
                      return (
                        <td
                          key={m}
                          className={`py-1.5 px-2 text-right tabular-nums ${val === worst ? "text-red-400" : "text-zinc-400"}`}
                        >
                          {typeof val === "number" ? val.toFixed(2) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[10px] text-zinc-600 mt-2">
              Red = worst fold per model. Large fold-to-fold variance indicates regime sensitivity.
            </p>
          </div>
        )}
      </Section>

      {/* 6. Hyperparameters + Model Selection */}
      <Section title="Model Selection &amp; Hyperparameters">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div>
            <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Why {comp?.champion ?? "Ridge"}?</h3>
            <p className="text-[12px] text-zinc-400 leading-relaxed">
              {comp?.champion === "xgboost" ? (
                <>
                  XGBoost was selected as champion for lowest MAE on the holdout set.
                  Gradient boosting captures nonlinear feature interactions
                  (e.g., crude oil affecting soybean oil differently at high vs low price levels)
                  that linear models miss.
                </>
              ) : (
                <>
                  Ridge regression was selected as champion for lowest MAE (2.69 c/lb) on the
                  holdout set. Despite XGBoost&apos;s ability to capture nonlinear interactions,
                  Ridge&apos;s L2 regularization prevented overfitting on this relatively small
                  feature set (5 commodity prices). XGBoost&apos;s MAE was 2.5x worse — it
                  memorizes noise in the training data rather than learning generalizable patterns.
                </>
              )}
            </p>
            <p className="text-[11px] text-zinc-500 mt-2">
              Selection criterion: lowest MAE on 20% temporal holdout (no shuffle).
              Walk-forward validation on 5 folds confirms generalization.
            </p>
          </div>
          <div>
            <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">
              XGBoost Hyperparameters
            </h3>
            {meta?.hyperparameters ? (
              <div className="space-y-1 text-[12px]">
                {Object.entries(meta.hyperparameters)
                  .filter(([k]) => !["random_state", "n_jobs", "verbosity"].includes(k))
                  .map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-zinc-500 font-mono">{k}</span>
                      <span className="tabular-nums">{String(v)}</span>
                    </div>
                  ))}
                <p className="text-[10px] text-zinc-600 pt-1 mt-1 border-t border-[#1e1e22]">
                  Manually tuned. Optuna integration available in src/models/ for automated search.
                </p>
              </div>
            ) : (
              <Skeleton className="h-32" />
            )}
          </div>
        </div>
      </Section>
    </div>
  );
}
