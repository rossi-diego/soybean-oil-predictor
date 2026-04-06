"use client";

import { useEffect, useState } from "react";
import {
  api,
  isDemoMode,
  HealthResponse,
  LivePrice,
  LivePredictionResponse,
  SpreadSignal,
  BacktestResponse,
} from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";
import {
  Area,
  ComposedChart,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";

const LABELS: Record<string, string> = {
  boc1: "Soybean Oil",
  sc1: "Soybeans",
  smc1: "Soybean Meal",
  lcoc1: "Brent Crude",
  hoc1: "Heating Oil",
  palm_oil: "Palm Oil",
  rsc1: "Wheat",
  zc1: "Corn",
};

const UNITS: Record<string, string> = {
  boc1: "c/lb",
  sc1: "c/bu",
  smc1: "$/ton",
  lcoc1: "$/bbl",
  hoc1: "$/gal",
  palm_oil: "USD/mt",
  rsc1: "c/bu",
  zc1: "c/bu",
};

function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} />;
}

const TREND_ICON: Record<string, string> = { up: "\u25B2", down: "\u25BC", flat: "\u2192" };
const SIGNAL_BORDER: Record<string, string> = {
  bullish: "border-l-green-500",
  bearish: "border-l-red-500",
  neutral: "border-l-yellow-500",
};
const SIGNAL_TEXT: Record<string, string> = {
  bullish: "text-green-400",
  bearish: "text-red-400",
  neutral: "text-yellow-500",
};

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [comp, setComp] = useState<{ models: any[]; champion: string } | null>(null);
  const [prices, setPrices] = useState<LivePrice[]>([]);
  const [livePred, setLivePred] = useState<LivePredictionResponse | null>(null);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [spreads, setSpreads] = useState<SpreadSignal[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [showPriceTable, setShowPriceTable] = useState(false);

  useEffect(() => {
    Promise.all([
      api.health().then(setHealth),
      api.modelsComparison().then(setComp).catch(() => api.listModels().then((d) => setComp({ models: d.models, champion: d.active_model })).catch(() => {})),
      api.livePrices().then((d) => setPrices(d.prices)),
      api.predictLive().then(setLivePred),
      api.backtest().then(setBacktest),
      api.spreads().then((d) => setSpreads(d.spreads)),
    ]).finally(() => setLoaded(true));
  }, []);

  const demo = loaded && isDemoMode();
  const boc1 = prices.find((p) => p.name === "boc1")?.price;
  const diff = livePred && boc1 ? livePred.predicted_price - boc1 : null;

  return (
    <div className="space-y-6">
      {demo && <DemoBanner />}

      {/* Hero */}
      <section>
        <h1 className="text-2xl font-bold tracking-tight">
          BOC1 Soybean Oil Futures
        </h1>
        <p className="text-sm text-zinc-500 mt-1">
          Front-month contract forecast using commodity cross-correlations,
          crush economics, and Ridge regression. 56% directional accuracy on out-of-sample data.
        </p>
      </section>

      {/* 3x2 Key metrics grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
        {/* BOC1 Last */}
        <div className="glass-card p-4">
          <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">BOC1 Last</p>
          {!loaded ? <Skeleton className="h-8 w-24 mt-1.5" /> : (
            <p className="text-2xl font-bold mt-1 text-green-400">{boc1?.toFixed(2) ?? "N/A"}</p>
          )}
          <p className="text-[10px] text-zinc-600 mt-0.5">cents/lb, CBOT</p>
        </div>

        {/* Model Forecast */}
        <div className="glass-card p-4">
          <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">Model Forecast</p>
          {!loaded ? <Skeleton className="h-8 w-24 mt-1.5" /> : (
            <>
              <div className="flex items-baseline gap-2 mt-1">
                <p className="text-2xl font-bold text-blue-400">{livePred?.predicted_price.toFixed(2) ?? "N/A"}</p>
                {diff !== null && boc1 != null && (
                  <span className={`text-xs font-medium ${diff > 0 ? "text-green-400" : "text-red-400"}`}>
                    {diff > 0 ? "\u25B2" : "\u25BC"} {diff > 0 ? "+" : ""}{((diff / boc1) * 100).toFixed(1)}%
                  </span>
                )}
              </div>
              <p className="text-[10px] mt-0.5">
                {diff !== null ? (
                  <><span className={diff > 0 ? "text-green-400" : "text-red-400"}>{diff > 0 ? "Bullish" : "Bearish"}</span><span className="text-zinc-600"> vs {boc1?.toFixed(2)}</span></>
                ) : <span className="text-zinc-600">{livePred?.model_name ?? "loading"}</span>}
              </p>
            </>
          )}
        </div>

        {/* Directional Scorecard */}
        <div className="glass-card p-4">
          <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">Direction Accuracy</p>
          {!loaded || !backtest ? <Skeleton className="h-8 w-24 mt-1.5" /> : (() => {
            const pts = backtest.points.slice(-20);
            let correct = 0;
            const dots: boolean[] = [];
            for (let i = 1; i < pts.length; i++) {
              const predDir = Math.sign(pts[i].predicted - pts[i - 1].actual);
              const actDir = Math.sign(pts[i].actual - pts[i - 1].actual);
              const ok = predDir === actDir;
              dots.push(ok);
              if (ok) correct++;
            }
            const pct = dots.length > 0 ? (correct / dots.length * 100) : 0;
            return (
              <>
                <p className="text-2xl font-bold mt-1 tabular-nums">{pct.toFixed(0)}%</p>
                <div className="flex gap-[2px] mt-1">
                  {dots.map((ok, i) => (
                    <div key={i} className={`w-[6px] h-[6px] rounded-full ${ok ? "bg-green-400" : "bg-red-400"}`} />
                  ))}
                </div>
                <p className="text-[10px] text-zinc-600 mt-0.5">Last {dots.length} predictions</p>
              </>
            );
          })()}
        </div>

        {/* Spread signal cards fill the second row */}
        {!loaded ? (
          <>
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </>
        ) : (
          spreads.map((s) => (
            <div key={s.name} className={`glass-card p-4 border-l-[3px] ${SIGNAL_BORDER[s.signal]}`}>
              <p className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium">{s.label}</p>
              <div className="flex items-baseline gap-2 mt-1">
                <p className="text-xl font-bold tabular-nums">
                  {typeof s.value === "number" && s.value > 10 ? s.value.toFixed(0) : s.value.toFixed(2)}
                </p>
                <span className="text-[10px] text-zinc-600">{s.unit}</span>
                <span className={`text-[10px] font-medium ${s.deviation_pct > 0 ? "text-green-400" : s.deviation_pct < 0 ? "text-red-400" : "text-zinc-400"}`}>
                  {TREND_ICON[s.trend]} {s.deviation_pct > 0 ? "+" : ""}{s.deviation_pct.toFixed(1)}%
                </span>
              </div>
              <p className="text-[10px] text-zinc-500 mt-1 line-clamp-2">{s.interpretation}</p>
            </div>
          ))
        )}
      </div>

      {/* Walk-forward backtest chart */}
      <div className="glass-card p-5">
        <div className="flex items-baseline justify-between mb-1">
          <h2 className="text-sm font-semibold">Actual vs Predicted</h2>
          {backtest && (
            <span className="text-[10px] text-zinc-600">
              {backtest.model}
            </span>
          )}
        </div>
        <p className="text-[11px] text-zinc-500 mb-4">
          {backtest
            ? `Walk-forward validation, ${backtest.n_points.toLocaleString()} out-of-sample predictions across ${backtest.n_folds} folds${
                backtest.points.length > 0
                  ? `, ${backtest.points[0].date} to ${backtest.points[backtest.points.length - 1].date}`
                  : ""
              }`
            : "Loading backtest data..."}
        </p>
        {!loaded ? (
          <Skeleton className="h-72 w-full" />
        ) : backtest && backtest.points.length > 0 ? (
          <>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={backtest.points}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                  <XAxis
                    dataKey="date"
                    stroke="#3f3f46"
                    fontSize={10}
                    interval={Math.ceil(backtest.points.length / 10)}
                    tickFormatter={(d: string) => {
                      if (!d || d.length < 7) return d;
                      const [y, m] = d.split("-");
                      const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
                      return `${months[parseInt(m, 10) - 1]} ${y}`;
                    }}
                  />
                  <YAxis
                    stroke="#3f3f46"
                    fontSize={10}
                    domain={["auto", "auto"]}
                    tickFormatter={(v: number) => v.toFixed(0)}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#111113",
                      border: "1px solid #1e1e22",
                      borderRadius: 8,
                      fontSize: 11,
                      lineHeight: 1.6,
                    }}
                    labelFormatter={(d: string) => d}
                    formatter={(v: number, name: string) => {
                      if (name === "upper" || name === "lower") return [null, null];
                      return [`${v?.toFixed(2)} c/lb`, name];
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 10, color: "#71717a" }} />
                  {/* 95% prediction interval band */}
                  <Area
                    type="monotone"
                    dataKey="upper"
                    stroke="none"
                    fill="#3b82f6"
                    fillOpacity={0.07}
                    legendType="none"
                    name="upper"
                  />
                  <Area
                    type="monotone"
                    dataKey="lower"
                    stroke="none"
                    fill="#09090b"
                    fillOpacity={1}
                    legendType="none"
                    name="lower"
                  />
                  {/* Fold boundary lines */}
                  {backtest.fold_boundaries?.slice(1).map((fb) => (
                    <ReferenceLine
                      key={fb.fold}
                      x={fb.start_date}
                      stroke="#3f3f46"
                      strokeDasharray="2 4"
                      label={{
                        value: `F${fb.fold}`,
                        position: "insideTopRight",
                        fill: "#52525b",
                        fontSize: 9,
                      }}
                    />
                  ))}
                  <Line type="monotone" dataKey="actual" stroke="#22c55e" strokeWidth={1.5} dot={false} name="Actual" />
                  <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={1.5} strokeDasharray="4 2" dot={false} name="Predicted" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-wrap gap-3 mt-3 pt-3 border-t border-[#1e1e22]">
              {[
                { label: "MAE", value: backtest.metrics.mae.toFixed(2), unit: "c/lb" },
                { label: "RMSE", value: backtest.metrics.rmse.toFixed(2), unit: "c/lb" },
                { label: "R²", value: backtest.metrics.r2.toFixed(4), unit: "" },
                { label: "Dir. Accuracy", value: `${(backtest.metrics.directional_accuracy * 100).toFixed(1)}%`, unit: "" },
                { label: "Folds", value: String(backtest.n_folds), unit: "" },
                { label: "OOS Predictions", value: backtest.n_points.toLocaleString(), unit: "" },
              ].map((m) => (
                <div key={m.label} className="bg-white/[0.03] rounded-md px-3 py-1.5">
                  <span className="text-[10px] text-zinc-500">{m.label}</span>
                  <span className="text-[13px] font-semibold tabular-nums ml-1.5">{m.value}</span>
                  {m.unit && <span className="text-[10px] text-zinc-600 ml-0.5">{m.unit}</span>}
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="text-sm text-zinc-600 text-center py-8">No backtest data available</p>
        )}
      </div>

      {/* Ticker strip */}
      <div className="glass-card px-4 py-3">
        {!loaded ? (
          <Skeleton className="h-5 w-full" />
        ) : (
          <>
            <div
              className="flex items-center gap-4 overflow-x-auto cursor-pointer"
              onClick={() => setShowPriceTable(!showPriceTable)}
            >
              {prices.map((p) => {
                const up = (p.change_pct ?? 0) >= 0;
                return (
                  <div
                    key={p.name}
                    className="flex items-center gap-1.5 shrink-0"
                  >
                    <span className={`text-[11px] font-medium ${p.name === "boc1" ? "text-green-400" : "text-zinc-400"}`}>
                      {LABELS[p.name]?.split(" ")[0] ?? p.name.toUpperCase()}
                    </span>
                    <span className="text-[12px] font-semibold tabular-nums">
                      {p.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                    <span className={`text-[10px] tabular-nums font-medium ${up ? "text-green-400" : "text-red-400"}`}>
                      {up ? "\u25B2" : "\u25BC"}
                      {Math.abs(p.change_pct ?? 0).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
              <span className="text-[9px] text-zinc-600 shrink-0 ml-auto">
                yfinance {prices[0] && `· ${new Date(prices[0].timestamp).toLocaleTimeString()}`}
              </span>
            </div>

            {showPriceTable && (
              <div className="mt-3 pt-3 border-t border-[#1e1e22]">
                <table className="w-full text-[12px]">
                  <thead>
                    <tr className="text-zinc-500">
                      <th className="text-left py-1 font-medium">Commodity</th>
                      <th className="text-right py-1 font-medium">Price</th>
                      <th className="text-right py-1 font-medium">Prev</th>
                      <th className="text-right py-1 font-medium">Chg</th>
                      <th className="text-right py-1 font-medium">% Chg</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prices.map((p) => {
                      const up = (p.change_pct ?? 0) >= 0;
                      return (
                        <tr key={p.name} className="border-t border-[#1e1e22]/50">
                          <td className="py-1.5 font-medium">{LABELS[p.name] ?? p.name}</td>
                          <td className="py-1.5 text-right tabular-nums">{p.price.toFixed(2)}</td>
                          <td className="py-1.5 text-right tabular-nums text-zinc-500">{(p.prev_close ?? p.price).toFixed(2)}</td>
                          <td className={`py-1.5 text-right tabular-nums ${up ? "text-green-400" : "text-red-400"}`}>
                            {up ? "+" : ""}{(p.change ?? 0).toFixed(2)}
                          </td>
                          <td className={`py-1.5 text-right tabular-nums font-medium ${up ? "text-green-400" : "text-red-400"}`}>
                            {up ? "+" : ""}{(p.change_pct ?? 0).toFixed(2)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>

      {/* Market Intelligence */}
      {loaded && livePred && boc1 != null && (() => {
        const d = livePred.predicted_price - boc1;
        const pct = (d / boc1) * 100;
        const modelBias = Math.abs(pct) < 2 ? "neutral" : pct > 0 ? "bullish" : "bearish";
        const farFromMean = Math.abs(boc1 - 41) / 41 > 0.5;

        // Determine overall signal by checking model + spread alignment
        const spreadBullish = spreads.filter((s) => s.signal === "bullish").length;
        const spreadBearish = spreads.filter((s) => s.signal === "bearish").length;
        const spreadsAgree = (modelBias === "bullish" && spreadBullish >= spreadBearish) ||
                             (modelBias === "bearish" && spreadBearish >= spreadBullish);

        let signalText: string;
        let signalColor: string;
        let signalDot: string;
        if (farFromMean) {
          signalText = "LOW CONFIDENCE — price outside training range";
          signalColor = "text-red-400";
          signalDot = "bg-red-400";
        } else if (modelBias === "neutral") {
          signalText = "NEUTRAL — model sees no clear directional edge";
          signalColor = "text-zinc-400";
          signalDot = "bg-zinc-400";
        } else if (spreadsAgree) {
          signalText = `${modelBias.toUpperCase()} — model and spreads aligned`;
          signalColor = modelBias === "bullish" ? "text-green-400" : "text-red-400";
          signalDot = modelBias === "bullish" ? "bg-green-400" : "bg-red-400";
        } else {
          signalText = `MIXED — model ${modelBias} but spreads diverge`;
          signalColor = "text-yellow-500";
          signalDot = "bg-yellow-500";
        }

        // Build spread summary line
        const crushSpread = spreads.find((s) => s.name === "crush_spread");
        const bopoSpread = spreads.find((s) => s.name === "bopo_spread");
        const scRatio = spreads.find((s) => s.name === "soy_corn_ratio");

        return (
          <div className="glass-card p-5">
            <div className="space-y-2">
              {/* Line 1: Signal */}
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${signalDot}`} />
                <p className={`text-[13px] font-semibold ${signalColor}`}>{signalText}</p>
              </div>

              {/* Line 2: Forecast */}
              <p className="text-[13px] text-zinc-400">
                Model predicts{" "}
                <strong className={d > 0 ? "text-green-400" : "text-red-400"}>
                  {d > 0 ? "upside" : "downside"} of {Math.abs(pct).toFixed(1)}%
                </strong>{" "}
                ({livePred.predicted_price.toFixed(2)} vs {boc1.toFixed(2)} c/lb).
                Next-day directional accuracy: 56%.
              </p>

              {/* Line 3: Spread context */}
              {(crushSpread || bopoSpread || scRatio) && (
                <p className="text-[13px] text-zinc-500">
                  {crushSpread && <>Crush: <strong className="text-zinc-300">${crushSpread.value}/bu</strong> ({crushSpread.deviation_pct > 0 ? "+" : ""}{crushSpread.deviation_pct}% vs avg). </>}
                  {bopoSpread && <>BOPO: <strong className="text-zinc-300">${bopoSpread.value}/mt</strong> ({(bopoSpread.value as number) > 200 ? "wide" : "narrow"}). </>}
                  {scRatio && <>Soy/Corn: <strong className="text-zinc-300">{scRatio.value}</strong>.</>}
                </p>
              )}

              {/* Line 4: Regime warning */}
              {farFromMean && (
                <p className="text-[12px] text-yellow-500/80">
                  Current BOC1 ({boc1.toFixed(0)} c/lb) is {((boc1 - 41) / 41 * 100).toFixed(0)}% above training mean. Prediction reliability is reduced in this regime.
                </p>
              )}
            </div>
          </div>
        );
      })()}

      {/* Model table */}
      {comp && comp.models.length > 0 && (
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold mb-3">Model Performance</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b border-[#1e1e22] text-zinc-500">
                  <th className="text-left py-2 px-3 font-medium">Model</th>
                  <th className="text-right py-2 px-3 font-medium">MAE</th>
                  <th className="text-right py-2 px-3 font-medium">RMSE</th>
                  <th className="text-right py-2 px-3 font-medium">R&sup2;</th>
                  <th className="text-right py-2 px-3 font-medium">Dir. Acc.</th>
                </tr>
              </thead>
              <tbody>
                {comp.models.map((m: any) => (
                  <tr
                    key={m.model ?? m.name}
                    className="border-b border-[#1e1e22]/50 hover:bg-white/[0.02]"
                  >
                    <td className="py-2 px-3 font-medium">
                      {m.model ?? m.name}
                      {(m.model ?? m.name) === comp.champion && (
                        <span className="ml-2 text-[10px] text-green-400 bg-green-500/10 px-1.5 py-0.5 rounded">
                          champion
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-right tabular-nums text-zinc-400">
                      {(m.mae ?? m.metrics?.mae)?.toFixed(2) ?? "-"}
                    </td>
                    <td className="py-2 px-3 text-right tabular-nums text-zinc-400">
                      {(m.rmse ?? m.metrics?.rmse)?.toFixed(2) ?? "-"}
                    </td>
                    <td className="py-2 px-3 text-right tabular-nums">
                      {(m.r2 ?? m.metrics?.r2)?.toFixed(4) ?? "-"}
                    </td>
                    <td className="py-2 px-3 text-right tabular-nums text-zinc-400">
                      {(m.directional_accuracy ?? m.metrics?.directional_accuracy)
                        ? `${((m.directional_accuracy ?? m.metrics?.directional_accuracy) * 100).toFixed(0)}%`
                        : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
