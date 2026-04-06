"use client";

import { useEffect, useState, useMemo } from "react";
import {
  api,
  isDemoMode,
  EdaPricesResponse,
  EdaCorrelationsResponse,
  EdaDistributionsResponse,
  EdaSpreadsResponse,
  EdaSeasonalityResponse,
  EdaStationarityResponse,
} from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";
import {
  LineChart, Line, BarChart, Bar, ComposedChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, ReferenceLine,
} from "recharts";

const COLORS = ["#22c55e", "#3b82f6", "#eab308", "#ef4444", "#8b5cf6", "#06b6d4", "#f97316"];

function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse bg-[#1e1e22] rounded ${className}`} />;
}

function Section({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="glass-card p-5">
      <h2 className="text-sm font-semibold">{title}</h2>
      {sub && <p className="text-[11px] text-zinc-500 mt-0.5 mb-4 leading-relaxed">{sub}</p>}
      {!sub && <div className="mb-4" />}
      {children}
    </div>
  );
}

function rollingMean(arr: (number | null)[], w: number): (number | null)[] {
  return arr.map((_, i) => {
    if (i < w - 1) return null;
    let s = 0, c = 0;
    for (let j = i - w + 1; j <= i; j++) { if (arr[j] != null) { s += arr[j]!; c++; } }
    return c > 0 ? s / c : null;
  });
}

function rollingStd(arr: (number | null)[], w: number, means: (number | null)[]): (number | null)[] {
  return arr.map((_, i) => {
    if (i < w - 1 || means[i] == null) return null;
    let s = 0, c = 0;
    for (let j = i - w + 1; j <= i; j++) { if (arr[j] != null) { s += (arr[j]! - means[i]!) ** 2; c++; } }
    return c > 1 ? Math.sqrt(s / (c - 1)) : null;
  });
}

const SPREAD_INFO: Record<string, { title: string; desc: string }> = {
  crush: { title: "Board Crush (CME)", desc: "Gross processing margin: (Meal × 0.022) + (Oil × 0.11) − Beans. Result in $/bu. Positive = profitable to crush. Above 30d avg = strong processor demand." },
  bopo: { title: "BOPO Spread", desc: "Soy oil premium to palm oil in USD/mt. Wide (>$200) = soy oil expensive, substitution risk. Narrow (<$100) = soy oil competitive." },
  soy_grain: { title: "Soy / Wheat Ratio", desc: "Soybeans (c/bu) ÷ Wheat (c/bu). Measures relative value between the two crops. Different from the Soy/Corn ratio on the Dashboard which uses ZC=F." },
  soy_corn_ratio: { title: "Soy / Corn Ratio", desc: "US farmer planting economics. Above 2.5 = more soy planted. Below 2.2 = more corn. Range: 2.0–3.0." },
};

const VISIBLE_DEFAULT = new Set(["boc1", "sc1", "smc1", "lcoc1"]);

export default function EDAPage() {
  const [prices, setPrices] = useState<EdaPricesResponse | null>(null);
  const [corr, setCorr] = useState<EdaCorrelationsResponse | null>(null);
  const [dist, setDist] = useState<EdaDistributionsResponse | null>(null);
  const [spreads, setSpreads] = useState<EdaSpreadsResponse | null>(null);
  const [season, setSeason] = useState<EdaSeasonalityResponse | null>(null);
  const [station, setStation] = useState<EdaStationarityResponse | null>(null);
  const [rollingCorr, setRollingCorr] = useState<{ dates: string[]; series: Record<string, (number | null)[]>; labels: Record<string, string> } | null>(null);
  const [loaded, setLoaded] = useState(false);

  const [period, setPeriod] = useState("3Y");
  const [normalized, setNormalized] = useState(true);
  const [showBollinger, setShowBollinger] = useState(false);
  const [visibleSeries, setVisibleSeries] = useState<Set<string>>(VISIBLE_DEFAULT);
  const [corrMethod, setCorrMethod] = useState("pearson");
  const [corrPeriod, setCorrPeriod] = useState("1Y");
  const [rollPair, setRollPair] = useState("lcoc1");
  const [distWindow, setDistWindow] = useState("3M");
  const [selectedFeature, setSelectedFeature] = useState("boc1");

  useEffect(() => {
    Promise.all([
      api.edaCorrelations(corrMethod, corrPeriod).then(setCorr).catch(() => {}),
      api.edaDistributions(distWindow).then(setDist).catch(() => {}),
      api.edaRollingCorrelations().then(setRollingCorr).catch(() => {}),
      api.edaSpreads().then(setSpreads).catch(() => {}),
      api.edaSeasonality().then(setSeason).catch(() => {}),
      api.edaStationarity().then(setStation).catch(() => {}),
    ]).finally(() => setLoaded(true));
  }, []);

  useEffect(() => { api.edaPrices(period).then(setPrices).catch(() => {}); }, [period]);
  useEffect(() => { setCorr(null); api.edaCorrelations(corrMethod, corrPeriod).then(setCorr).catch(() => {}); }, [corrMethod, corrPeriod]);
  useEffect(() => { api.edaDistributions(distWindow).then(setDist).catch(() => {}); }, [distWindow]);

  const demo = loaded && isDemoMode();

  const priceChartData = useMemo(() => {
    if (!prices) return [];
    const boc1 = prices.series.boc1 ?? [];
    const ma20 = rollingMean(boc1, 20);
    const ma50 = rollingMean(boc1, 50);
    const bbMid = rollingMean(boc1, 20);
    const bbStd = rollingStd(boc1, 20, bbMid);
    const src = normalized ? prices.normalized : prices.series;

    return prices.dates.map((d, i) => {
      const row: Record<string, any> = { date: d };
      Object.keys(src).forEach((k) => { if (visibleSeries.has(k)) row[k] = src[k][i]; });
      if (!normalized) {
        row.ma20 = ma20[i];
        row.ma50 = ma50[i];
        if (showBollinger && bbMid[i] != null && bbStd[i] != null) {
          row.bbUpper = bbMid[i]! + 2 * bbStd[i]!;
          row.bbLower = bbMid[i]! - 2 * bbStd[i]!;
        }
      }
      return row;
    });
  }, [prices, normalized, showBollinger, visibleSeries]);

  const toggleSeries = (k: string) => {
    const next = new Set(visibleSeries);
    if (next.has(k)) next.delete(k); else next.add(k);
    setVisibleSeries(next);
  };

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-bold tracking-tight">Exploratory Data Analysis</h1>
        <p className="text-sm text-zinc-500 mt-1">
          {prices ? `${prices.n_points} observations, ${prices.dates[0]} to ${prices.dates[prices.dates.length - 1]}` : "Loading..."}
        </p>
      </section>

      {demo && <DemoBanner />}

      {/* 1. Price History */}
      <Section title="Price History" sub="Normalized view (base 100) compares commodities across different scales. Toggle Bollinger Bands (20d ± 2σ) and MAs in absolute view.">
        <div className="flex flex-wrap items-center gap-2 mb-2">
          {["1Y", "3Y", "5Y", "all"].map((p) => (
            <button key={p} onClick={() => setPeriod(p)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${period === p ? "bg-white/10 text-white" : "text-zinc-500 hover:text-zinc-300"}`}>{p === "all" ? "All" : p}</button>
          ))}
          <span className="text-zinc-700">|</span>
          <button onClick={() => setNormalized(!normalized)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${normalized ? "bg-blue-600/20 text-blue-400" : "text-zinc-500 hover:text-zinc-300"}`}>Normalized</button>
          {!normalized && <button onClick={() => setShowBollinger(!showBollinger)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${showBollinger ? "bg-purple-600/20 text-purple-400" : "text-zinc-500 hover:text-zinc-300"}`}>Bollinger</button>}
        </div>
        {/* Commodity toggles */}
        {prices && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {Object.keys(prices.labels).map((k, i) => (
              <button key={k} onClick={() => toggleSeries(k)} className={`text-[10px] px-2 py-0.5 rounded font-medium transition-colors ${visibleSeries.has(k) ? "text-white" : "text-zinc-600"}`} style={visibleSeries.has(k) ? { background: COLORS[i % COLORS.length] + "30", color: COLORS[i % COLORS.length] } : {}}>
                {prices.labels[k]}
              </button>
            ))}
          </div>
        )}
        {!prices ? <Skeleton className="h-72" /> : (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={priceChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                <XAxis dataKey="date" stroke="#3f3f46" fontSize={10} tickFormatter={(d: string) => d.slice(0, 7)} />
                <YAxis stroke="#3f3f46" fontSize={10} tickFormatter={(v: number) => normalized ? v.toFixed(0) : v.toFixed(1)} />
                <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                {!normalized && showBollinger && (
                  <>
                    <Area type="monotone" dataKey="bbUpper" stroke="none" fill="#8b5cf6" fillOpacity={0.06} legendType="none" name="BB" />
                    <Area type="monotone" dataKey="bbLower" stroke="none" fill="#09090b" fillOpacity={1} legendType="none" name="BBL" />
                  </>
                )}
                {prices && Object.keys(prices.labels).filter((k) => visibleSeries.has(k)).map((k, idx) => (
                  <Line key={k} type="monotone" dataKey={k} stroke={COLORS[Object.keys(prices.labels).indexOf(k) % COLORS.length]} strokeWidth={k === "boc1" ? 2 : 0.8} strokeOpacity={k === "boc1" ? 1 : 0.6} dot={false} name={prices.labels[k]} />
                ))}
                {!normalized && <>
                  <Line type="monotone" dataKey="ma20" stroke="#eab308" strokeWidth={1} strokeDasharray="4 2" dot={false} name="MA 20d" />
                  <Line type="monotone" dataKey="ma50" stroke="#ef4444" strokeWidth={1} strokeDasharray="6 3" dot={false} name="MA 50d" />
                </>}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      {/* 2. Correlation Heatmap */}
      <Section title="Correlation Heatmap" sub={`Pearson: linear relationships. Spearman: monotonic (rank-based).${corr?.n_days ? ` ${corr.start} to ${corr.end} (${corr.n_days} days).` : ""}`}>
        <div className="flex flex-wrap items-center gap-2 mb-3">
          {["pearson", "spearman"].map((m) => (
            <button key={m} onClick={() => setCorrMethod(m)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium capitalize transition-colors ${corrMethod === m ? "bg-white/10 text-white" : "text-zinc-500 hover:text-zinc-300"}`}>{m}</button>
          ))}
          <span className="text-zinc-700">|</span>
          {["1M", "3M", "6M", "1Y", "3Y"].map((p) => (
            <button key={p} onClick={() => setCorrPeriod(p)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${corrPeriod === p ? "bg-white/10 text-white" : "text-zinc-500 hover:text-zinc-300"}`}>{p}</button>
          ))}
        </div>
        {!corr ? <Skeleton className="h-64" /> : corr.error ? (
          <div className="flex items-center justify-center h-32 text-zinc-500 text-sm">{corr.error}</div>
        ) : !corr.matrix?.length ? (
          <div className="flex items-center justify-center h-32 text-zinc-500 text-sm">Not enough data for this period. Select a longer window.</div>
        ) : (() => {
          const SHORT: Record<string, string> = {
            boc1: "Soy Oil", smc1: "Soy Meal", sc1: "Soybeans",
            lcoc1: "Brent", hoc1: "Heat Oil", palm_oil: "Palm Oil", rsc1: "Wheat",
          };
          const shortLabel = (f: string) => SHORT[f] ?? corr.labels[f] ?? f;

          const cellColor = (v: number | null, diag: boolean): [string, string] => {
            if (diag) return ["#404040", "#71717a"];
            if (v == null || isNaN(v)) return ["#1e1e22", "#555"];
            if (v >= 0.8) return ["#67000D", "#fff"];
            if (v >= 0.6) return ["#CB181D", "#fff"];
            if (v >= 0.4) return ["#FB6A4A", "#fff"];
            if (v >= 0.2) return ["#FCBBA1", "#333"];
            if (v >= 0.0) return ["#2a2a2d", "#a1a1aa"];
            if (v >= -0.2) return ["#C6DBEF", "#333"];
            if (v >= -0.4) return ["#6BAED6", "#fff"];
            if (v >= -0.6) return ["#2171B5", "#fff"];
            return ["#08306B", "#fff"];
          };

          return (
          <div className="overflow-x-auto">
            <table className="text-[11px] border-separate" style={{ borderSpacing: 1 }}>
              <thead><tr><th className="p-1" />{corr.features.map((f) => (
                <th key={f} className="p-1 text-zinc-500 font-medium text-center min-w-[54px] -rotate-45 origin-bottom-left h-16 align-bottom">
                  <span className="inline-block">{shortLabel(f)}</span>
                </th>
              ))}</tr></thead>
              <tbody>
                {corr.features.map((row, i) => (
                  <tr key={row}>
                    <td className="p-1 text-zinc-500 font-medium pr-2 text-right whitespace-nowrap">{shortLabel(row)}</td>
                    {corr.matrix[i].map((val, j) => {
                      const isDiag = i === j;
                      const [bg, fg] = cellColor(val, isDiag);
                      const strong = val != null && Math.abs(val) > 0.7 && !isDiag;
                      return (
                        <td key={j}
                          className={`p-1 text-center font-mono tabular-nums rounded-sm ${strong ? "ring-1 ring-white/40" : ""}`}
                          style={{ background: bg, color: fg, minWidth: 48 }}
                          title={`${corr.labels[row]} vs ${corr.labels[corr.features[j]]}: r = ${val != null ? val.toFixed(4) : "N/A"} (${corrMethod})`}
                        >{isDiag ? "—" : val != null ? val.toFixed(2) : "—"}</td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          );
        })()}
      </Section>

      {/* 2b. Rolling Correlation */}
      {rollingCorr && rollingCorr.dates.length > 0 && (() => {
        const pairData = rollingCorr.series[rollPair] ?? [];
        const currentVal = pairData[pairData.length - 1];
        const pairLabel = rollingCorr.labels[rollPair] ?? rollPair;
        let interp = "";
        if (currentVal != null) {
          if (currentVal > 0.7) interp = "Strong positive — these commodities are moving together.";
          else if (currentVal > 0.3) interp = "Moderate positive — some co-movement.";
          else if (currentVal > -0.3) interp = "Weak — prices are moving independently.";
          else interp = "Inverse relationship — they tend to move in opposite directions.";
        }

        return (
        <Section title={`Soy Oil vs ${pairLabel} — 60-day Rolling Correlation`} sub="How the relationship evolves over time. Commodity correlations are not static — regime changes (biodiesel policy, supply shocks) shift them.">
          <div className="flex flex-wrap gap-1.5 mb-3">
            {Object.keys(rollingCorr.labels).map((k) => (
              <button key={k} onClick={() => setRollPair(k)} className={`text-[10px] px-2 py-1 rounded font-medium transition-colors ${rollPair === k ? "bg-white/10 text-white" : "text-zinc-600 hover:text-zinc-400"}`}>
                {rollingCorr.labels[k]}
              </button>
            ))}
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rollingCorr.dates.map((d, i) => ({ date: d, value: pairData[i] }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                <XAxis dataKey="date" stroke="#3f3f46" fontSize={10} tickFormatter={(d: string) => d.slice(0, 7)} />
                <YAxis stroke="#3f3f46" fontSize={10} domain={[-1, 1]} tickFormatter={(v: number) => v.toFixed(1)} />
                <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} formatter={(v: number) => [v?.toFixed(3), "Correlation"]} />
                <Area type="monotone" dataKey={() => 0.3} stroke="none" fill="#22c55e" fillOpacity={0.03} />
                <Area type="monotone" dataKey={() => -0.3} stroke="none" fill="#09090b" fillOpacity={1} />
                <ReferenceLine y={0} stroke="#3f3f46" strokeDasharray="4 2" />
                <ReferenceLine y={0.7} stroke="#22c55e" strokeDasharray="3 3" opacity={0.3} />
                <ReferenceLine y={-0.3} stroke="#ef4444" strokeDasharray="3 3" opacity={0.3} />
                <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={1.8} dot={false} name={pairLabel} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-baseline gap-3 mt-2">
            {currentVal != null && (
              <span className="text-[13px] font-semibold tabular-nums">Current: {currentVal.toFixed(3)}</span>
            )}
            <span className="text-[11px] text-zinc-500">{interp}</span>
          </div>
        </Section>
        );
      })()}

      {/* 3. Distribution Analysis */}
      <Section title="Distribution Analysis" sub={`Blue = training data (${dist?.training_start ?? ""} to ${dist?.training_end ?? ""}). Yellow = recent ${distWindow}. Orange line = current live price. Values outside the training range reduce prediction reliability.`}>
        {!dist ? <Skeleton className="h-48" /> : (
          <>
            <div className="flex flex-wrap items-center gap-2 mb-3">
              {Object.keys(dist.features).map((f) => (
                <button key={f} onClick={() => setSelectedFeature(f)} className={`text-[10px] px-2 py-1 rounded font-medium transition-colors ${selectedFeature === f ? "bg-white/10 text-white" : "text-zinc-600 hover:text-zinc-400"}`}>{dist.labels[f] ?? f}</button>
              ))}
              <span className="text-zinc-700">|</span>
              <span className="text-[10px] text-zinc-600">Recent:</span>
              {["1M", "3M", "6M", "1Y"].map((w) => (
                <button key={w} onClick={() => setDistWindow(w)} className={`text-[10px] px-2 py-0.5 rounded font-medium transition-colors ${distWindow === w ? "bg-yellow-600/20 text-yellow-400" : "text-zinc-600 hover:text-zinc-400"}`}>{w}</button>
              ))}
            </div>
            {dist.features[selectedFeature] && (() => {
              const feat = dist.features[selectedFeature];
              const skewLabel = feat.mean > feat.quartiles.median + 0.5 ? "Right-skewed" : feat.mean < feat.quartiles.median - 0.5 ? "Left-skewed" : "Approx. normal";
              const cur = feat.current;
              const pctl = feat.percentile;
              // Color code: green = Q1-Q3, amber = outside IQR but in range, red = outside training range
              const curColor = cur == null ? "text-zinc-400"
                : (cur < feat.quartiles.min || cur > feat.quartiles.max) ? "text-red-400"
                : (cur < feat.quartiles.q1 || cur > feat.quartiles.q3) ? "text-yellow-500"
                : "text-green-400";
              const curLabel = cur == null ? ""
                : (cur < feat.quartiles.min || cur > feat.quartiles.max) ? "Outside training range"
                : (cur < feat.quartiles.q1 || cur > feat.quartiles.q3) ? "Outside IQR"
                : "Normal range";
              return (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="md:col-span-2 h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={feat.bins}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                        <XAxis dataKey="x" stroke="#3f3f46" fontSize={9} tickFormatter={(v: number) => v.toFixed(0)} />
                        <YAxis stroke="#3f3f46" fontSize={9} />
                        <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                        <Bar dataKey="count" fill="#3b82f6" opacity={0.5} name="Training" />
                        <Bar dataKey="recent_count" fill="#eab308" opacity={0.7} name={`Recent ${distWindow}`} />
                        <ReferenceLine x={feat.mean} stroke="#22c55e" strokeDasharray="3 3" />
                        {cur != null && <ReferenceLine x={cur} stroke="#f97316" strokeWidth={2} label={{ value: `Now: ${cur}`, fill: "#f97316", fontSize: 9 }} />}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="text-[12px]">
                    {(() => {
                      const rq = feat.recent_quartiles;
                      const hasRecent = rq != null;
                      const driftColor = (train: number, recent: number) => {
                        if (feat.std === 0) return "";
                        return Math.abs(recent - train) > feat.std ? "text-yellow-500" : "";
                      };
                      const rows: { label: string; train: number; recent?: number }[] = [
                        { label: "Min", train: feat.quartiles.min, recent: rq?.min },
                        { label: "Q1", train: feat.quartiles.q1, recent: rq?.q1 },
                        { label: "Median", train: feat.quartiles.median, recent: rq?.median },
                        { label: "Q3", train: feat.quartiles.q3, recent: rq?.q3 },
                        { label: "Max", train: feat.quartiles.max, recent: rq?.max },
                        { label: "Mean", train: feat.mean, recent: feat.recent_mean ?? undefined },
                        { label: "Std", train: feat.std, recent: feat.recent_std ?? undefined },
                      ];
                      return (
                        <>
                          <p className="text-zinc-300 font-medium text-[11px] mb-2">
                            {dist.labels[selectedFeature]} <span className="text-zinc-600">| {skewLabel}</span>
                          </p>
                          {cur != null && (
                            <div className="flex justify-between mb-1">
                              <span className="text-zinc-500">Current</span>
                              <span className={`tabular-nums font-medium ${curColor}`}>{cur}{pctl != null && ` (${pctl}th pctl)`}</span>
                            </div>
                          )}
                          {curLabel && <p className={`text-[9px] ${curColor} mb-1`}>{curLabel}</p>}
                          <table className="w-full text-[11px]">
                            <thead>
                              <tr className="text-zinc-600">
                                <th className="text-left font-medium pb-1"></th>
                                <th className="text-right font-medium pb-1">Train</th>
                                {hasRecent && <th className="text-right font-medium pb-1 text-yellow-600">{distWindow}</th>}
                              </tr>
                            </thead>
                            <tbody>
                              {rows.map((r) => (
                                <tr key={r.label}>
                                  <td className="text-zinc-500 py-[1px]">{r.label}</td>
                                  <td className={`text-right tabular-nums py-[1px] ${r.label === "Mean" ? "text-green-400" : ""}`}>{r.train}</td>
                                  {hasRecent && (
                                    <td className={`text-right tabular-nums py-[1px] ${r.recent != null ? driftColor(r.train, r.recent) : ""}`}>
                                      {r.recent ?? "—"}
                                    </td>
                                  )}
                                </tr>
                              ))}
                              <tr>
                                <td className="text-zinc-500 py-[1px]">N</td>
                                <td className="text-right tabular-nums py-[1px]">{feat.n}</td>
                                {hasRecent && <td className="text-right tabular-nums py-[1px]">{feat.recent_n}</td>}
                              </tr>
                            </tbody>
                          </table>
                        </>
                      );
                    })()}
                  </div>
                </div>
              );
            })()}
          </>
        )}
      </Section>

      {/* 4. Ratio Analysis */}
      <Section title="Ratio Analysis" sub="Key commodity ratios traders monitor. Shaded band = 10th–90th percentile range. Dashed line = historical average.">
        {!spreads ? <Skeleton className="h-48" /> : (
          <div className="space-y-6">
            {Object.keys(spreads.spreads).map((key) => {
              const s = spreads.spreads[key];
              const info = SPREAD_INFO[key];
              const current = s.values.findLast((v: number | null) => v != null) ?? null;
              const range = s.p90 - s.p10;
              const rawPctl = current != null && range > 0 ? Math.round(((current - s.p10) / range) * 100) : null;
              const pctl = rawPctl != null ? Math.max(0, Math.min(rawPctl, 99)) : null;
              const pctlLabel = rawPctl != null && rawPctl > 99 ? ">99th" : rawPctl != null && rawPctl < 1 ? "<1st" : pctl != null ? `${pctl}th` : null;
              const data = spreads.dates.map((d, i) => ({ date: d, value: s.values[i] }));
              return (
                <div key={key}>
                  <div className="flex items-baseline gap-2 mb-1">
                    <h3 className="text-[12px] font-medium">{info?.title ?? s.label}</h3>
                    <span className="text-[10px] text-zinc-500">
                      Current: <strong className="text-zinc-300">{current != null ? (typeof current === "number" && current > 10 ? current.toFixed(0) : current.toFixed(2)) : "N/A"} {s.unit}</strong>
                      {pctlLabel != null && ` (${pctlLabel} percentile)`}
                    </span>
                  </div>
                  {info && <p className="text-[11px] text-zinc-500 mb-2">{info.desc}</p>}
                  <div className="h-36">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                        <XAxis dataKey="date" stroke="#3f3f46" fontSize={9} tickFormatter={(d: string) => d.slice(0, 7)} />
                        <YAxis stroke="#3f3f46" fontSize={9} />
                        <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                        <ReferenceLine y={s.p90} stroke="#22c55e" strokeDasharray="3 3" opacity={0.3} />
                        <ReferenceLine y={s.p10} stroke="#ef4444" strokeDasharray="3 3" opacity={0.3} />
                        <ReferenceLine y={s.mean} stroke="#3f3f46" strokeDasharray="2 2" />
                        <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={1.5} dot={false} name={info?.title ?? s.label} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Section>

      {/* 5. Seasonality */}
      <Section title="Seasonality" sub="Monthly BOC1 price distribution. Green = median, blue dashed = mean, red = extremes.">
        {!season ? <Skeleton className="h-52" /> : (() => {
          const medians = season.months.map((m) => m.median);
          const peak = season.months[medians.indexOf(Math.max(...medians))];
          const trough = season.months[medians.indexOf(Math.min(...medians))];
          const cur = season.months.find((m) => m.month === new Date().getMonth() + 1);
          return (
            <>
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={season.months}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                    <XAxis dataKey="name" stroke="#3f3f46" fontSize={10} />
                    <YAxis stroke="#3f3f46" fontSize={10} domain={["auto", "auto"]} />
                    <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} formatter={(v: number) => [`${v.toFixed(2)} c/lb`]} />
                    <Area type="monotone" dataKey="q3" stroke="none" fill="#3b82f6" fillOpacity={0.05} name="Q3" legendType="none" />
                    <Area type="monotone" dataKey="q1" stroke="none" fill="#09090b" fillOpacity={1} name="Q1" legendType="none" />
                    <Line type="monotone" dataKey="median" stroke="#22c55e" strokeWidth={2} dot={{ r: 3, fill: "#22c55e" }} name="Median" />
                    <Line type="monotone" dataKey="mean" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 2" dot={false} name="Mean" />
                    <Line type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={0.5} dot={false} name="Max" opacity={0.3} />
                    <Line type="monotone" dataKey="min" stroke="#ef4444" strokeWidth={0.5} dot={false} name="Min" opacity={0.3} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[11px] text-zinc-500 mt-2">
                Historically peaks in <strong className="text-zinc-300">{peak?.name}</strong> ({peak?.median.toFixed(1)} c/lb median) and troughs in <strong className="text-zinc-300">{trough?.name}</strong> ({trough?.median.toFixed(1)}).
                {cur && ` Current month (${cur.name}): median ${cur.median.toFixed(1)} c/lb. Shaded band = Q1–Q3 range.`}
              </p>
            </>
          );
        })()}
      </Section>

      {/* 6. Stationarity & Returns */}
      <Section title="Stationarity &amp; Returns" sub="Can we predict tomorrow&apos;s price from today&apos;s? These tests reveal the statistical structure of BOC1 price movements, which determines what types of models work and what signals traders can rely on.">
        {!station ? <Skeleton className="h-48" /> : (() => {
          const { returns_stats: rs, confidence_interval: ci } = station;
          const acfLag1 = station.acf.find((a) => a.lag === 1);
          const acfSignificant = acfLag1 && Math.abs(acfLag1.value) > ci;
          const stationaryCount = Object.values(station.adf_tests).filter((r) => r.stationary).length;
          const totalCount = Object.keys(station.adf_tests).length;

          return (
          <div className="space-y-6">
            {/* Returns histogram */}
            <div>
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">BOC1 Daily Returns (%)</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-2 h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={station.returns_hist}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                      <XAxis dataKey="x" stroke="#3f3f46" fontSize={9} tickFormatter={(v: number) => `${v.toFixed(1)}%`} />
                      <YAxis stroke="#3f3f46" fontSize={9} />
                      <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                      <ReferenceLine x={0} stroke="#3f3f46" />
                      <Bar dataKey="count" fill="#3b82f6" opacity={0.7} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="text-[12px] space-y-1">
                  <div className="flex justify-between"><span className="text-zinc-500">Mean</span><span className="tabular-nums">{rs.mean.toFixed(4)}%</span></div>
                  <div className="flex justify-between"><span className="text-zinc-500">Std</span><span className="tabular-nums">{rs.std.toFixed(4)}%</span></div>
                  <div className="flex justify-between"><span className="text-zinc-500">Skewness</span><span className="tabular-nums">{rs.skew.toFixed(2)}</span></div>
                  <div className="flex justify-between"><span className="text-zinc-500">Kurtosis</span><span className="tabular-nums">{rs.kurtosis.toFixed(2)}</span></div>
                </div>
              </div>
              <div className="mt-2 p-3 bg-white/[0.02] rounded-md text-[11px] text-zinc-400 space-y-1">
                <p>Average daily move is near zero ({rs.mean.toFixed(2)}%) &mdash; no persistent drift.</p>
                <p>Std of {rs.std.toFixed(2)}% means a typical day moves ±{(rs.std * 0.69).toFixed(1)} c/lb at current prices.</p>
                {rs.kurtosis > 3 && <p className="text-yellow-500/80">Kurtosis of {rs.kurtosis.toFixed(1)} (normal = 3) confirms fat tails &mdash; extreme moves (crashes, spikes) happen more often than a bell curve predicts. Standard VaR models underestimate tail risk.</p>}
              </div>
            </div>

            {/* ADF test */}
            <div>
              <div className="p-3 bg-blue-500/5 border border-blue-500/10 rounded-md mb-3 text-[12px] text-zinc-300">
                <strong>Bottom line:</strong> Soybean oil prices wander randomly (non-stationary). You cannot predict tomorrow&apos;s <em>price</em> from today&apos;s price alone. However, daily <em>returns</em> and <em>spreads</em> are predictable &mdash; which is why the model uses stationary features like crush z-score, momentum, and mean-reversion signals.
              </div>
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Augmented Dickey-Fuller Test</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-[12px]">
                  <thead><tr className="border-b border-[#1e1e22] text-zinc-500">
                    <th className="text-left py-1.5 px-2 font-medium">Feature</th>
                    <th className="text-right py-1.5 px-2 font-medium">ADF Stat</th>
                    <th className="text-right py-1.5 px-2 font-medium">p-value</th>
                    <th className="text-center py-1.5 px-2 font-medium">Stationary?</th>
                  </tr></thead>
                  <tbody>
                    {Object.entries(station.adf_tests).map(([feat, r]) => (
                      <tr key={feat} className="border-b border-[#1e1e22]/50">
                        <td className="py-1.5 px-2 font-medium">{station.labels[feat] ?? feat}</td>
                        <td className="py-1.5 px-2 text-right tabular-nums">{r.statistic?.toFixed(2) ?? "—"}</td>
                        <td className="py-1.5 px-2 text-right tabular-nums">{r.p_value != null ? r.p_value.toFixed(4) : "—"}</td>
                        <td className="py-1.5 px-2 text-center">
                          {r.stationary === true && <span className="text-green-400 text-[10px]">Stationary (p &lt; 0.05)</span>}
                          {r.stationary === false && <span className="text-red-400 text-[10px]">Unit root</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] text-zinc-500 mt-2">
                {stationaryCount} of {totalCount} commodities are stationary. A unit root means the price has no tendency to revert to an average &mdash; it can drift to any level. For trading: don&apos;t use mean-reversion on raw prices. Instead, trade <strong className="text-zinc-300">spreads</strong> (crush, BOPO) which are mean-reverting, or model <strong className="text-zinc-300">returns</strong> which are stationary.
              </p>
            </div>

            {/* ACF / PACF */}
            <div>
              <p className="text-[11px] text-zinc-400 mb-3">
                <strong className="text-zinc-300">Autocorrelation:</strong> Do yesterday&apos;s returns predict today&apos;s? Bars exceeding the red dashed lines (95% confidence bounds) are statistically significant. If most bars are near zero, returns are roughly independent day-to-day.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  { data: station.acf.filter((d) => d.lag > 0), title: "ACF (Returns)", color: "#3b82f6" },
                  { data: station.pacf.filter((d) => d.lag > 0), title: "PACF (Returns)", color: "#22c55e" },
                ].map(({ data, title, color }) => (
                  <div key={title}>
                    <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">{title}</h3>
                    <div className="h-36">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                          <XAxis dataKey="lag" stroke="#3f3f46" fontSize={9} />
                          <YAxis stroke="#3f3f46" fontSize={9} domain={[-0.12, 0.12]} />
                          <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} formatter={(v: number) => [v?.toFixed(4), "r"]} />
                          <ReferenceLine y={ci} stroke="#ef4444" strokeDasharray="3 3" opacity={0.6} label={{ value: "95% CI", fill: "#ef4444", fontSize: 8 }} />
                          <ReferenceLine y={-ci} stroke="#ef4444" strokeDasharray="3 3" opacity={0.6} />
                          <ReferenceLine y={0} stroke="#3f3f46" />
                          <Bar dataKey="value" fill={color} opacity={0.8} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-zinc-500 mt-2">
                {acfSignificant
                  ? `Lag 1 autocorrelation (${acfLag1?.value.toFixed(3)}) is significant — slight short-term momentum exists in daily returns. The model captures this signal via lagged return features.`
                  : "Returns show minimal autocorrelation — daily moves are largely independent. This limits the accuracy ceiling for any single-day prediction model, consistent with weak-form market efficiency."}
              </p>
            </div>
          </div>
          );
        })()}
      </Section>
    </div>
  );
}
