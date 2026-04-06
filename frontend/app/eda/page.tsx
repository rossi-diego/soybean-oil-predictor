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
  crush: { title: "Board Crush (CME)", desc: "Gross processing margin: (Meal \u00D7 0.022) + (Oil \u00D7 0.11) \u2212 Beans. Result in $/bu. Positive = profitable to crush. Above 30d avg = strong processor demand." },
  bopo: { title: "BOPO Spread", desc: "Soy oil premium to palm oil in USD/mt. Wide (>$200) = soy oil expensive, substitution risk. Narrow (<$100) = soy oil competitive." },
  soy_grain: { title: "Soy / Grain Ratio", desc: "Drives planting decisions. Above ~2.5 = farmers plant more soybeans (bearish supply). Below ~2.2 = more grain planted (bullish soy)." },
  soy_corn_ratio: { title: "Soy / Corn Ratio", desc: "US farmer planting economics. Above 2.5 = more soy planted. Below 2.2 = more corn. Range: 2.0\u20133.0." },
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
  const [selectedFeature, setSelectedFeature] = useState("boc1");

  useEffect(() => {
    Promise.all([
      api.edaCorrelations(corrMethod).then(setCorr).catch(() => {}),
      api.edaDistributions().then(setDist).catch(() => {}),
      api.edaRollingCorrelations().then(setRollingCorr).catch(() => {}),
      api.edaSpreads().then(setSpreads).catch(() => {}),
      api.edaSeasonality().then(setSeason).catch(() => {}),
      api.edaStationarity().then(setStation).catch(() => {}),
    ]).finally(() => setLoaded(true));
  }, []);

  useEffect(() => { api.edaPrices(period).then(setPrices).catch(() => {}); }, [period]);
  useEffect(() => { api.edaCorrelations(corrMethod).then(setCorr).catch(() => {}); }, [corrMethod]);

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
      <Section title="Price History" sub="Normalized view (base 100) compares commodities across different scales. Toggle Bollinger Bands (20d \u00B1 2\u03C3) and MAs in absolute view.">
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
      <Section title="Correlation Heatmap" sub={`Pearson: linear relationships. Spearman: monotonic (rank-based). If Spearman >> Pearson for a pair, the relationship is nonlinear.${corr ? ` Period: ${prices?.dates?.[0] ?? ""} to ${prices?.dates?.[prices.dates.length - 1] ?? ""}.` : ""}`}>
        <div className="flex gap-2 mb-3">
          {["pearson", "spearman"].map((m) => (
            <button key={m} onClick={() => setCorrMethod(m)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium capitalize transition-colors ${corrMethod === m ? "bg-white/10 text-white" : "text-zinc-500 hover:text-zinc-300"}`}>{m}</button>
          ))}
        </div>
        {!corr ? <Skeleton className="h-64" /> : (() => {
          const SHORT: Record<string, string> = {
            boc1: "Soy Oil", smc1: "Soy Meal", sc1: "Soybeans",
            lcoc1: "Brent", hoc1: "Heat Oil", fcpoc1: "Palm Oil", rsc1: "Wheat",
          };
          const shortLabel = (f: string) => SHORT[f] ?? corr.labels[f] ?? f;

          const cellColor = (v: number, diag: boolean): [string, string] => {
            if (diag) return ["#404040", "#71717a"];
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
                      const strong = Math.abs(val) > 0.7 && !isDiag;
                      return (
                        <td key={j}
                          className={`p-1 text-center font-mono tabular-nums rounded-sm ${strong ? "ring-1 ring-white/40" : ""}`}
                          style={{ background: bg, color: fg, minWidth: 48 }}
                          title={`${corr.labels[row]} vs ${corr.labels[corr.features[j]]}: r = ${val.toFixed(4)} (${corrMethod})`}
                        >{isDiag ? "\u2014" : val.toFixed(2)}</td>
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
      {rollingCorr && rollingCorr.dates.length > 0 && (
        <Section title="Rolling Correlation (60-day)" sub="How the correlation between Soy Oil and each commodity evolves over time. Relationships that were strong in 2020 may be weak in 2025. Post-2020, soy oil became more correlated with crude oil due to biodiesel demand expansion.">
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rollingCorr.dates.map((d, i) => {
                const row: Record<string, any> = { date: d };
                Object.keys(rollingCorr.series).forEach((k) => { row[k] = rollingCorr.series[k][i]; });
                return row;
              })}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                <XAxis dataKey="date" stroke="#3f3f46" fontSize={10} tickFormatter={(d: string) => d.slice(0, 7)} />
                <YAxis stroke="#3f3f46" fontSize={10} domain={[-0.2, 1]} tickFormatter={(v: number) => v.toFixed(1)} />
                <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} formatter={(v: number) => v?.toFixed(3)} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <ReferenceLine y={0} stroke="#3f3f46" />
                {Object.keys(rollingCorr.labels).map((k, i) => (
                  <Line key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} strokeWidth={1.2} dot={false} name={rollingCorr.labels[k]} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Section>
      )}

      {/* 3. Distribution Analysis */}
      <Section title="Distribution Analysis" sub="Distributions show whether current prices are typical or extreme. Values near the tails may reduce model accuracy. Blue = training data. Yellow = recent 30 days.">
        {!dist ? <Skeleton className="h-48" /> : (
          <>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {Object.keys(dist.features).map((f) => (
                <button key={f} onClick={() => setSelectedFeature(f)} className={`text-[10px] px-2 py-1 rounded font-medium transition-colors ${selectedFeature === f ? "bg-white/10 text-white" : "text-zinc-600 hover:text-zinc-400"}`}>{dist.labels[f] ?? f}</button>
              ))}
            </div>
            {dist.features[selectedFeature] && (() => {
              const feat = dist.features[selectedFeature];
              const skewLabel = feat.mean > feat.quartiles.median + 0.5 ? "Right-skewed" : feat.mean < feat.quartiles.median - 0.5 ? "Left-skewed" : "Approximately normal";
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
                        <Bar dataKey="recent_count" fill="#eab308" opacity={0.7} name="Recent 30d" />
                        <ReferenceLine x={feat.mean} stroke="#22c55e" strokeDasharray="3 3" />
                        <ReferenceLine x={feat.quartiles.median} stroke="#8b5cf6" strokeDasharray="3 3" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="text-[12px] space-y-1">
                    <p className="text-zinc-300 font-medium text-[11px] mb-2">{dist.labels[selectedFeature]} <span className="text-zinc-600">| {skewLabel}</span></p>
                    {Object.entries(feat.quartiles).map(([k, v]) => (
                      <div key={k} className="flex justify-between"><span className="text-zinc-500 capitalize">{k}</span><span className="tabular-nums">{v}</span></div>
                    ))}
                    <div className="flex justify-between"><span className="text-zinc-500">Mean</span><span className="tabular-nums text-green-400">{feat.mean}</span></div>
                    <div className="flex justify-between"><span className="text-zinc-500">Std</span><span className="tabular-nums">{feat.std}</span></div>
                    <div className="flex justify-between"><span className="text-zinc-500">N</span><span className="tabular-nums">{feat.n} / {feat.recent_n} recent</span></div>
                  </div>
                </div>
              );
            })()}
          </>
        )}
      </Section>

      {/* 4. Ratio Analysis */}
      <Section title="Ratio Analysis" sub="Key commodity ratios traders monitor. Shaded band = 10th\u201390th percentile range. Dashed line = historical average.">
        {!spreads ? <Skeleton className="h-48" /> : (
          <div className="space-y-6">
            {Object.keys(spreads.spreads).map((key) => {
              const s = spreads.spreads[key];
              const info = SPREAD_INFO[key];
              const current = s.values[s.values.length - 1];
              const range = s.p90 - s.p10;
              const pctl = current != null && range > 0 ? Math.round(((current - s.p10) / range) * 100) : null;
              const data = spreads.dates.map((d, i) => ({ date: d, value: s.values[i] }));
              return (
                <div key={key}>
                  <div className="flex items-baseline gap-2 mb-1">
                    <h3 className="text-[12px] font-medium">{info?.title ?? s.label}</h3>
                    <span className="text-[10px] text-zinc-500">
                      Current: <strong className="text-zinc-300">{current != null ? (typeof current === "number" && current > 10 ? current.toFixed(0) : current.toFixed(2)) : "N/A"} {s.unit}</strong>
                      {pctl != null && ` (${pctl}th percentile)`}
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
                {cur && ` Current month (${cur.name}): median ${cur.median.toFixed(1)} c/lb. Shaded band = Q1\u2013Q3 range.`}
              </p>
            </>
          );
        })()}
      </Section>

      {/* 6. Stationarity & Returns */}
      <Section title="Stationarity &amp; Returns" sub="ADF test for unit roots. ACF/PACF computed on daily returns (not levels). Fat tails indicate extreme moves are more frequent than a normal distribution predicts.">
        {!station ? <Skeleton className="h-48" /> : (
          <div className="space-y-5">
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
                  <div className="flex justify-between"><span className="text-zinc-500">Mean</span><span className="tabular-nums">{station.returns_stats.mean.toFixed(4)}%</span></div>
                  <div className="flex justify-between"><span className="text-zinc-500">Std</span><span className="tabular-nums">{station.returns_stats.std.toFixed(4)}%</span></div>
                  <div className="flex justify-between"><span className="text-zinc-500">Skewness</span><span className="tabular-nums">{station.returns_stats.skew.toFixed(4)}</span></div>
                  <div className="flex justify-between"><span className="text-zinc-500">Kurtosis</span><span className="tabular-nums">{station.returns_stats.kurtosis.toFixed(4)}</span></div>
                  <p className="text-[10px] text-zinc-500 pt-1 border-t border-[#1e1e22]">
                    {station.returns_stats.kurtosis > 0 ? "Fat tails \u2014 extreme daily moves more frequent than normal. Typical for commodity prices." : "Approximately normal distribution of returns."}
                  </p>
                </div>
              </div>
            </div>

            <div>
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
                        <td className="py-1.5 px-2 text-right tabular-nums">{r.statistic?.toFixed(2) ?? "\u2014"}</td>
                        <td className="py-1.5 px-2 text-right tabular-nums">{r.p_value != null ? r.p_value.toFixed(4) : "\u2014"}</td>
                        <td className="py-1.5 px-2 text-center">
                          {r.stationary === true && <span className="text-green-400 text-[10px]">Yes (p &lt; 0.05)</span>}
                          {r.stationary === false && <span className="text-red-400 text-[10px]">No (unit root)</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-zinc-600 mt-1">Non-stationary series require differencing for pure time-series models. Cross-sectional features (used here) are robust to non-stationarity.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[{ data: station.acf, title: "ACF (Returns)", color: "#3b82f6" }, { data: station.pacf, title: "PACF (Returns)", color: "#22c55e" }].map(({ data, title, color }) => (
                <div key={title}>
                  <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">{title}</h3>
                  <div className="h-36">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                        <XAxis dataKey="lag" stroke="#3f3f46" fontSize={9} />
                        <YAxis stroke="#3f3f46" fontSize={9} domain={[-0.15, 0.15]} />
                        <ReferenceLine y={station.confidence_interval} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
                        <ReferenceLine y={-station.confidence_interval} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
                        <Bar dataKey="value" fill={color} opacity={0.7} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </Section>
    </div>
  );
}
