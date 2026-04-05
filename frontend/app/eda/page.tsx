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

function rollingMean(arr: (number | null)[], window: number): (number | null)[] {
  return arr.map((_, i) => {
    if (i < window - 1) return null;
    let sum = 0, count = 0;
    for (let j = i - window + 1; j <= i; j++) {
      if (arr[j] != null) { sum += arr[j]!; count++; }
    }
    return count > 0 ? sum / count : null;
  });
}

function rollingStd(arr: (number | null)[], window: number, means: (number | null)[]): (number | null)[] {
  return arr.map((_, i) => {
    if (i < window - 1 || means[i] == null) return null;
    let sum = 0, count = 0;
    for (let j = i - window + 1; j <= i; j++) {
      if (arr[j] != null && means[i] != null) { sum += (arr[j]! - means[i]!) ** 2; count++; }
    }
    return count > 1 ? Math.sqrt(sum / (count - 1)) : null;
  });
}

const SPREAD_INFO: Record<string, { title: string; desc: string }> = {
  crush: {
    title: "Crush Spread",
    desc: "Processor profitability: (11 \u00D7 Oil + Meal) \u2212 Soybeans. Above $0 = profitable to crush. Traders watch for margin compression as a signal of reduced demand.",
  },
  oil_palm: {
    title: "Oil / Palm Ratio",
    desc: "Relative value between soy oil and palm oil. High ratio = soy oil expensive vs palm \u2192 substitution pressure. Low ratio = demand support for soy oil.",
  },
  soy_wheat: {
    title: "Soy / Wheat Ratio",
    desc: "Influences planting decisions. High ratio (>2.5) = farmers plant more soy \u2192 bearish long-term supply. Typical range: 1.5\u20133.0.",
  },
};

export default function EDAPage() {
  const [prices, setPrices] = useState<EdaPricesResponse | null>(null);
  const [corr, setCorr] = useState<EdaCorrelationsResponse | null>(null);
  const [dist, setDist] = useState<EdaDistributionsResponse | null>(null);
  const [spreads, setSpreads] = useState<EdaSpreadsResponse | null>(null);
  const [season, setSeason] = useState<EdaSeasonalityResponse | null>(null);
  const [station, setStation] = useState<EdaStationarityResponse | null>(null);
  const [loaded, setLoaded] = useState(false);

  const [period, setPeriod] = useState("all");
  const [normalized, setNormalized] = useState(false);
  const [showBollinger, setShowBollinger] = useState(true);
  const [corrMethod, setCorrMethod] = useState("pearson");
  const [selectedFeature, setSelectedFeature] = useState("boc1");

  useEffect(() => {
    Promise.all([
      api.edaCorrelations(corrMethod).then(setCorr).catch(() => {}),
      api.edaDistributions().then(setDist).catch(() => {}),
      api.edaSpreads().then(setSpreads).catch(() => {}),
      api.edaSeasonality().then(setSeason).catch(() => {}),
      api.edaStationarity().then(setStation).catch(() => {}),
    ]).finally(() => setLoaded(true));
  }, []);

  useEffect(() => { api.edaPrices(period).then(setPrices).catch(() => {}); }, [period]);
  useEffect(() => { api.edaCorrelations(corrMethod).then(setCorr).catch(() => {}); }, [corrMethod]);

  const demo = loaded && isDemoMode();

  // Compute MAs and Bollinger Bands for BOC1
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
      Object.keys(src).forEach((k) => { row[k] = src[k][i]; });
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
  }, [prices, normalized, showBollinger]);

  const spreadKeys = spreads ? Object.keys(spreads.spreads) : [];

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-bold tracking-tight">Exploratory Data Analysis</h1>
        <p className="text-sm text-zinc-500 mt-1">
          {prices ? `${prices.n_points} observations` : "Loading"}
          {prices?.dates?.length ? `, ${prices.dates[0]} to ${prices.dates[prices.dates.length - 1]}` : ""}
        </p>
      </section>

      {demo && <DemoBanner />}

      {/* 1. Price History with MAs and Bollinger Bands */}
      <Section
        title="Price History"
        sub="Multi-commodity time series. BOC1 includes 20/50-day moving averages and Bollinger Bands (20d \u00B1 2\u03C3). Golden cross (20d > 50d) = bullish. Death cross (20d < 50d) = bearish."
      >
        <div className="flex flex-wrap items-center gap-2 mb-3">
          {["1Y", "3Y", "5Y", "all"].map((p) => (
            <button key={p} onClick={() => setPeriod(p)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${period === p ? "bg-white/10 text-white" : "text-zinc-500 hover:text-zinc-300"}`}>{p === "all" ? "All" : p}</button>
          ))}
          <span className="text-zinc-700 mx-1">|</span>
          <button onClick={() => setNormalized(!normalized)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${normalized ? "bg-blue-600/20 text-blue-400" : "text-zinc-500 hover:text-zinc-300"}`}>Normalized</button>
          {!normalized && (
            <button onClick={() => setShowBollinger(!showBollinger)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${showBollinger ? "bg-purple-600/20 text-purple-400" : "text-zinc-500 hover:text-zinc-300"}`}>Bollinger</button>
          )}
        </div>
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
                    <Area type="monotone" dataKey="bbUpper" stroke="none" fill="#8b5cf6" fillOpacity={0.06} legendType="none" name="BB Upper" />
                    <Area type="monotone" dataKey="bbLower" stroke="none" fill="#09090b" fillOpacity={1} legendType="none" name="BB Lower" />
                  </>
                )}
                {Object.keys(prices.labels).map((k, i) => (
                  <Line key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} strokeWidth={k === "boc1" ? 2 : 0.8} strokeOpacity={k === "boc1" ? 1 : 0.5} dot={false} name={prices.labels[k]} />
                ))}
                {!normalized && (
                  <>
                    <Line type="monotone" dataKey="ma20" stroke="#eab308" strokeWidth={1} strokeDasharray="4 2" dot={false} name="MA 20d" />
                    <Line type="monotone" dataKey="ma50" stroke="#ef4444" strokeWidth={1} strokeDasharray="6 3" dot={false} name="MA 50d" />
                  </>
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      {/* 2. Correlation Matrix */}
      <Section
        title="Correlation Heatmap"
        sub="Pearson measures linear relationships. Spearman measures monotonic (rank-based) relationships. If Spearman >> Pearson for a pair, the relationship is nonlinear."
      >
        <div className="flex gap-2 mb-3">
          {["pearson", "spearman"].map((m) => (
            <button key={m} onClick={() => setCorrMethod(m)} className={`text-[11px] px-2.5 py-1 rounded-md font-medium capitalize transition-colors ${corrMethod === m ? "bg-white/10 text-white" : "text-zinc-500 hover:text-zinc-300"}`}>{m}</button>
          ))}
        </div>
        {!corr ? <Skeleton className="h-64" /> : (
          <div className="overflow-x-auto">
            <table className="text-[11px]">
              <thead>
                <tr>
                  <th className="p-1" />
                  {corr.features.map((f) => (<th key={f} className="p-1 text-zinc-500 font-medium text-center min-w-[56px]">{corr.labels[f]?.slice(0, 8) ?? f}</th>))}
                </tr>
              </thead>
              <tbody>
                {corr.features.map((row, i) => (
                  <tr key={row}>
                    <td className="p-1 text-zinc-500 font-medium pr-2 text-right whitespace-nowrap">{corr.labels[row] ?? row}</td>
                    {corr.matrix[i].map((val, j) => {
                      const abs = Math.abs(val);
                      const hue = val > 0 ? 142 : 0;
                      const strong = abs > 0.7 && i !== j;
                      return (
                        <td key={j} className={`p-1 text-center font-mono tabular-nums ${strong ? "ring-1 ring-zinc-400" : ""}`}
                          style={{ background: `hsl(${hue}, ${abs * 80}%, ${15 + (1 - abs) * 10}%)` }}
                          title={`${corr.labels[row]} vs ${corr.labels[corr.features[j]]}: ${val.toFixed(4)}`}
                        >{val.toFixed(2)}</td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* 3. Distribution Analysis */}
      <Section title="Distribution Analysis" sub="Training distribution (blue) vs recent 30-day window (yellow). Divergence indicates potential data drift.">
        {!dist ? <Skeleton className="h-48" /> : (
          <>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {Object.keys(dist.features).map((f) => (
                <button key={f} onClick={() => setSelectedFeature(f)} className={`text-[10px] px-2 py-1 rounded font-medium transition-colors ${selectedFeature === f ? "bg-white/10 text-white" : "text-zinc-600 hover:text-zinc-400"}`}>{dist.labels[f] ?? f}</button>
              ))}
            </div>
            {dist.features[selectedFeature] && (() => {
              const feat = dist.features[selectedFeature];
              const skew = feat.mean > feat.quartiles.median ? "Positively skewed" : feat.mean < feat.quartiles.median ? "Negatively skewed" : "Approximately symmetric";
              return (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="md:col-span-2 h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={feat.bins}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                        <XAxis dataKey="x" stroke="#3f3f46" fontSize={9} tickFormatter={(v: number) => v.toFixed(0)} />
                        <YAxis stroke="#3f3f46" fontSize={9} />
                        <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                        <Bar dataKey="count" fill="#3b82f6" opacity={0.6} name="Training" />
                        <Bar dataKey="recent_count" fill="#eab308" opacity={0.8} name="Recent 30d" />
                        <ReferenceLine x={feat.mean} stroke="#22c55e" strokeDasharray="3 3" label={{ value: "Mean", fill: "#22c55e", fontSize: 9 }} />
                        <ReferenceLine x={feat.quartiles.median} stroke="#8b5cf6" strokeDasharray="3 3" label={{ value: "Median", fill: "#8b5cf6", fontSize: 9 }} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="text-[12px] space-y-1">
                    <h3 className="text-zinc-400 font-medium text-[11px] mb-2">{dist.labels[selectedFeature]}</h3>
                    {Object.entries(feat.quartiles).map(([k, v]) => (
                      <div key={k} className="flex justify-between"><span className="text-zinc-500 capitalize">{k}</span><span className="tabular-nums">{v}</span></div>
                    ))}
                    <div className="flex justify-between"><span className="text-zinc-500">Mean</span><span className="tabular-nums">{feat.mean}</span></div>
                    <div className="flex justify-between"><span className="text-zinc-500">Std</span><span className="tabular-nums">{feat.std}</span></div>
                    <div className="pt-1 mt-1 border-t border-[#1e1e22] text-[10px] text-zinc-500">{skew}</div>
                  </div>
                </div>
              );
            })()}
          </>
        )}
      </Section>

      {/* 4. Ratio Analysis */}
      <Section title="Ratio Analysis" sub="Key commodity ratios that traders monitor for relative value, processor economics, and planting signals.">
        {!spreads ? <Skeleton className="h-48" /> : (
          <div className="space-y-6">
            {spreadKeys.map((key) => {
              const s = spreads.spreads[key];
              const info = SPREAD_INFO[key];
              const data = spreads.dates.map((d, i) => ({ date: d, value: s.values[i], p10: s.p10, p90: s.p90 }));
              const current = s.values[s.values.length - 1];
              const range = s.p90 - s.p10;
              const percentile = current != null && range > 0 ? Math.round(((current - s.p10) / range) * 100) : null;
              return (
                <div key={key}>
                  <div className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-3 mb-1">
                    <h3 className="text-[12px] font-medium">{info?.title ?? s.label}</h3>
                    <span className="text-[10px] text-zinc-600">
                      Current: {current?.toFixed(2)} {s.unit}
                      {percentile != null && ` | ${percentile}th percentile`}
                      {` | Avg: ${s.mean} | P10: ${s.p10} | P90: ${s.p90}`}
                    </span>
                  </div>
                  {info && <p className="text-[11px] text-zinc-500 mb-2 leading-relaxed">{info.desc}</p>}
                  <div className="h-36">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                        <XAxis dataKey="date" stroke="#3f3f46" fontSize={9} tickFormatter={(d: string) => d.slice(0, 7)} />
                        <YAxis stroke="#3f3f46" fontSize={9} />
                        <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                        <ReferenceLine y={s.p90} stroke="#22c55e" strokeDasharray="3 3" opacity={0.4} />
                        <ReferenceLine y={s.p10} stroke="#ef4444" strokeDasharray="3 3" opacity={0.4} />
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
          const peakMonth = season.months[medians.indexOf(Math.max(...medians))];
          const troughMonth = season.months[medians.indexOf(Math.min(...medians))];
          const currentMonth = new Date().getMonth() + 1;
          const currentSeason = season.months.find((m) => m.month === currentMonth);
          return (
            <>
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={season.months}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                    <XAxis dataKey="name" stroke="#3f3f46" fontSize={10} />
                    <YAxis stroke="#3f3f46" fontSize={10} domain={["auto", "auto"]} />
                    <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} formatter={(v: number) => [`${v.toFixed(2)} c/lb`]} />
                    <Line type="monotone" dataKey="median" stroke="#22c55e" strokeWidth={2} dot={{ r: 3, fill: "#22c55e" }} name="Median" />
                    <Line type="monotone" dataKey="mean" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 2" dot={false} name="Mean" />
                    <Line type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={0.5} dot={false} name="Max" opacity={0.4} />
                    <Line type="monotone" dataKey="min" stroke="#ef4444" strokeWidth={0.5} dot={false} name="Min" opacity={0.4} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[11px] text-zinc-500 mt-2">
                BOC1 historically peaks in <strong className="text-zinc-300">{peakMonth?.name}</strong> (median {peakMonth?.median.toFixed(1)} c/lb)
                and troughs in <strong className="text-zinc-300">{troughMonth?.name}</strong> (median {troughMonth?.median.toFixed(1)} c/lb).
                {currentSeason && ` Current month (${currentSeason.name}): median ${currentSeason.median.toFixed(1)} c/lb.`}
              </p>
            </>
          );
        })()}
      </Section>

      {/* 6. Stationarity & Returns */}
      <Section title="Stationarity &amp; Returns" sub="ADF test for unit roots, daily returns distribution, and autocorrelation structure.">
        {!station ? <Skeleton className="h-48" /> : (
          <div className="space-y-5">
            {/* Returns distribution */}
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
                      <Bar dataKey="count" fill="#3b82f6" opacity={0.7} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="text-[12px] space-y-1">
                  <div className="flex justify-between"><span className="text-zinc-500">Mean</span><span className="tabular-nums">{station.returns_stats.mean.toFixed(4)}%</span></div>
                  <div className="flex justify-between"><span className="text-zinc-500">Std</span><span className="tabular-nums">{station.returns_stats.std.toFixed(4)}%</span></div>
                  <div className="flex justify-between"><span className="text-zinc-500">Skewness</span><span className="tabular-nums">{station.returns_stats.skew.toFixed(4)}</span></div>
                  <div className="flex justify-between"><span className="text-zinc-500">Kurtosis</span><span className="tabular-nums">{station.returns_stats.kurtosis.toFixed(4)}</span></div>
                  {station.returns_stats.kurtosis > 0 && (
                    <p className="text-[10px] text-zinc-500 pt-1 border-t border-[#1e1e22]">
                      Kurtosis &gt; 0 indicates fat tails — extreme daily moves are more frequent than a normal distribution predicts. This is typical for commodity prices.
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* ADF tests */}
            <div>
              <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Augmented Dickey-Fuller Test</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-[12px]">
                  <thead>
                    <tr className="border-b border-[#1e1e22] text-zinc-500">
                      <th className="text-left py-1.5 px-2 font-medium">Feature</th>
                      <th className="text-right py-1.5 px-2 font-medium">ADF Stat</th>
                      <th className="text-right py-1.5 px-2 font-medium">p-value</th>
                      <th className="text-center py-1.5 px-2 font-medium">Stationary?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(station.adf_tests).map(([feat, r]) => (
                      <tr key={feat} className="border-b border-[#1e1e22]/50">
                        <td className="py-1.5 px-2 font-medium">{station.labels[feat] ?? feat}</td>
                        <td className="py-1.5 px-2 text-right tabular-nums">{r.statistic?.toFixed(2) ?? "\u2014"}</td>
                        <td className="py-1.5 px-2 text-right tabular-nums">{r.p_value != null ? r.p_value.toFixed(4) : "\u2014"}</td>
                        <td className="py-1.5 px-2 text-center">
                          {r.stationary === true && <span className="text-green-400 text-[10px] font-medium">Yes (p &lt; 0.05)</span>}
                          {r.stationary === false && <span className="text-red-400 text-[10px] font-medium">No (unit root)</span>}
                          {r.stationary == null && <span className="text-zinc-600">\u2014</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-zinc-600 mt-1">Non-stationary series (p &gt; 0.05) should be differenced before modeling. The model uses level prices with cross-commodity features, which is standard for spread-based commodity forecasting.</p>
            </div>

            {/* ACF / PACF */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Autocorrelation (BOC1)</h3>
                <div className="h-36">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={station.acf}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                      <XAxis dataKey="lag" stroke="#3f3f46" fontSize={9} />
                      <YAxis stroke="#3f3f46" fontSize={9} domain={[-0.2, 1]} />
                      <ReferenceLine y={station.confidence_interval} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
                      <ReferenceLine y={-station.confidence_interval} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
                      <Bar dataKey="value" fill="#3b82f6" opacity={0.7} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <h3 className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium mb-2">Partial Autocorrelation (BOC1)</h3>
                <div className="h-36">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={station.pacf}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                      <XAxis dataKey="lag" stroke="#3f3f46" fontSize={9} />
                      <YAxis stroke="#3f3f46" fontSize={9} domain={[-0.2, 1]} />
                      <ReferenceLine y={station.confidence_interval} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
                      <ReferenceLine y={-station.confidence_interval} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
                      <Bar dataKey="value" fill="#22c55e" opacity={0.7} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}
      </Section>
    </div>
  );
}
