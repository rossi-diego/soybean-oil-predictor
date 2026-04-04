"use client";

import { useEffect, useState } from "react";
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
  Legend, Cell, ReferenceLine,
} from "recharts";

const COLORS = ["#22c55e", "#3b82f6", "#eab308", "#ef4444", "#8b5cf6", "#06b6d4", "#f97316"];

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
  const [corrMethod, setCorrMethod] = useState("pearson");
  const [selectedFeature, setSelectedFeature] = useState("boc1");

  useEffect(() => {
    Promise.all([
      api.edaPrices(period).then(setPrices).catch(() => {}),
      api.edaCorrelations(corrMethod).then(setCorr).catch(() => {}),
      api.edaDistributions().then(setDist).catch(() => {}),
      api.edaSpreads().then(setSpreads).catch(() => {}),
      api.edaSeasonality().then(setSeason).catch(() => {}),
      api.edaStationarity().then(setStation).catch(() => {}),
    ]).finally(() => setLoaded(true));
  }, []);

  useEffect(() => {
    api.edaPrices(period).then(setPrices).catch(() => {});
  }, [period]);

  useEffect(() => {
    api.edaCorrelations(corrMethod).then(setCorr).catch(() => {});
  }, [corrMethod]);

  const demo = loaded && isDemoMode();

  // Build chart data for price history
  const priceChartData = prices
    ? prices.dates.map((d, i) => {
        const row: Record<string, any> = { date: d };
        const src = normalized ? prices.normalized : prices.series;
        Object.keys(src).forEach((k) => { row[k] = src[k][i]; });
        return row;
      })
    : [];

  // Build spread chart data
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

      {/* 1. Price History */}
      <Section title="Price History" sub="Multi-commodity time series with optional normalization (base 100)">
        <div className="flex flex-wrap items-center gap-2 mb-3">
          {["1Y", "3Y", "5Y", "all"].map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${
                period === p ? "bg-white/10 text-white" : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {p === "all" ? "All" : p}
            </button>
          ))}
          <span className="text-zinc-700 mx-1">|</span>
          <button
            onClick={() => setNormalized(!normalized)}
            className={`text-[11px] px-2.5 py-1 rounded-md font-medium transition-colors ${
              normalized ? "bg-blue-600/20 text-blue-400" : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            Normalized
          </button>
        </div>
        {!prices ? <Skeleton className="h-64" /> : (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={priceChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                <XAxis dataKey="date" stroke="#3f3f46" fontSize={10} tickFormatter={(d: string) => d.slice(0, 7)} />
                <YAxis stroke="#3f3f46" fontSize={10} tickFormatter={(v: number) => normalized ? v.toFixed(0) : v.toFixed(1)} />
                <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                {Object.keys(prices.labels).map((k, i) => (
                  <Line key={k} type="monotone" dataKey={k} stroke={COLORS[i % COLORS.length]} strokeWidth={k === "boc1" ? 2 : 1} dot={false} name={prices.labels[k]} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      {/* 2. Correlation Matrix */}
      <Section title="Correlation Matrix">
        <div className="flex gap-2 mb-3">
          {["pearson", "spearman"].map((m) => (
            <button
              key={m}
              onClick={() => setCorrMethod(m)}
              className={`text-[11px] px-2.5 py-1 rounded-md font-medium capitalize transition-colors ${
                corrMethod === m ? "bg-white/10 text-white" : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
        {!corr ? <Skeleton className="h-64" /> : (
          <div className="overflow-x-auto">
            <table className="text-[11px]">
              <thead>
                <tr>
                  <th className="p-1" />
                  {corr.features.map((f) => (
                    <th key={f} className="p-1 text-zinc-500 font-medium text-center min-w-[56px]">
                      {corr.labels[f]?.slice(0, 8) ?? f}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {corr.features.map((row, i) => (
                  <tr key={row}>
                    <td className="p-1 text-zinc-500 font-medium pr-2 text-right whitespace-nowrap">
                      {corr.labels[row] ?? row}
                    </td>
                    {corr.matrix[i].map((val, j) => {
                      const abs = Math.abs(val);
                      const hue = val > 0 ? 142 : 0;
                      const sat = abs * 80;
                      const light = 15 + (1 - abs) * 10;
                      return (
                        <td
                          key={j}
                          className="p-1 text-center font-mono tabular-nums cursor-default"
                          style={{ background: `hsl(${hue}, ${sat}%, ${light}%)` }}
                          title={`${corr.labels[row]} vs ${corr.labels[corr.features[j]]}: ${val.toFixed(4)}`}
                        >
                          {val.toFixed(2)}
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

      {/* 3. Distribution Analysis */}
      <Section title="Distribution Analysis" sub="Training distribution vs recent 30-day window (drift indicator)">
        {!dist ? <Skeleton className="h-48" /> : (
          <>
            <div className="flex flex-wrap gap-1.5 mb-3">
              {Object.keys(dist.features).map((f) => (
                <button
                  key={f}
                  onClick={() => setSelectedFeature(f)}
                  className={`text-[10px] px-2 py-1 rounded font-medium transition-colors ${
                    selectedFeature === f ? "bg-white/10 text-white" : "text-zinc-600 hover:text-zinc-400"
                  }`}
                >
                  {dist.labels[f] ?? f}
                </button>
              ))}
            </div>
            {dist.features[selectedFeature] && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-2 h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={dist.features[selectedFeature].bins}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                      <XAxis dataKey="x" stroke="#3f3f46" fontSize={9} tickFormatter={(v: number) => v.toFixed(0)} />
                      <YAxis stroke="#3f3f46" fontSize={9} />
                      <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                      <Bar dataKey="count" fill="#3b82f6" opacity={0.6} name="Training" />
                      <Bar dataKey="recent_count" fill="#eab308" opacity={0.8} name="Recent 30d" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="text-[12px] space-y-1">
                  <h3 className="text-zinc-500 font-medium text-[11px] uppercase tracking-wider mb-2">
                    {dist.labels[selectedFeature]}
                  </h3>
                  {Object.entries(dist.features[selectedFeature].quartiles).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-zinc-500 capitalize">{k}</span>
                      <span className="tabular-nums">{v}</span>
                    </div>
                  ))}
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Mean</span>
                    <span className="tabular-nums">{dist.features[selectedFeature].mean}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Std</span>
                    <span className="tabular-nums">{dist.features[selectedFeature].std}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">N (train / recent)</span>
                    <span className="tabular-nums">
                      {dist.features[selectedFeature].n} / {dist.features[selectedFeature].recent_n}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </Section>

      {/* 4. Spread Analysis */}
      <Section title="Spread Analysis" sub="Historical spreads with 10th/90th percentile bands">
        {!spreads ? <Skeleton className="h-48" /> : (
          <div className="space-y-6">
            {spreadKeys.map((key) => {
              const s = spreads.spreads[key];
              const data = spreads.dates.map((d, i) => ({
                date: d,
                value: s.values[i],
                p10: s.p10,
                p90: s.p90,
              }));
              return (
                <div key={key}>
                  <div className="flex items-baseline gap-2 mb-2">
                    <h3 className="text-[12px] font-medium">{s.label}</h3>
                    <span className="text-[10px] text-zinc-600">
                      Avg: {s.mean} {s.unit} | P10: {s.p10} | P90: {s.p90}
                    </span>
                  </div>
                  <div className="h-40">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                        <XAxis dataKey="date" stroke="#3f3f46" fontSize={9} tickFormatter={(d: string) => d.slice(0, 7)} />
                        <YAxis stroke="#3f3f46" fontSize={9} />
                        <Tooltip contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }} />
                        <ReferenceLine y={s.p90} stroke="#22c55e" strokeDasharray="3 3" opacity={0.5} />
                        <ReferenceLine y={s.p10} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
                        <ReferenceLine y={s.mean} stroke="#3f3f46" strokeDasharray="2 2" />
                        <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={1.5} dot={false} name={s.label} />
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
      <Section title="Seasonality" sub="Monthly BOC1 price distribution — box plot by month">
        {!season ? <Skeleton className="h-48" /> : (
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={season.months}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e22" />
                <XAxis dataKey="name" stroke="#3f3f46" fontSize={10} />
                <YAxis stroke="#3f3f46" fontSize={10} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{ background: "#111113", border: "1px solid #1e1e22", borderRadius: 8, fontSize: 11 }}
                  formatter={(v: number, name: string) => [`${v.toFixed(2)} c/lb`, name]}
                />
                <Bar dataKey="q1" stackId="box" fill="transparent" />
                <Bar dataKey="median" stackId="box2" fill="transparent" />
                {season.months.map((m, i) => (
                  <ReferenceLine key={`med-${i}`} x={m.name} stroke="transparent" />
                ))}
                <Line type="monotone" dataKey="median" stroke="#22c55e" strokeWidth={2} dot={{ r: 3, fill: "#22c55e" }} name="Median" />
                <Line type="monotone" dataKey="mean" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 2" dot={false} name="Mean" />
                <Line type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={0.5} dot={false} name="Max" opacity={0.5} />
                <Line type="monotone" dataKey="min" stroke="#ef4444" strokeWidth={0.5} dot={false} name="Min" opacity={0.5} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}
      </Section>

      {/* 6. Stationarity & Returns */}
      <Section title="Stationarity &amp; Returns" sub="ADF tests, return distribution, and autocorrelation">
        {!station ? <Skeleton className="h-48" /> : (
          <div className="space-y-5">
            {/* Returns distribution */}
            <div>
              <h3 className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-2">
                BOC1 Daily Returns (%)
              </h3>
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
                </div>
              </div>
            </div>

            {/* ADF tests */}
            <div>
              <h3 className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-2">
                Augmented Dickey-Fuller Test
              </h3>
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
                      <tr key={feat} className="border-b border-[#1e1e22]/50 hover:bg-white/[0.02]">
                        <td className="py-1.5 px-2 font-medium">{station.labels[feat] ?? feat}</td>
                        <td className="py-1.5 px-2 text-right tabular-nums">{r.statistic?.toFixed(2) ?? "—"}</td>
                        <td className="py-1.5 px-2 text-right tabular-nums">{r.p_value?.toFixed(4) ?? "—"}</td>
                        <td className="py-1.5 px-2 text-center">
                          {r.stationary === true && <span className="text-green-400 text-[10px] font-medium">Yes</span>}
                          {r.stationary === false && <span className="text-red-400 text-[10px] font-medium">No</span>}
                          {r.stationary === null && <span className="text-zinc-600">—</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* ACF / PACF */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-2">ACF (BOC1)</h3>
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
                <h3 className="text-[11px] text-zinc-500 uppercase tracking-wider font-medium mb-2">PACF (BOC1)</h3>
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
