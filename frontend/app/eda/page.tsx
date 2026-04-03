"use client";

import { useEffect, useState } from "react";
import { api, isDemoMode, FeatureImportance } from "@/lib/api";
import { DemoBanner } from "@/components/layout/demo-banner";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

export default function EDAPage() {
  const [stats, setStats] = useState<Record<string, Record<string, number>>>({});
  const [importances, setImportances] = useState<FeatureImportance[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    Promise.all([
      api.featureStats().then((data) => setStats(data.features)),
      api.featureImportance().then((data) => setImportances(data.importances)),
    ]).finally(() => setLoaded(true));
  }, []);

  const demo = loaded && isDemoMode();
  const featureNames = Object.keys(stats);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold gradient-text">
          Exploratory Data Analysis
        </h1>
        <p className="text-gray-400 mt-2">
          Feature statistics and importance analysis from the training dataset.
          Understanding feature distributions and correlations is critical for
          interpreting model behavior.
        </p>
      </div>

      {demo && <DemoBanner />}

      {importances.length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-xl font-semibold mb-4">Feature Importance</h2>
          <p className="text-sm text-gray-400 mb-4">
            How much each input feature contributes to the BOC1 price prediction.
            Higher importance means the model relies more on that feature.
          </p>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={importances}
                layout="vertical"
                margin={{ left: 80 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis type="number" stroke="#737373" fontSize={12} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  stroke="#737373"
                  fontSize={12}
                />
                <Tooltip
                  contentStyle={{
                    background: "#141414",
                    border: "1px solid #262626",
                    borderRadius: 8,
                  }}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {importances.map((_, idx) => (
                    <Cell
                      key={idx}
                      fill={idx === 0 ? "#22c55e" : idx < 3 ? "#4ade80" : "#737373"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {featureNames.length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-xl font-semibold mb-4">Feature Statistics</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#262626] text-gray-400">
                  <th className="text-left py-2 px-3">Feature</th>
                  <th className="text-right py-2 px-3">Mean</th>
                  <th className="text-right py-2 px-3">Std</th>
                  <th className="text-right py-2 px-3">Min</th>
                  <th className="text-right py-2 px-3">Max</th>
                </tr>
              </thead>
              <tbody>
                {featureNames.map((name) => (
                  <tr
                    key={name}
                    className="border-b border-[#262626]/50 hover:bg-white/5"
                  >
                    <td className="py-2 px-3 font-medium font-mono">{name}</td>
                    <td className="py-2 px-3 text-right text-gray-400">
                      {stats[name]?.mean?.toFixed(2) ?? "-"}
                    </td>
                    <td className="py-2 px-3 text-right text-gray-400">
                      {stats[name]?.std?.toFixed(2) ?? "-"}
                    </td>
                    <td className="py-2 px-3 text-right text-gray-400">
                      {stats[name]?.min?.toFixed(2) ?? "-"}
                    </td>
                    <td className="py-2 px-3 text-right text-gray-400">
                      {stats[name]?.max?.toFixed(2) ?? "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="glass-card p-6">
        <h2 className="text-xl font-semibold mb-4">Key Commodity Relationships</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-400">
          <div className="p-4 bg-black/20 rounded-lg">
            <h3 className="font-semibold text-gray-200 mb-2">
              Crush Spread
            </h3>
            <p>
              (11 x Oil + Meal) - Soybeans. Measures processing margin for
              crushers like Oleoplan, Bunge, Cargill. When positive, crushing
              is profitable.
            </p>
          </div>
          <div className="p-4 bg-black/20 rounded-lg">
            <h3 className="font-semibold text-gray-200 mb-2">
              BOC1 / CPO Spread
            </h3>
            <p>
              Soybean oil vs palm oil price gap. Drives substitution between
              the two largest vegetable oils. Widening = soy oil relatively
              expensive.
            </p>
          </div>
          <div className="p-4 bg-black/20 rounded-lg">
            <h3 className="font-semibold text-gray-200 mb-2">
              Soy / Corn Ratio
            </h3>
            <p>
              Influences farmer planting decisions. High ratio = more soybeans
              planted (bearish long-term). Typical range: 2.0 - 3.0.
            </p>
          </div>
          <div className="p-4 bg-black/20 rounded-lg">
            <h3 className="font-semibold text-gray-200 mb-2">Energy Link</h3>
            <p>
              Crude oil and heating oil correlate with soybean oil via the
              biodiesel pathway. Soybean oil competes as a biodiesel feedstock.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
