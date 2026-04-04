import type { Metadata } from "next";
import { Navbar } from "@/components/layout/navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Soybean Oil Predictor — BOC1 Futures Forecasting",
  description:
    "Domain-driven ML forecasting for front-month soybean oil futures (BOC1). " +
    "XGBoost baseline with walk-forward validation and SHAP explainability.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased">
        <Navbar />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>
        <footer className="border-t border-[#1e1e22] py-5 mt-16">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-zinc-600">
            <span>
              Built by Diego Rossi &middot; Market Risk &amp; Quantitative
              Analysis
            </span>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/rossi-diego/soybean-oil-predictor"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-zinc-400 transition-colors"
              >
                GitHub
              </a>
              <a
                href="https://soybean-oil-predictor-api.onrender.com/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-zinc-400 transition-colors"
              >
                API Docs
              </a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
