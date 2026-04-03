import type { Metadata } from "next";
import { Navbar } from "@/components/layout/navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Soybean Oil Predictor",
  description:
    "Commodity price forecasting for the front-month soybean oil contract (BOC1). " +
    "XGBoost + statsforecast models with SHAP explainability.",
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
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <footer className="border-t border-[#262626] py-6 mt-12">
          <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
            Soybean Oil Predictor v0.2.0 &middot; Built by Diego Rossi
            &middot; BOC1 forecasting with domain-driven ML
          </div>
        </footer>
      </body>
    </html>
  );
}
