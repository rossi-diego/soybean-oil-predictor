/*
 * TODO: Monitoring page — re-enable when real data is available.
 *
 * Requirements before this page goes live:
 * 1. Evidently AI integration for data drift detection (feature distribution shifts)
 * 2. Scheduled retraining pipeline (Airflow DAG or GitHub Actions cron)
 * 3. API endpoint: GET /api/v1/monitoring/drift — returns drift metrics per feature
 * 4. API endpoint: GET /api/v1/monitoring/performance — historical model performance
 * 5. Historical model performance tracking (store predictions + actuals over time)
 * 6. Alert thresholds: MAE > 5.0, RMSE > 7.0, R² < 0.50 → trigger retraining
 *
 * Backend infrastructure exists in:
 * - src/monitoring/drift.py (Evidently report generation)
 * - src/monitoring/performance.py (model health checks)
 * - dags/dag_retrain_model.py (Airflow DAG for weekly retraining)
 *
 * This page should show real drift data, not placeholders.
 */

export default function MonitoringPage() {
  return (
    <div className="flex items-center justify-center min-h-[50vh]">
      <div className="text-center">
        <p className="text-zinc-500 text-sm">Monitoring — coming soon</p>
        <p className="text-zinc-600 text-xs mt-1">
          Drift detection and model performance tracking are being integrated.
        </p>
      </div>
    </div>
  );
}
