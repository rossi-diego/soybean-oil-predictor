"""Airflow DAG: Daily futures price ingestion.

Fetches latest commodity futures prices from Yahoo Finance
and writes them to the Bronze layer (data/bronze/).
Runs every trading day at 18:00 UTC (after US market close).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "soybean-oil-predictor",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def run_futures_ingestion(**kwargs):
    """Execute incremental futures price ingestion."""
    from src.ingestion.yfinance_ingest import ingest_incremental
    from src.log import setup_logging

    setup_logging("INFO")
    result = ingest_incremental()
    return f"Ingested {len(result)} new rows"


with DAG(
    dag_id="ingest_futures_prices",
    default_args=default_args,
    description="Ingest daily commodity futures prices from Yahoo Finance",
    schedule="0 18 * * 1-5",  # Mon-Fri at 18:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "bronze", "futures"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_futures_prices",
        python_callable=run_futures_ingestion,
    )
