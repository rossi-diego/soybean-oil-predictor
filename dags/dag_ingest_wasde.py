"""Airflow DAG: Monthly WASDE report ingestion.

Fetches the latest WASDE report from USDA and writes to Bronze layer.
WASDE reports are published around the 10th of each month.
Runs on the 11th to ensure availability.
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
    "retry_delay": timedelta(minutes=10),
}


def run_wasde_ingestion(**kwargs):
    """Execute WASDE report ingestion."""
    from src.ingestion.usda_ingest import ingest_wasde
    from src.log import setup_logging

    setup_logging("INFO")
    result = ingest_wasde()
    return f"Ingested {len(result)} WASDE rows"


with DAG(
    dag_id="ingest_wasde_reports",
    default_args=default_args,
    description="Ingest monthly WASDE reports from USDA",
    schedule="0 12 11 * *",  # 11th of each month at 12:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "bronze", "wasde", "fundamentals"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_wasde_reports",
        python_callable=run_wasde_ingestion,
    )
