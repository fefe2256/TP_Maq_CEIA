"""
DAG: train_ecobici_full
Variante para hosts con muchos recursos (>= 16 GB RAM asignados a Docker):
entrena los 5 modelos en paralelo (hasta 2 simultáneos, según
AIRFLOW__CELERY__WORKER_CONCURRENCY), igual que la versión original del DAG.

Flujo:
    validate_data
        ↓
    train_baseline_trivial ─┐
    train_logistic_reg      ├─ (en paralelo)
    train_random_forest     │
    train_xgboost           │
    train_catboost         ─┘
        ↓
    select_champion → Model Registry MLflow (alias: champion)

Para equipos con menos RAM, ver `train_ecobici_light.py` — mismos modelos e
hiperparámetros, pero entrenados en secuencia (uno por vez) para evitar picos
de memoria. La lógica compartida entre ambas variantes vive en
`_ecobici_train_common.py`.

Prerequisito: correr primero el DAG `etl_ecobici`, que genera los splits en
s3://data/ecobici/processed/ a partir del raw.
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

import _ecobici_train_common as common


@dag(
    dag_id="train_ecobici_full",
    description="Entrena modelos EcoBici en paralelo, loguea en MLflow y registra el champion — para hosts con muchos recursos",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ecobici", "training", "mlflow", "full"],
)
def train_ecobici_full():
    validate_data             = task(common.validate_data)
    train_baseline_trivial    = task(common.train_baseline_trivial)
    train_logistic_regression = task(common.train_logistic_regression)
    train_random_forest       = task(common.train_random_forest)
    train_xgboost              = task(common.train_xgboost)
    train_catboost             = task(common.train_catboost)
    select_champion            = task(common.select_champion)

    info = validate_data()

    r_baseline = train_baseline_trivial(info)
    r_lr       = train_logistic_regression(info)
    r_rf       = train_random_forest(info)
    r_xgb      = train_xgboost(info)
    r_cat      = train_catboost(info)

    select_champion([r_baseline, r_lr, r_rf, r_xgb, r_cat])


train_ecobici_full()
