"""
DAG: train_ecobici_light
Variante para hosts con pocos recursos: entrena los mismos 5 modelos que
`train_ecobici_full`, con los mismos hiperparámetros, pero uno por vez
(batch) en lugar de en paralelo.

Motivo: con Docker Desktop en ~8 GB de RAM, si 2+ modelos entrenan al mismo
tiempo (Random Forest + XGBoost, por ejemplo), el SO mata los workers de
Celery por SIGKILL antes de que terminen (ver README, sección "Ajustes de
hiperparámetros para ejecución local con Docker"). Serializar el entrenamiento
evita el pico de memoria simultánea sin tener que bajar más los hiperparámetros.

Flujo (secuencial):
    validate_data
        → train_baseline_trivial
        → train_logistic_regression
        → train_random_forest
        → train_xgboost
        → train_catboost
        → select_champion → Model Registry MLflow (alias: champion)

`max_active_tasks=1` es un cinturón de seguridad adicional al encadenamiento
explícito con `>>`: garantiza que nunca haya dos tasks de esta corrida
ejecutando al mismo tiempo, sin depender de la concurrencia configurada en
el worker de Celery.

Prerequisito: correr primero el DAG `etl_ecobici`, que genera los splits en
s3://data/ecobici/processed/ a partir del raw.
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

import _ecobici_train_common as common


@dag(
    dag_id="train_ecobici_light",
    description="Entrena modelos EcoBici en secuencia (batch, uno por vez), loguea en MLflow y registra el champion — para hosts con pocos recursos",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_tasks=1,
    tags=["ecobici", "training", "mlflow", "light"],
)
def train_ecobici_light():
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

    # Ninguno de los 5 depende del output del otro (todos parten del mismo
    # `info`) — se encadenan a mano para forzar orden estrictamente secuencial.
    r_baseline >> r_lr >> r_rf >> r_xgb >> r_cat

    select_champion([r_baseline, r_lr, r_rf, r_xgb, r_cat])


train_ecobici_light()
