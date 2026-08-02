"""
DAG: train_ecobici
Entrena múltiples modelos sobre EcoBici dentro de Docker, loguea cada uno en
MLflow y registra automáticamente el mejor como 'champion' en el Model Registry.

Flujo:
    validate_data
        ↓
    train_baseline_trivial ─┐
    train_logistic_reg      ├─ (paralelos, hasta 2 simultáneos por el worker)
    train_random_forest     │
    train_xgboost           │
    train_catboost         ─┘
        ↓
    select_champion → Model Registry MLflow (alias: champion)

Notas:
  - Datos leídos desde MinIO: s3://data/ecobici/processed/
  - MLflow en red interna Docker: http://mlflow:5000
  - Los hiperparámetros de RF y XGBoost son los mejores encontrados por Optuna
    en los notebooks locales (TPMAQ2_VML_mlflow_parte1/parte2.ipynb).
  - KNN excluido: inviable a escala (>800 seg de entrenamiento sobre 50k muestras).
"""

from __future__ import annotations

import io
import logging
from datetime import datetime

import boto3
import mlflow
import mlflow.sklearn
import pandas as pd
from airflow.decorators import dag, task
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ── Configuración global ─────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://mlflow:5000"
DATA_BUCKET         = "data"
PROC_PREFIX         = "ecobici/processed"
EXPERIMENT_NAME     = "TPMAQ2_Ecobici_Duracion_v2"
REGISTRY_NAME       = "ecobici_duracion_viaje"
RANDOM_STATE        = 42
CLASS_ORDER         = ["Corto", "Mediano", "Largo"]

# 19 features — idénticas a las usadas en el notebook de EDA
FEATURES = [
    "hora_sin", "hora_cos", "mes_sin", "mes_cos", "es_fin_de_semana",
    "lat_estacion_origen", "long_estacion_origen", "modelo_bicicleta",
    "genero_MALE", "genero_OTHER",
    "dia_semana_Monday", "dia_semana_Saturday", "dia_semana_Sunday",
    "dia_semana_Thursday", "dia_semana_Tuesday", "dia_semana_Wednesday",
    "turno_mañana", "turno_noche", "turno_tarde",
]

# Mejores hiperparámetros de Random Forest encontrados por Optuna (notebook parte1)
RF_PARAMS = {
    "n_estimators":     200,
    "max_depth":        15,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "random_state":     RANDOM_STATE,
    "n_jobs":           1,
}

# Mejores hiperparámetros de XGBoost encontrados por Optuna (notebook parte2)
XGB_PARAMS = {
    "n_estimators":    198,
    "max_depth":       13,
    "learning_rate":   0.14105317872869383,
    "subsample":       0.604029269472136,
    "colsample_bytree": 0.6196644551821394,
    "gamma":           0.11785870241440835,
    "reg_alpha":       0.8303233813305007,
    "reg_lambda":      1.9626047840433123,
    "objective":       "multi:softmax",
    "num_class":       3,
    "eval_metric":     "mlogloss",
    "random_state":    RANDOM_STATE,
    "n_jobs":          1,
    "verbosity":       0,
}

# CatBoost con parámetros default — fuerte sin tuning (notebook parte2)
CAT_PARAMS = {
    "iterations":         300,
    "learning_rate":      0.1,
    "depth":              8,
    "auto_class_weights": "Balanced",
    "random_seed":        RANDOM_STATE,
    "verbose":            0,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_s3():
    """Cliente boto3 — toma credenciales de las variables de entorno del contenedor."""
    return boto3.client("s3")


def _load_data(bucket: str, prefix: str) -> tuple:
    """Descarga y devuelve (X_train, X_test, y_train, y_test) desde MinIO."""
    s3 = _get_s3()

    def _read(key: str) -> pd.DataFrame:
        obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/{key}")
        return pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=0)

    X_train = _read("X_train.csv")[FEATURES].fillna(0)
    X_test  = _read("X_test.csv")[FEATURES].fillna(0)
    y_train = _read("y_train.csv").squeeze()
    y_test  = _read("y_test.csv").squeeze()

    logger.info("Datos cargados — X_train: %s | X_test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def _log_metrics(y_test: pd.Series, y_pred) -> float:
    """Loguea f1_macro, f1_weighted y f1 por clase en el run activo de MLflow."""
    f1_macro    = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_por_clase = f1_score(y_test, y_pred, average=None, labels=CLASS_ORDER)

    metrics = {"f1_macro": f1_macro, "f1_weighted": f1_weighted}
    for clase, valor in zip(CLASS_ORDER, f1_por_clase):
        metrics[f"f1_{clase.lower()}"] = float(valor)

    mlflow.log_metrics(metrics)
    return f1_macro


def _setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ── DAG ──────────────────────────────────────────────────────────────────────

@dag(
    dag_id="train_ecobici",
    description="Entrena modelos EcoBici, loguea en MLflow y registra el champion",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ecobici", "training", "mlflow"],
)
def train_ecobici():

    @task
    def validate_data() -> dict:
        """Verifica que los cuatro splits procesados existen en MinIO."""
        s3 = _get_s3()
        archivos = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
        for archivo in archivos:
            key = f"{PROC_PREFIX}/{archivo}"
            try:
                s3.head_object(Bucket=DATA_BUCKET, Key=key)
                logger.info("OK: s3://%s/%s", DATA_BUCKET, key)
            except Exception:
                raise FileNotFoundError(f"No encontrado en MinIO: s3://{DATA_BUCKET}/{key}")
        return {"bucket": DATA_BUCKET, "prefix": PROC_PREFIX}

    # ── Modelos ───────────────────────────────────────────────────────────────

    @task
    def train_baseline_trivial(info: dict) -> dict:
        """Baseline trivial: predice siempre la clase mayoritaria (Mediano)."""
        X_train, X_test, y_train, y_test = _load_data(info["bucket"], info["prefix"])
        _setup_mlflow()

        with mlflow.start_run(run_name="Baseline_Trivial") as run:
            mlflow.set_tags({"modelo": "DummyClassifier", "categoria": "baseline"})
            mlflow.log_params({"strategy": "most_frequent"})

            model = DummyClassifier(strategy="most_frequent")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            f1 = _log_metrics(y_test, y_pred)
            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info("Baseline Trivial — F1-macro: %.4f", f1)

            return {"run_id": run.info.run_id, "artifact_path": "model"}

    @task
    def train_logistic_regression(info: dict) -> dict:
        """Baseline ML: Regresión Logística Multinomial entrenada sobre 1.5M muestras."""
        X_train, X_test, y_train, y_test = _load_data(info["bucket"], info["prefix"])

        SAMPLE_SIZE = 1_500_000
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=SAMPLE_SIZE,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

        _setup_mlflow()
        params = {
            "solver":      "saga",
            "max_iter":    1000,
            "multi_class": "multinomial",
            "C":           1.0,
            "random_state": RANDOM_STATE,
        }

        with mlflow.start_run(run_name="Baseline_LR") as run:
            mlflow.set_tags({"modelo": "LogisticRegression", "categoria": "baseline_ml"})
            mlflow.log_params({**params, "sample_size": SAMPLE_SIZE})

            model = LogisticRegression(**params)
            model.fit(X_sample, y_sample)
            y_pred = model.predict(X_test)

            f1 = _log_metrics(y_test, y_pred)
            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info("Logistic Regression — F1-macro: %.4f", f1)

            return {"run_id": run.info.run_id, "artifact_path": "model"}

    @task
    def train_random_forest(info: dict) -> dict:
        """Random Forest con los mejores hiperparámetros encontrados por Optuna (parte1)."""
        X_train, X_test, y_train, y_test = _load_data(info["bucket"], info["prefix"])

        SAMPLE_TRAIN = 300_000
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=SAMPLE_TRAIN,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

        _setup_mlflow()

        with mlflow.start_run(run_name="RandomForest_Optuna") as run:
            mlflow.set_tags({"modelo": "RandomForestClassifier", "categoria": "ensemble"})
            mlflow.log_params({**RF_PARAMS, "sample_size": SAMPLE_TRAIN})

            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(X_sample, y_sample)
            y_pred = model.predict(X_test)

            f1 = _log_metrics(y_test, y_pred)
            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info("Random Forest — F1-macro: %.4f", f1)

            return {"run_id": run.info.run_id, "artifact_path": "model"}

    @task
    def train_xgboost(info: dict) -> dict:
        """XGBoost con los mejores hiperparámetros encontrados por Optuna (parte2)."""
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder

        X_train, X_test, y_train, y_test = _load_data(info["bucket"], info["prefix"])

        SAMPLE_TRAIN = 500_000
        X_sample, _, y_sample_str, _ = train_test_split(
            X_train, y_train,
            train_size=SAMPLE_TRAIN,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

        # XGBoost requiere labels numéricas
        le = LabelEncoder()
        le.fit(CLASS_ORDER)
        y_sample  = le.transform(y_sample_str)
        y_test_enc = le.transform(y_test)

        _setup_mlflow()
        params_a_logear = {k: v for k, v in XGB_PARAMS.items()
                           if k not in ("objective", "num_class", "eval_metric")}

        with mlflow.start_run(run_name="XGBoost_Optuna") as run:
            mlflow.set_tags({"modelo": "XGBClassifier", "categoria": "boosting"})
            mlflow.log_params({**params_a_logear, "sample_size": SAMPLE_TRAIN})

            model = xgb.XGBClassifier(**XGB_PARAMS)
            model.fit(X_sample, y_sample)

            y_pred_enc = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)

            f1 = _log_metrics(y_test, y_pred)
            mlflow.xgboost.log_model(model, artifact_path="model")
            logger.info("XGBoost — F1-macro: %.4f", f1)

            return {"run_id": run.info.run_id, "artifact_path": "model"}

    @task
    def train_catboost(info: dict) -> dict:
        """CatBoost con parámetros default — resultados fuertes sin tuning (parte2)."""
        from catboost import CatBoostClassifier

        X_train, X_test, y_train, y_test = _load_data(info["bucket"], info["prefix"])

        SAMPLE_TRAIN = 1_500_000
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train,
            train_size=SAMPLE_TRAIN,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )

        _setup_mlflow()

        with mlflow.start_run(run_name="CatBoost_Default") as run:
            mlflow.set_tags({"modelo": "CatBoostClassifier", "categoria": "boosting"})
            mlflow.log_params({**CAT_PARAMS, "sample_size": SAMPLE_TRAIN})

            model = CatBoostClassifier(**CAT_PARAMS)
            model.fit(X_sample, y_sample)
            y_pred = model.predict(X_test)

            f1 = _log_metrics(y_test, y_pred)
            mlflow.catboost.log_model(model, artifact_path="model")
            logger.info("CatBoost — F1-macro: %.4f", f1)

            return {"run_id": run.info.run_id, "artifact_path": "model"}

    # ── Champion ──────────────────────────────────────────────────────────────

    @task
    def select_champion(resultados: list) -> str:
        """
        Compara el F1-macro de todos los runs, registra el mejor en el
        Model Registry y le asigna el alias 'champion'.
        """
        _setup_mlflow()
        client = MlflowClient()

        mejor_run_id    = None
        mejor_artifact  = None
        mejor_f1        = -1.0

        logger.info("─── Comparación de modelos ───────────────────────────")
        for resultado in resultados:
            run  = mlflow.get_run(resultado["run_id"])
            f1   = run.data.metrics.get("f1_macro", 0.0)
            nombre = run.data.tags.get("mlflow.runName", resultado["run_id"])
            logger.info("  %-30s F1-macro: %.4f", nombre, f1)

            if f1 > mejor_f1:
                mejor_f1       = f1
                mejor_run_id   = resultado["run_id"]
                mejor_artifact = resultado["artifact_path"]

        mejor_run    = mlflow.get_run(mejor_run_id)
        mejor_nombre = mejor_run.data.tags.get("mlflow.runName", "desconocido")
        model_uri    = f"runs:/{mejor_run_id}/{mejor_artifact}"

        # Crear el modelo registrado si todavía no existe
        try:
            client.create_registered_model(
                name=REGISTRY_NAME,
                description="Clasificador de duración de viaje EcoBici (Corto / Mediano / Largo).",
            )
        except Exception:
            pass  # Ya existía

        mv = client.create_model_version(
            name=REGISTRY_NAME,
            source=model_uri,
            run_id=mejor_run_id,
            description=f"Champion: {mejor_nombre} — F1-macro: {mejor_f1:.4f}",
        )
        client.set_registered_model_alias(REGISTRY_NAME, "champion", mv.version)

        logger.info("─────────────────────────────────────────────────────")
        logger.info("Champion: %s (F1-macro: %.4f) → versión %s", mejor_nombre, mejor_f1, mv.version)
        logger.info("URI: models:/%s@champion", REGISTRY_NAME)

        return f"models:/{REGISTRY_NAME}@champion"

    # ── Definición del pipeline ───────────────────────────────────────────────
    info = validate_data()

    r_baseline  = train_baseline_trivial(info)
    r_lr        = train_logistic_regression(info)
    r_rf        = train_random_forest(info)
    r_xgb       = train_xgboost(info)
    r_cat       = train_catboost(info)

    select_champion([r_baseline, r_lr, r_rf, r_xgb, r_cat])


train_ecobici()
