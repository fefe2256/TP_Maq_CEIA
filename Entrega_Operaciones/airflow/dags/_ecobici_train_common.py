"""
Lógica compartida por las dos variantes del DAG de entrenamiento de EcoBici:

  - train_ecobici_full.py   → tasks en paralelo (5 branches). Para hosts con
                               muchos recursos (>= 16 GB RAM asignados a Docker).
  - train_ecobici_light.py  → mismas tasks, mismos hiperparámetros, pero
                               encadenadas en secuencia (una por vez / batch).
                               Para hosts con poca RAM, donde 2+ modelos
                               entrenando en simultáneo tumban los workers de
                               Celery con SIGKILL (ver README).

Este archivo NO define ningún DAG (`@dag`) — Airflow no lo lista en la UI,
solo lo importan los dos archivos de arriba. Está excluido del escaneo del
dag-processor vía `.airflowignore`.

Nota: `mlflow` se importa recién adentro de cada función que lo necesita
(no a nivel de módulo). Es una importación pesada — a nivel de módulo hace
que el dag-processor la pague en cada ciclo de parseo de ambos DAGs, lo cual
puede superar el `AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT` bajo carga.
"""

from __future__ import annotations

import io
import logging
import os

import boto3
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Configuración compartida ─────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DATA_BUCKET         = os.getenv("DATA_BUCKET", "data")
PROC_PREFIX         = "ecobici/processed"
EXPERIMENT_NAME     = "TPMAQ2_Ecobici_Duracion_v2"
REGISTRY_NAME       = os.getenv("MLFLOW_REGISTRY_NAME", "ecobici_duracion_viaje")
MODEL_ALIAS         = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
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

# Hiperparámetros — mismos valores para full y light (ver README, sección
# "Ajustes de hiperparámetros para ejecución local con Docker"). La diferencia
# entre las dos variantes es solo el paralelismo de tasks, no estos valores.
RF_PARAMS = {
    "n_estimators":      200,
    "max_depth":         15,
    "min_samples_leaf":  1,
    "min_samples_split": 2,
    "random_state":      RANDOM_STATE,
    "n_jobs":            1,
}

XGB_PARAMS = {
    "n_estimators":     198,
    "max_depth":        13,
    "learning_rate":    0.14105317872869383,
    "subsample":        0.604029269472136,
    "colsample_bytree": 0.6196644551821394,
    "gamma":            0.11785870241440835,
    "reg_alpha":        0.8303233813305007,
    "reg_lambda":       1.9626047840433123,
    "objective":        "multi:softmax",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "random_state":     RANDOM_STATE,
    "n_jobs":           1,
    "verbosity":        0,
}

CAT_PARAMS = {
    "iterations":         300,
    "learning_rate":      0.1,
    "depth":              8,
    "auto_class_weights": "Balanced",
    "random_seed":        RANDOM_STATE,
    "verbose":            0,
}

# Coordenadas de origen — únicas features que se escalan (notebook EDA, 3.9).
SCALED_COLS = ["lat_estacion_origen", "long_estacion_origen"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_s3():
    """Cliente boto3 — toma credenciales de las variables de entorno del contenedor."""
    return boto3.client("s3")


def load_data(bucket: str, prefix: str) -> tuple:
    """Descarga y devuelve (X_train, X_test, y_train, y_test) desde MinIO."""
    s3 = get_s3()

    def _read(key: str) -> pd.DataFrame:
        obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/{key}")
        return pd.read_csv(io.BytesIO(obj["Body"].read()), index_col=0)

    X_train = _read("X_train.csv")[FEATURES].fillna(0)
    X_test  = _read("X_test.csv")[FEATURES].fillna(0)
    y_train = _read("y_train.csv").squeeze()
    y_test  = _read("y_test.csv").squeeze()

    logger.info("Datos cargados — X_train: %s | X_test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def log_metrics(y_test: pd.Series, y_pred) -> float:
    """Loguea f1_macro, f1_weighted y f1 por clase en el run activo de MLflow."""
    import mlflow

    f1_macro     = f1_score(y_test, y_pred, average="macro")
    f1_weighted  = f1_score(y_test, y_pred, average="weighted")
    f1_por_clase = f1_score(y_test, y_pred, average=None, labels=CLASS_ORDER)

    metrics = {"f1_macro": f1_macro, "f1_weighted": f1_weighted}
    for clase, valor in zip(CLASS_ORDER, f1_por_clase):
        metrics[f"f1_{clase.lower()}"] = float(valor)

    mlflow.log_metrics(metrics)
    return f1_macro


def setup_mlflow() -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def make_pipeline(estimator) -> Pipeline:
    """Empaqueta el StandardScaler junto con el estimador en un único Pipeline
    de sklearn. MLflow loguea y sirve ambos como un solo objeto, así que
    entrenamiento e inferencia (FastAPI) aplican siempre exactamente la misma
    transformación sobre las coordenadas."""
    preprocessor = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), SCALED_COLS)],
        remainder="passthrough",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


# ── Tasks (funciones planas — cada DAG las envuelve con `task(...)`) ─────────

def validate_data() -> dict:
    """Verifica que los cuatro splits procesados existen en MinIO."""
    s3 = get_s3()
    archivos = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for archivo in archivos:
        key = f"{PROC_PREFIX}/{archivo}"
        try:
            s3.head_object(Bucket=DATA_BUCKET, Key=key)
            logger.info("OK: s3://%s/%s", DATA_BUCKET, key)
        except Exception:
            raise FileNotFoundError(f"No encontrado en MinIO: s3://{DATA_BUCKET}/{key}")
    return {"bucket": DATA_BUCKET, "prefix": PROC_PREFIX}


def train_baseline_trivial(info: dict) -> dict:
    """Baseline trivial: predice siempre la clase mayoritaria (Mediano)."""
    import mlflow
    import mlflow.sklearn

    X_train, X_test, y_train, y_test = load_data(info["bucket"], info["prefix"])
    setup_mlflow()

    with mlflow.start_run(run_name="Baseline_Trivial") as run:
        mlflow.set_tags({"modelo": "DummyClassifier", "categoria": "baseline"})
        mlflow.log_params({"strategy": "most_frequent"})

        model = make_pipeline(DummyClassifier(strategy="most_frequent"))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = log_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Baseline Trivial — F1-macro: %.4f", f1)

        return {"run_id": run.info.run_id, "artifact_path": "model"}


def train_logistic_regression(info: dict) -> dict:
    """Baseline ML: Regresión Logística Multinomial entrenada sobre 1.5M muestras."""
    import mlflow
    import mlflow.sklearn

    X_train, X_test, y_train, y_test = load_data(info["bucket"], info["prefix"])

    SAMPLE_SIZE = 1_500_000
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train,
        train_size=SAMPLE_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    setup_mlflow()
    params = {
        "solver":       "saga",
        "max_iter":     1000,
        "multi_class":  "multinomial",
        "C":            1.0,
        "random_state": RANDOM_STATE,
    }

    with mlflow.start_run(run_name="Baseline_LR") as run:
        mlflow.set_tags({"modelo": "LogisticRegression", "categoria": "baseline_ml"})
        mlflow.log_params({**params, "sample_size": SAMPLE_SIZE})

        model = make_pipeline(LogisticRegression(**params))
        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_test)

        f1 = log_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Logistic Regression — F1-macro: %.4f", f1)

        return {"run_id": run.info.run_id, "artifact_path": "model"}


def train_random_forest(info: dict) -> dict:
    """Random Forest con los mejores hiperparámetros encontrados por Optuna (parte1)."""
    import mlflow
    import mlflow.sklearn

    X_train, X_test, y_train, y_test = load_data(info["bucket"], info["prefix"])

    SAMPLE_TRAIN = 300_000
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train,
        train_size=SAMPLE_TRAIN,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    setup_mlflow()

    with mlflow.start_run(run_name="RandomForest_Optuna") as run:
        mlflow.set_tags({"modelo": "RandomForestClassifier", "categoria": "ensemble"})
        mlflow.log_params({**RF_PARAMS, "sample_size": SAMPLE_TRAIN})

        model = make_pipeline(RandomForestClassifier(**RF_PARAMS))
        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_test)

        f1 = log_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Random Forest — F1-macro: %.4f", f1)

        return {"run_id": run.info.run_id, "artifact_path": "model"}


def train_xgboost(info: dict) -> dict:
    """XGBoost con los mejores hiperparámetros encontrados por Optuna (parte2)."""
    import mlflow
    import mlflow.sklearn
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    X_train, X_test, y_train, y_test = load_data(info["bucket"], info["prefix"])

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
    y_sample   = le.transform(y_sample_str)
    y_test_enc = le.transform(y_test)

    setup_mlflow()
    params_a_logear = {k: v for k, v in XGB_PARAMS.items()
                        if k not in ("objective", "num_class", "eval_metric")}

    with mlflow.start_run(run_name="XGBoost_Optuna") as run:
        mlflow.set_tags({"modelo": "XGBClassifier", "categoria": "boosting"})
        mlflow.log_params({**params_a_logear, "sample_size": SAMPLE_TRAIN})

        model = make_pipeline(xgb.XGBClassifier(**XGB_PARAMS))
        model.fit(X_sample, y_sample)

        y_pred_enc = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)

        f1 = log_metrics(y_test, y_pred)
        # mlflow.sklearn (no mlflow.xgboost): el objeto logueado es un
        # Pipeline(scaler + XGBClassifier), no un XGBClassifier "pelado".
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("XGBoost — F1-macro: %.4f", f1)

        return {"run_id": run.info.run_id, "artifact_path": "model"}


def train_catboost(info: dict) -> dict:
    """CatBoost con parámetros default — resultados fuertes sin tuning (parte2)."""
    import mlflow
    import mlflow.sklearn
    from catboost import CatBoostClassifier

    X_train, X_test, y_train, y_test = load_data(info["bucket"], info["prefix"])

    SAMPLE_TRAIN = 1_500_000
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train,
        train_size=SAMPLE_TRAIN,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    setup_mlflow()

    with mlflow.start_run(run_name="CatBoost_Default") as run:
        mlflow.set_tags({"modelo": "CatBoostClassifier", "categoria": "boosting"})
        mlflow.log_params({**CAT_PARAMS, "sample_size": SAMPLE_TRAIN})

        model = make_pipeline(CatBoostClassifier(**CAT_PARAMS))
        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_test)

        f1 = log_metrics(y_test, y_pred)
        # mlflow.sklearn (no mlflow.catboost): el objeto logueado es un
        # Pipeline(scaler + CatBoostClassifier), no un CatBoostClassifier "pelado".
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("CatBoost — F1-macro: %.4f", f1)

        return {"run_id": run.info.run_id, "artifact_path": "model"}


def select_champion(resultados: list) -> str:
    """
    Compara el F1-macro de todos los runs, registra el mejor en el
    Model Registry y le asigna el alias `champion` (o el que indique
    MLFLOW_MODEL_ALIAS).
    """
    import mlflow
    from mlflow import MlflowClient

    setup_mlflow()
    client = MlflowClient()

    mejor_run_id   = None
    mejor_artifact = None
    mejor_f1       = -1.0

    logger.info("─── Comparación de modelos ───────────────────────────")
    for resultado in resultados:
        run    = mlflow.get_run(resultado["run_id"])
        f1     = run.data.metrics.get("f1_macro", 0.0)
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
    client.set_registered_model_alias(REGISTRY_NAME, MODEL_ALIAS, mv.version)

    logger.info("─────────────────────────────────────────────────────")
    logger.info("Champion: %s (F1-macro: %.4f) → versión %s", mejor_nombre, mejor_f1, mv.version)
    logger.info("URI: models:/%s@%s", REGISTRY_NAME, MODEL_ALIAS)

    return f"models:/{REGISTRY_NAME}@{MODEL_ALIAS}"
