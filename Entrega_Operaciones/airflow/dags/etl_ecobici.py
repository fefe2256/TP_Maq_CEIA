"""
DAG: etl_ecobici
Genera los splits de entrenamiento (X_train, X_test, y_train, y_test) a partir
del CSV crudo de EcoBici, dentro del ambiente Docker — sin depender de correr
el notebook de EDA localmente ni de subir los CSV procesados a mano.

La limpieza y el feature engineering replican exactamente la lógica validada
en el notebook de EDA (EDA/notebook/TP_Grp7_V3_ecobici_presentation_ready.ipynb,
secciones 2 a 3.7 y 3.8 a 3.9): mismos umbrales de duración, mismas columnas,
mismo orden de encoding.

Diferencia intencional respecto al notebook: este DAG NO aplica StandardScaler
sobre las coordenadas de origen. Las deja crudas — el escalado se hace dentro
del Pipeline de cada modelo en el DAG `train_ecobici`, para que entrenamiento
e inferencia (FastAPI) usen siempre exactamente la misma transformación.

Flujo:
    load_raw → clean → feature_engineer → split_and_encode → upload_splits

Prerequisito: el CSV crudo (ecobici_data.csv) ya subido a s3://data/ecobici/raw/
(ver notebook_example/test.ipynb).

Los DataFrames intermedios entre tasks viajan como Parquet en un prefijo
temporal de MinIO (ecobici/tmp/), no por XCom: son varios millones de filas y
XCom no está pensado para payloads de ese tamaño.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
from airflow.decorators import dag, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Configuración global ─────────────────────────────────────────────────────
DATA_BUCKET  = "data"
RAW_PREFIX   = "ecobici/raw"
RAW_FILE     = "ecobici_data.csv"
TMP_PREFIX   = "ecobici/tmp"
PROC_PREFIX  = "ecobici/processed"

RANDOM_STATE = 42
TEST_SIZE    = 0.30

# Duración válida de un viaje, en segundos (notebook EDA, sección 2.2)
DURACION_MIN_SEG = 30           # menos de esto: click accidental
DURACION_MAX_SEG = 24 * 3600    # más de esto: bici no devuelta

# Discretización del target (notebook EDA, sección 3.4)
CORTES_DURACION  = [0, 10, 30, float("inf")]
ETIQUETAS_TARGET = ["Corto", "Mediano", "Largo"]

COLUMN_NAMES = [
    "id_recorrido", "duracion_recorrido", "fecha_origen_recorrido",
    "id_estacion_origen", "nombre_estacion_origen", "direccion_estacion_origen",
    "long_estacion_origen", "lat_estacion_origen", "fecha_destino_recorrido",
    "id_estacion_destino", "nombre_estacion_destino", "direccion_estacion_destino",
    "long_estacion_destino", "lat_estacion_destino", "id_usuario",
    "modelo_bicicleta", "genero",
]

FEATURES_NUMERICAS = [
    "hora_sin", "hora_cos", "mes_sin", "mes_cos", "es_fin_de_semana",
    "lat_estacion_origen", "long_estacion_origen",
]
FEATURES_CATEGORICAS = ["modelo_bicicleta", "genero", "dia_semana", "turno"]
TARGET = "tipo_viaje"


# ── Helpers de I/O contra MinIO ────────────────────────────────────────────

def _get_s3():
    return boto3.client("s3")


def _read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = _get_s3()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _write_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    s3 = _get_s3()
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def _write_csv_to_s3(df, bucket: str, key: str) -> None:
    s3 = _get_s3()
    buffer = io.StringIO()
    df.to_csv(buffer, index=True, header=True)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode("utf-8"))


def _franja_horaria(hora: int) -> str:
    """Idéntica a `franja_horaria` del notebook de EDA (sección 3.2)."""
    if 6 <= hora < 12:
        return "mañana"
    elif 12 <= hora < 18:
        return "tarde"
    elif 18 <= hora < 24:
        return "noche"
    return "madrugada"


# ── DAG ──────────────────────────────────────────────────────────────────────

@dag(
    dag_id="etl_ecobici",
    description="Genera los splits de entrenamiento de EcoBici desde el raw, dentro del ambiente",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ecobici", "etl"],
)
def etl_ecobici():

    @task
    def load_raw() -> str:
        """Descarga el CSV crudo desde MinIO y lo tipa mínimamente."""
        s3 = _get_s3()
        try:
            obj = s3.get_object(Bucket=DATA_BUCKET, Key=f"{RAW_PREFIX}/{RAW_FILE}")
        except Exception as exc:
            raise FileNotFoundError(
                f"No se encontró s3://{DATA_BUCKET}/{RAW_PREFIX}/{RAW_FILE}. "
                "Subilo primero con notebook_example/test.ipynb (Paso 2 del INSTRUCTIVO)."
            ) from exc

        df = pd.read_csv(
            io.BytesIO(obj["Body"].read()),
            skiprows=2,           # salta 'Tabla 1' y la fila de headers originales
            names=COLUMN_NAMES,
            on_bad_lines="skip",
            encoding="utf-8",
        )
        logger.info("Raw cargado — shape: %s", df.shape)

        key = f"{TMP_PREFIX}/01_raw.parquet"
        _write_parquet_to_s3(df, DATA_BUCKET, key)
        return key

    @task
    def clean(raw_key: str) -> str:
        """Tipado + filtro de duración por dominio + drop de nulos en genero.
        Notebook EDA, secciones 2.1, 2.2 y 2.4."""
        df = _read_parquet_from_s3(DATA_BUCKET, raw_key)

        for col in ["genero", "modelo_bicicleta"]:
            df[col] = df[col].astype(str).str.replace(";", "").str.strip()
            df[col] = df[col].replace("nan", np.nan)

        coord_cols = [
            "long_estacion_origen", "lat_estacion_origen",
            "long_estacion_destino", "lat_estacion_destino",
        ]
        for col in coord_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(";", ""), errors="coerce")

        df["fecha_origen_recorrido"] = pd.to_datetime(
            df["fecha_origen_recorrido"].astype(str).str.replace(";", ""), errors="coerce"
        )
        df["duracion_recorrido"] = pd.to_numeric(df["duracion_recorrido"], errors="coerce")

        n_total = len(df)
        df = df[df["duracion_recorrido"].between(DURACION_MIN_SEG, DURACION_MAX_SEG)].copy()
        df["duracion_minutos"] = df["duracion_recorrido"] / 60
        logger.info("Filtro de duración: %s → %s filas", n_total, len(df))

        n_antes = len(df)
        df = df.dropna(subset=["genero"]).copy()
        logger.info("Drop de nulos en genero: %s → %s filas", n_antes, len(df))

        key = f"{TMP_PREFIX}/02_clean.parquet"
        _write_parquet_to_s3(df, DATA_BUCKET, key)
        return key

    @task
    def feature_engineer(clean_key: str) -> str:
        """Encoding cíclico, es_fin_de_semana, turno y discretización del target.
        Notebook EDA, secciones 3.1 a 3.4."""
        df = _read_parquet_from_s3(DATA_BUCKET, clean_key)

        df["hora"]       = df["fecha_origen_recorrido"].dt.hour
        df["dia_semana"] = df["fecha_origen_recorrido"].dt.day_name()
        df["mes"]        = df["fecha_origen_recorrido"].dt.month

        df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
        df["mes_sin"]  = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"]  = np.cos(2 * np.pi * df["mes"] / 12)

        df["es_fin_de_semana"] = df["dia_semana"].isin(["Saturday", "Sunday"]).astype(int)
        df["turno"] = df["hora"].apply(_franja_horaria)

        df["tipo_viaje"] = pd.cut(
            df["duracion_minutos"], bins=CORTES_DURACION, labels=ETIQUETAS_TARGET
        )

        df_model = df[FEATURES_NUMERICAS + FEATURES_CATEGORICAS + [TARGET]].dropna().copy()
        logger.info("Filas listas para modelado: %s", len(df_model))

        key = f"{TMP_PREFIX}/03_features.parquet"
        _write_parquet_to_s3(df_model, DATA_BUCKET, key)
        return key

    @task
    def split_and_encode(features_key: str) -> dict:
        """Split estratificado + encoding de categóricas. Notebook EDA,
        secciones 3.6 a 3.8. NO escala coordenadas: eso lo hace el Pipeline
        de cada modelo en train_ecobici, para que no diverja de inferencia."""
        df_model = _read_parquet_from_s3(DATA_BUCKET, features_key)

        X = df_model.drop(columns=[TARGET])
        y = df_model[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        le = LabelEncoder()
        X_train["modelo_bicicleta"] = le.fit_transform(X_train["modelo_bicicleta"])
        X_test["modelo_bicicleta"]  = le.transform(X_test["modelo_bicicleta"])

        for col in ["genero", "dia_semana", "turno"]:
            X_train = pd.get_dummies(X_train, columns=[col], drop_first=True, dtype=int)
            X_test  = pd.get_dummies(X_test,  columns=[col], drop_first=True, dtype=int)
            X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

        X_train = X_train.fillna(0)
        X_test  = X_test.fillna(0)

        logger.info("X_train: %s | X_test: %s", X_train.shape, X_test.shape)

        keys = {
            "X_train": f"{TMP_PREFIX}/X_train.parquet",
            "X_test":  f"{TMP_PREFIX}/X_test.parquet",
            "y_train": f"{TMP_PREFIX}/y_train.parquet",
            "y_test":  f"{TMP_PREFIX}/y_test.parquet",
        }
        _write_parquet_to_s3(X_train, DATA_BUCKET, keys["X_train"])
        _write_parquet_to_s3(X_test,  DATA_BUCKET, keys["X_test"])
        _write_parquet_to_s3(y_train.to_frame(TARGET), DATA_BUCKET, keys["y_train"])
        _write_parquet_to_s3(y_test.to_frame(TARGET),  DATA_BUCKET, keys["y_test"])
        return keys

    @task
    def upload_splits(keys: dict) -> str:
        """Convierte los splits de Parquet temporal a CSV definitivo en
        ecobici/processed/ — el formato que ya espera train_ecobici."""
        for nombre, key in keys.items():
            df = _read_parquet_from_s3(DATA_BUCKET, key)
            _write_csv_to_s3(df, DATA_BUCKET, f"{PROC_PREFIX}/{nombre}.csv")
            logger.info("Subido: s3://%s/%s/%s.csv — %s", DATA_BUCKET, PROC_PREFIX, nombre, df.shape)

        return f"s3://{DATA_BUCKET}/{PROC_PREFIX}/"

    # ── Definición del pipeline ───────────────────────────────────────────────
    raw_key      = load_raw()
    clean_key    = clean(raw_key)
    features_key = feature_engineer(clean_key)
    split_keys   = split_and_encode(features_key)
    upload_splits(split_keys)


etl_ecobici()
