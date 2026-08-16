"""
API REST para predicción de duración de viaje en EcoBici CABA.

Endpoints:
  GET  /         → health check + estado del modelo
  POST /predict  → recibe las 19 features y devuelve Corto / Mediano / Largo

El modelo se carga al iniciar el servicio desde el MLflow Model Registry
(alias 'champion'). Si el DAG de entrenamiento todavía no corrió y el modelo
no está registrado, el endpoint /predict devuelve 503 hasta que esté disponible.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_URI           = "models:/ecobici_duracion_viaje@champion"

DESCRIPCIONES = {
    "Corto":   "Menos de 10 minutos",
    "Mediano": "Entre 10 y 30 minutos (tramo gratuito del sistema)",
    "Largo":   "Más de 30 minutos",
}

# Variable global — se carga una sola vez al arrancar el servicio
_model = None


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        logger.info("Cargando modelo champion desde MLflow Model Registry...")
        _model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info("Modelo cargado correctamente — %s", MODEL_URI)
    except Exception as exc:
        logger.warning(
            "No se pudo cargar el modelo: %s. "
            "El endpoint /predict no estará disponible hasta que el DAG train_ecobici corra.",
            exc,
        )
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EcoBici — Predicción de duración de viaje",
    description=(
        "Predice si un viaje en EcoBici CABA será **Corto** (< 10 min), "
        "**Mediano** (10–30 min) o **Largo** (> 30 min) "
        "a partir de la información disponible al momento de retirar la bicicleta."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ViajeInput(BaseModel):
    """19 features de entrada — idénticas a las usadas en el entrenamiento."""

    # Temporales cíclicas
    hora_sin: float
    hora_cos: float
    mes_sin:  float
    mes_cos:  float

    # Contexto del viaje
    es_fin_de_semana: int

    # Geográficas de la estación de origen
    lat_estacion_origen:  float
    long_estacion_origen: float

    # Equipamiento
    modelo_bicicleta: int

    # Perfil del usuario (OHE)
    genero_MALE:  int
    genero_OTHER: int

    # Día de la semana (OHE — referencia: viernes)
    dia_semana_Monday:    int
    dia_semana_Saturday:  int
    dia_semana_Sunday:    int
    dia_semana_Thursday:  int
    dia_semana_Tuesday:   int
    dia_semana_Wednesday: int

    # Turno del día (OHE — referencia: madrugada)
    turno_mañana: int
    turno_noche:  int
    turno_tarde:  int

    model_config = {"json_schema_extra": {
        "example": {
            "hora_sin": 0.5, "hora_cos": 0.866,
            "mes_sin": 0.5,  "mes_cos": 0.866,
            "es_fin_de_semana": 0,
            "lat_estacion_origen": -34.603, "long_estacion_origen": -58.381,
            "modelo_bicicleta": 1,
            "genero_MALE": 1, "genero_OTHER": 0,
            "dia_semana_Monday": 1, "dia_semana_Saturday": 0, "dia_semana_Sunday": 0,
            "dia_semana_Thursday": 0, "dia_semana_Tuesday": 0, "dia_semana_Wednesday": 0,
            "turno_mañana": 0, "turno_noche": 0, "turno_tarde": 1,
        }
    }}


class PrediccionOutput(BaseModel):
    tipo_viaje:  str
    descripcion: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    """Health check — indica si el modelo está cargado y listo para predecir."""
    return {
        "status":          "ok",
        "modelo_cargado":  _model is not None,
        "modelo_uri":      MODEL_URI if _model is not None else None,
    }


@app.post("/reload")
def reload_model():
    """
    Vuelve a resolver el alias `champion` en el MLflow Model Registry y
    recarga el modelo en memoria — sin reiniciar el contenedor.

    Llamar a este endpoint después de que el DAG `train_ecobici` registre una
    versión nueva del champion, para que la API empiece a servirla de
    inmediato (antes había que hacer `docker compose restart fastapi`).
    """
    global _model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        _model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info("Modelo recargado correctamente — %s", MODEL_URI)
    except Exception as exc:
        logger.warning("No se pudo recargar el modelo: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo recargar el modelo desde {MODEL_URI}: {exc}",
        )
    return {"status": "ok", "modelo_uri": MODEL_URI}


@app.post("/predict", response_model=PrediccionOutput)
def predict(viaje: ViajeInput):
    """
    Recibe las 19 features de un viaje EcoBici y devuelve la categoría de duración
    predicha por el modelo champion registrado en MLflow.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no disponible. "
                "Correr el DAG 'train_ecobici' en Airflow (http://localhost:8080) "
                "para entrenar y registrar el modelo champion."
            ),
        )

    X = pd.DataFrame([viaje.model_dump()])
    prediccion = str(_model.predict(X)[0])

    return PrediccionOutput(
        tipo_viaje=prediccion,
        descripcion=DESCRIPCIONES.get(prediccion, "—"),
    )
