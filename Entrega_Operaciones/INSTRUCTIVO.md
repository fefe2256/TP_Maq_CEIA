# Instructivo de uso — Stack MLOps EcoBici

Este documento describe paso a paso cómo levantar el ambiente, cargar los datos y ejecutar el pipeline de entrenamiento completo. Al final se detalla lo que queda pendiente para la entrega final.

---

## Prerequisitos

- Docker Desktop instalado y corriendo (mínimo 4 GB de RAM asignados)
- Los datasets en `EDA/dataset/` — ver cómo obtenerlos más abajo
- Un entorno Python local con `boto3` instalado (solo para correr la notebook de upload)

### Cómo obtener los datasets

Los CSV no están en el repositorio por su tamaño (~1.5 GB). Hay dos opciones:

**Opción A — Correr el notebook de EDA (genera todo automáticamente):**
```bash
pip install gdown folium jupyter
jupyter notebook EDA/notebook/TP_Grp7_V3_ecobici_presentation_ready.ipynb
```
La Sección 1 descarga el raw desde Google Drive y la Sección 12 exporta los 4 splits.

**Opción B — Descarga manual del raw + correr solo la exportación:**
```bash
pip install gdown
gdown 1t-QLtl__u1JCIXtjXEMUy_VfIttB0Kbp -O EDA/dataset/ecobici_data.csv
```
Luego correr el notebook de EDA para generar los splits procesados.

En ambos casos los archivos deben quedar en `EDA/dataset/` antes de continuar. Ver `EDA/dataset/README.md` para más detalle.

---

## Paso 1 — Levantar el stack Docker

```bash
cd Entrega_Operaciones/

# Solo la primera vez: copiar las variables de entorno
cp .env.example .env

# Construir imágenes y levantar todos los servicios
docker compose --profile all up -d --build
```

> La primera vez tarda varios minutos porque construye las imágenes custom de Airflow, MLflow y FastAPI, e instala las dependencias Python (scikit-learn, xgboost, catboost, mlflow, boto3, etc.).

Verificar que todos los servicios están healthy:

```bash
docker compose --profile all ps
```

Todos deben aparecer como `healthy` o `running`. Si alguno queda en `starting`, esperar unos minutos y volver a verificar.

### URLs de acceso

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | — |
| MinIO Console | http://localhost:9001 | minio / minio123 |
| FastAPI | http://localhost:8800 | — |

---

## Paso 2 — Subir el raw a MinIO y generar los splits

Con Docker corriendo, ejecutar la notebook `notebook_example/test.ipynb` desde tu entorno local. Sube el CSV crudo al bucket `s3://data`:

| Prefijo en MinIO | Archivos | Descripción |
|---|---|---|
| `ecobici/raw/` | `ecobici_data.csv` (765 MB) | Dataset crudo original |

Para verificar que el upload fue exitoso, ir a http://localhost:9001 → bucket `data` → carpeta `ecobici/raw/`.

Con el raw ya en MinIO, generá los splits procesados corriendo el **DAG de ETL** (Paso 2.1) en vez de subirlos a mano — así el ambiente arranca de cero sin depender de correr el notebook de EDA localmente.

### Paso 2.1 — Correr el DAG `etl_ecobici`

1. Ir a http://localhost:8080 (Airflow UI)
2. Buscar el DAG **`etl_ecobici`**, activarlo con el toggle y dispararlo (▶)
3. Corre 5 tasks en orden: `load_raw → clean → feature_engineer → split_and_encode → upload_splits`
4. Al terminar, deja los 4 CSV en `s3://data/ecobici/processed/` — con las coordenadas de estación **sin escalar** (el escalado ahora lo hace el Pipeline de cada modelo en `train_ecobici`, no un paso separado).

---

## Paso 3 — Ejecutar el DAG de entrenamiento

1. Ir a http://localhost:8080 (Airflow UI)
2. Buscar el DAG `train_ecobici`
3. Activarlo con el toggle (por defecto arranca pausado)
4. Hacer click en **"Trigger DAG"** (botón ▶)

### Qué hace el DAG

El DAG `train_ecobici` corre 7 tareas en total:

```
validate_data
    ↓
train_baseline_trivial ─┐
train_logistic_reg      ├── (paralelos)
train_random_forest     │
train_xgboost           │
train_catboost         ─┘
    ↓
select_champion
```

| Tarea | Modelo | Muestra de entrenamiento |
|---|---|---|
| `validate_data` | — | Verifica que los CSVs existen en MinIO |
| `train_baseline_trivial` | DummyClassifier (siempre Mediano) | Todo X_train |
| `train_logistic_regression` | LogisticRegression (multinomial) | 1.5 M filas |
| `train_random_forest` | RandomForestClassifier | 1 M filas |
| `train_xgboost` | XGBClassifier | 1.5 M filas |
| `train_catboost` | CatBoostClassifier | 1.5 M filas |
| `select_champion` | — | Compara por F1-macro y registra el mejor |

Los hiperparámetros de Random Forest y XGBoost son los mejores encontrados por Optuna en los notebooks locales (`TPMAQ2_VML_mlflow_parte1/parte2.ipynb`). KNN no se incluye por ser inviable a escala (>800 s de entrenamiento sobre 50k muestras).

### Duración estimada

El tiempo total depende de los recursos del contenedor. Estimado conservador:

| Tarea | Tiempo aproximado |
|---|---|
| validate_data | < 1 min |
| train_baseline_trivial | < 1 min |
| train_logistic_regression | 10–15 min |
| train_random_forest | 15–20 min |
| train_xgboost | 5–10 min |
| train_catboost | 5–10 min |
| select_champion | < 1 min |

Los modelos de entrenamiento corren en paralelo (hasta 2 simultáneos), así que el tiempo total es aproximadamente el de los dos modelos más lentos.

---

## Paso 4 — Verificar resultados en MLflow

1. Ir a http://localhost:5001
2. Seleccionar el experimento `TPMAQ2_Ecobici_Duracion_v2`
3. Ver los runs de cada modelo con sus métricas (F1-macro, F1 por clase)
4. Ir a **"Models"** → `ecobici_duracion_viaje` → verificar que existe una versión con el alias `champion`

---

## Paso 6 — Detener el stack

```bash
# Detener sin borrar datos (MinIO y PostgreSQL se preservan)
docker compose --profile all down

# Detener y borrar todo (datos incluidos)
docker compose --profile all down -v
```

---

## Paso 5 — Probar el endpoint de predicción

Con el DAG completado y el champion registrado en MLflow, FastAPI lo carga automáticamente al arrancar. Probar el endpoint:

```bash
curl -X POST http://localhost:8800/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hora_sin": 0.5,
    "hora_cos": 0.866,
    "mes_sin": 0.5,
    "mes_cos": 0.866,
    "es_fin_de_semana": 0,
    "lat_estacion_origen": -34.603,
    "long_estacion_origen": -58.381,
    "modelo_bicicleta": 1,
    "genero_MALE": 1,
    "genero_OTHER": 0,
    "dia_semana_Monday": 1,
    "dia_semana_Saturday": 0,
    "dia_semana_Sunday": 0,
    "dia_semana_Thursday": 0,
    "dia_semana_Tuesday": 0,
    "dia_semana_Wednesday": 0,
    "turno_mañana": 0,
    "turno_noche": 0,
    "turno_tarde": 1
  }'
```

Respuesta esperada:
```json
{
  "tipo_viaje": "Mediano",
  "descripcion": "Entre 10 y 30 minutos (tramo gratuito del sistema)"
}
```

La documentación interactiva con el formulario de prueba está en http://localhost:8800/docs.

> Si FastAPI devuelve `503`, significa que el DAG todavía no terminó o el modelo no quedó registrado. Verificar en MLflow (http://localhost:5001 → Models → `ecobici_duracion_viaje`) que exista una versión con el alias `champion`.

---

## Paso 5.1 — Recargar el champion sin reiniciar el contenedor

FastAPI carga el modelo `champion` **una sola vez**, al arrancar. Si volvés a correr `train_ecobici` más tarde (por ejemplo con datos nuevos) y se registra una versión distinta del champion, la API sigue sirviendo la versión vieja hasta que la avises.

En vez de reiniciar el contenedor (`docker compose restart fastapi`), llamá al endpoint `POST /reload`:

```bash
curl -X POST http://localhost:8800/reload
```

Respuesta esperada:
```json
{"status": "ok", "modelo_uri": "models:/ecobici_duracion_viaje@champion"}
```

Esto vuelve a resolver el alias `champion` contra el Model Registry de MLflow y recarga el modelo en memoria del proceso de FastAPI, sin downtime del contenedor.

---

## Resumen del estado actual

| Componente | Estado |
|---|---|
| Stack Docker (todos los servicios) | ✅ Listo |
| DAG `etl_ecobici` (raw → splits, sin scaler) | ✅ Listo |
| DAG `train_ecobici` (5 modelos con Pipeline scaler+modelo + champion) | ✅ Listo |
| FastAPI `POST /predict` | ✅ Listo |
| FastAPI `POST /reload` (recarga el champion sin reiniciar) | ✅ Listo |
| Prueba end-to-end | ⏳ Pendiente (requiere Docker corriendo) |
