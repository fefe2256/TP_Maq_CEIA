# TP3 · Arquitectura ML en Producción — EcoBici CABA 2024

**Carrera de Especialización en Inteligencia Artificial · FIUBA · 1B 2026 · Grupo 7**

Implementación del modelo EcoBici (Random Forest) en un ambiente productivo containerizado con Docker Compose. El stack integra Apache Airflow para orquestación, MLflow para ciclo de vida de modelos, FastAPI para servir predicciones y MinIO como Data Lake S3 local.

---

## Arquitectura

```
                        ┌─────────────────────────────────────────┐
                        │              Docker Network              │
                        │                                          │
  Usuario / DAG ──────► │  Apache Airflow     ──►  MinIO (S3)     │
                        │  (CeleryExecutor)        s3://data       │
                        │        │                 s3://mlflow     │
                        │        ▼                      ▲          │
                        │    Entrenar RF    ──────────── │          │
                        │        │                       │          │
                        │        ▼                       │          │
                        │     MLflow        ─────────────┘          │
                        │  (Model Registry)                         │
                        │        │                                  │
                        │        ▼                                  │
  Cliente HTTP ───────► │    FastAPI                                │
                        │  (POST /predict)                          │
                        │                                          │
                        │  PostgreSQL  ◄──── Airflow + MLflow      │
                        │  Valkey/Redis ◄─── Celery broker         │
                        └─────────────────────────────────────────┘
```

### Servicios

| Servicio | Rol | Puerto |
|---|---|---|
| Apache Airflow | Orquestación de DAGs (ETL, entrenamiento) | 8080 |
| MLflow | Tracking de experimentos + Model Registry | 5001 |
| FastAPI | API REST para servir predicciones | 8800 |
| MinIO | Data Lake S3 local (datasets y artefactos ML) | 9000 / 9001 |
| PostgreSQL | Backend de Airflow y MLflow | 5432 |
| Valkey | Broker de Celery para Airflow (fork open-source de Redis) | 6379 |

### Buckets MinIO

| Bucket | Contenido |
|---|---|
| `s3://mlflow` | Artefactos de MLflow (modelos, confusion matrix, plots) |
| `s3://data` | Datasets procesados para entrenamiento e inferencia |

---

## Estructura de la carpeta

```
Entrega_Operaciones/
├── docker-compose.yaml          ← orquestación de todos los servicios
├── .env.example                 ← plantilla de variables de entorno
│
├── airflow/
│   ├── dags/
│   │   └── train_ecobici.py     ← DAG: 5 modelos + selección automática de champion
│   ├── logs/                    ← logs de ejecución (generado al levantar)
│   ├── plugins/                 ← plugins custom de Airflow
│   ├── config/                  ← configuración de Airflow
│   └── secrets/
│       ├── variables.yaml       ← variables globales accesibles desde los DAGs
│       └── connections.yaml     ← conexiones registradas en Airflow
│
├── notebook_example/
│   └── test.ipynb               ← upload de datasets a MinIO (ejecutar antes del DAG)
│
└── dockerfiles/
    ├── airflow/                 ← imagen custom (scikit-learn, mlflow, boto3, xgboost, catboost)
    ├── fastapi/                 ← API REST + dependencias
    ├── mlflow/                  ← servidor MLflow con soporte S3
    └── postgres/                ← inicialización de schemas Airflow + MLflow
```

---

## Cómo levantar el ambiente

### Requisitos previos

- Docker Desktop (>= 4.x) con al menos **4 GB de RAM** asignados
- Docker Compose v2

### Pasos

```bash
# 1. Pararse en la carpeta del TP
cd Entrega_Operaciones

# 2. Configurar variables de entorno
cp .env.example .env

# En Linux, reemplazar AIRFLOW_UID con el UID real:
# sed -i "s/AIRFLOW_UID=501/AIRFLOW_UID=$(id -u)/" .env

# 3. Crear carpetas necesarias de Airflow
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/config

# 4. Levantar todos los servicios
docker compose --profile all up -d

# 5. Verificar que todos los contenedores están healthy
docker ps -a
```

La primera vez tarda varios minutos en construir las imágenes custom.

### Verificar estado de servicios

```bash
docker compose --profile all ps
```

Todos los servicios deben aparecer como `healthy` o `running`.

### Acceder a los servicios

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | — |
| MinIO Console | http://localhost:9001 | minio / minio123 |
| FastAPI | http://localhost:8800 | — |
| FastAPI Docs | http://localhost:8800/docs | — |

### Detener servicios

```bash
docker compose --profile all down

# Para eliminar también los volúmenes (base de datos y MinIO):
docker compose --profile all down -v
```

---

## Configuración post-arranque

### Airflow — variables y conexiones

Airflow carga automáticamente las variables y conexiones desde los archivos de secrets montados en `/opt/secrets/`. Para agregar o modificar valores, editar los archivos **antes** de levantar el stack (o reiniciar los servicios de Airflow tras el cambio):

**`airflow/secrets/variables.yaml`** — variables globales accesibles desde cualquier DAG:

```yaml
env: prod
mlflow_tracking_uri: http://mlflow:5000
s3_data_bucket: data
s3_mlflow_bucket: mlflow
```

**`airflow/secrets/connections.yaml`** — conexiones registradas en Airflow:

```yaml
pg_conn:
  conn_type: postgres
  host: postgres
  login: airflow
  password: airflow
  schema: airflow

mlflow_conn:
  conn_type: http
  host: mlflow
  port: 5000

s3_conn:
  conn_type: aws
  login: minio
  password: minio123
  extra: '{"endpoint_url": "http://s3:9000"}'
```

> Las variables y conexiones definidas en estos archivos **no aparecen en la UI de Airflow** (son read-only desde el backend de secrets). Para verificar que fueron cargadas, usar la CLI:
> ```bash
> docker exec airflow-apiserver airflow variables get env
> docker exec airflow-apiserver airflow connections get pg_conn
> ```

### MinIO — subir datasets

Los buckets `s3://mlflow` y `s3://data` se crean automáticamente al levantar el stack. Para subir los datasets al bucket de datos, la forma recomendada es ejecutar la notebook incluida:

**Opción A — Notebook (recomendada):**

Ejecutar `notebook_example/test.ipynb` con Docker levantado. Sube automáticamente:
- `ecobici/raw/ecobici_data.csv` — dataset crudo (765 MB)
- `ecobici/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` — splits procesados

**Opción B — MinIO Console (UI web):**

1. Ir a http://localhost:9001 → login `minio` / `minio123`
2. Navegar al bucket `data` → crear carpeta `ecobici/processed/`
3. Subir `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` desde `EDA/dataset/`

**Opción C — MinIO Client (`mc`) desde terminal:**

```bash
mc alias set local http://localhost:9000 minio minio123
mc cp ../EDA/dataset/X_train.csv local/data/ecobici/processed/X_train.csv
mc cp ../EDA/dataset/X_test.csv  local/data/ecobici/processed/X_test.csv
mc cp ../EDA/dataset/y_train.csv local/data/ecobici/processed/y_train.csv
mc cp ../EDA/dataset/y_test.csv  local/data/ecobici/processed/y_test.csv
mc ls local/data/ecobici/processed/
```

### MLflow — estado inicial

El servidor MLflow containerizado **arranca vacío** (sin experimentos ni modelos registrados). Esto es correcto y esperado: los experimentos del TP2 viven en `Entrega_Aprendizaje_Maq/mlflow.db` (SQLite local) y no se migran a este ambiente.

El experiment y los modelos en el Model Registry se crean automáticamente cuando corra el DAG `train_ecobici`. El nombre del experimento que crea el DAG es `TPMAQ2_Ecobici_Duracion_v2` (mismo nombre que los notebooks de TP2, para coherencia).

> Los artefactos (modelos, matrices de confusión) se guardan automáticamente en MinIO (`s3://mlflow/`) cuando se loguea un run desde Airflow.

---

## Perfiles Docker Compose

El `docker-compose.yaml` define perfiles para levantar subconjuntos de servicios:

| Perfil | Servicios incluidos |
|---|---|
| `all` | Todo el stack completo |
| `airflow` | Airflow + PostgreSQL + MinIO + Redis |
| `mlflow` | MLflow + PostgreSQL + MinIO |
| `debug` | Airflow CLI (uso interactivo) |

```bash
# Solo MLflow + MinIO (útil para experimentar):
docker compose --profile mlflow up -d

# Solo Airflow (sin FastAPI):
docker compose --profile airflow up -d
```

---

## Modelo en producción

El modelo campeón del TP2 es un **Random Forest** optimizado con Optuna, entrenado sobre 2.27 M filas del dataset EcoBici 2024.

| Métrica | Valor |
|---|---|
| F1-macro (test) | 0.4663 |
| Accuracy (test) | 0.4912 |
| Inferencia (975 k filas) | 3.7 s |

**Problema:** clasificación multiclase — predecir la categoría de duración de un viaje (`Corto` / `Mediano` / `Largo`) antes de que comience, usando las 19 features disponibles al momento de retirar la bici.

El modelo fue serializado en formato `.skops` y registrado como `champion` en el MLflow Model Registry del TP2. En este ambiente productivo se re-entrena y registra directamente en el MLflow containerizado.

---

## Estado de la entrega

### Completado

- [x] `docker-compose.yaml` con todos los servicios configurados (Airflow, MLflow, FastAPI, MinIO, PostgreSQL, Valkey)
- [x] Dockerfiles custom para Airflow, MLflow, FastAPI y PostgreSQL
- [x] `requirements.txt` de Airflow y FastAPI con dependencias ML (scikit-learn, mlflow, boto3, skops, xgboost, catboost)
- [x] Variables de entorno y secrets de Airflow (`variables.yaml`, `connections.yaml`)
- [x] Inicialización automática de buckets MinIO (`s3://mlflow`, `s3://data`)
- [x] MLflow configurado con backend PostgreSQL y artifact store en MinIO
- [x] Airflow configurado con CeleryExecutor y LocalFilesystemBackend para secrets
- [x] Notebook de upload de datos a MinIO (`notebook_example/test.ipynb`)
- [x] DAG `train_ecobici`: 5 modelos en paralelo → logging en MLflow → champion en Model Registry
- [x] FastAPI `POST /predict`: carga el modelo `champion` desde MLflow Model Registry y devuelve `Corto` / `Mediano` / `Largo`

### Pendiente para la entrega final

- [ ] Prueba end-to-end con Docker: upload de datos → triggerear DAG → verificar MLflow → probar `/predict`

---

## Integrantes

- Carmen María Rodríguez Pastrano
- Federico Agustín Fernández
- Francisco Meaca
- Matías Guido Bovio

---

## Tecnologías

| Capa | Tecnología |
|---|---|
| Orquestación | Apache Airflow 3.x (CeleryExecutor) |
| Ciclo de vida ML | MLflow 2.10 |
| Serving | FastAPI + Uvicorn |
| Data Lake | MinIO (compatible S3) |
| Base de datos | PostgreSQL 15 |
| Broker | Valkey 8.1 (fork open-source de Redis) |
| Contenedores | Docker Compose v2 |
| ML | scikit-learn · XGBoost · CatBoost · skops · boto3 · pandas · numpy |
