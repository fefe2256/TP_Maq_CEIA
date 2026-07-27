# EcoBici CABA 2024 · Trabajos Prácticos Finales

**Carrera de Especialización en Inteligencia Artificial · FIUBA · 1B 2026 · Grupo 7**

Este repositorio integra los tres trabajos prácticos finales de la especialización sobre el dataset público de EcoBici (sistema de bicicletas compartidas de CABA) correspondiente al año 2024:

| Carpeta | Materia | Contenido |
|---|---|---|
| `EDA/` | Análisis de Datos | Exploración, limpieza, feature engineering y preparación del dataset |
| `Entrega_Aprendizaje_Maq/` | Aprendizaje de Máquina | Entrenamiento, optimización y comparación de modelos de clasificación |
| `Entrega_Operaciones/` | Arquitectura ML | Ambiente productivo con Docker: Airflow, MLflow, FastAPI y MinIO |

## Integrantes

- Carmen María Rodríguez Pastrano
- Federico Agustín Fernández
- Francisco Meaca
- Matías Guido Bovio

---

## Estructura del repositorio

```
TP_Maq_CEIA/
├── README.md
│
├── EDA/                                                   ← Análisis de Datos
│   ├── README.md
│   ├── requirements.txt
│   ├── notebook/
│   │   └── TP_Grp7_V3_ecobici_presentation_ready.ipynb   ← EDA completo (12 secciones)
│   └── dataset/
│       ├── README.md
│       ├── X_train.csv  ·  X_test.csv                    ← no versionados (~450 MB)
│       └── y_train.csv  ·  y_test.csv                    ← no versionados (~46 MB)
│
├── Entrega_Aprendizaje_Maq/                               ← Aprendizaje de Máquina
│   ├── TPMAQ2_VML_mlflow_parte1.ipynb                     ← Baselines, KNN, LinearSVC, Random Forest
│   ├── TPMAQ2_VML_mlflow_parte2.ipynb                     ← XGBoost, CatBoost, comparación final
│   └── Ecobici_ML_V2.pptx                                 ← presentación de defensa
│
└── Entrega_Operaciones/                                   ← Arquitectura ML
    ├── docker-compose.yaml                                ← orquestación de todos los servicios
    ├── .env.example                                       ← variables de entorno (copiar como .env)
    ├── airflow/
    │   ├── dags/
    │   │   └── train_ecobici.py                           ← DAG de entrenamiento (5 modelos + champion)
    │   └── secrets/                                       ← variables y conexiones
    ├── notebook_example/
    │   └── test.ipynb                                     ← upload de datos a MinIO
    └── dockerfiles/
        ├── airflow/                                       ← imagen custom de Airflow
        ├── fastapi/                                       ← API REST para servir el modelo
        ├── mlflow/                                        ← servidor MLflow
        └── postgres/                                      ← base de datos PostgreSQL
```

> Los CSV del dataset no se versionan por su tamaño (~1.5 GB total). Se generan ejecutando el notebook del EDA. Ver `EDA/dataset/README.md`.

---

## TP1 · Análisis de Datos — EDA

Pipeline completo de preparación del dataset organizado en 12 secciones:

| § | Contenido |
|---|---|
| 1–2 | Carga, limpieza, filtro de outliers por criterio de dominio |
| 2.3 | Análisis de valores faltantes (MCAR/MAR/MNAR) |
| 3 | Feature engineering: encoding cíclico (hora, mes), distancia Haversine, discretización del target |
| 4 | EDA visual: mapa de estaciones, distribución de distancias, top estaciones, distribución del target |
| 5–6 | Selección de features (Mutual Information), split 70/30 stratificado |
| 7–9 | Balance de clases, encoding categóricas (Label + OHE), escalado (StandardScaler sin data leakage) |
| 10–12 | Verificación final, reducción de dimensionalidad, exportación de los 4 CSV |

---

## TP2 · Aprendizaje de Máquina — Modelado

### Problema

Clasificación multiclase supervisada: predecir la categoría de duración (`tipo_viaje`) de un recorrido **antes de que comience**, usando únicamente información disponible al momento de sacar la bici.

| Clase | Rango | Criterio |
|---|---|---|
| **Corto** | < 10 min | Uso rápido intra-barrial |
| **Mediano** | 10 – 30 min | Dentro del tramo gratuito del sistema |
| **Largo** | > 30 min | Excede el tramo gratuito · recreativo o largo recorrido |

**Dataset:** 3.25 M filas · 19 features · split 70/30 stratificado (2.27 M train / 975 k test)

### Resultados

Métrica principal: **F1-macro** (pondera igual las tres clases, penaliza ignorar la clase minoritaria `Largo`).

| Modelo | Accuracy | F1-macro | F1-weighted | Inferencia |
|---|---|---|---|---|
| Baseline Trivial (siempre Mediano) | 0.5565 | 0.2383 | 0.3979 | — |
| Baseline LR (Logistic Regression) | 0.3858 | 0.3827 | 0.3834 | — |
| KNN (Optuna, muestra 50 k) | 0.4540 | 0.3891 | 0.4531 | 816 s* |
| LinearSVC (Optuna) | 0.5440 | 0.3958 | 0.4845 | — |
| XGBoost (Optuna) | 0.5809 | 0.4182 | 0.5168 | 20 s |
| CatBoost (default) | 0.4547 | 0.4414 | 0.4634 | 24 s |
| **Random Forest (Optuna)** | **0.4912** | **0.4663** | **0.5001** | **3.7 s** |

*KNN evaluado sobre 50 k muestras por limitaciones de complejidad O(n).

**Random Forest** es el modelo ganador. Fue registrado como `champion` en el MLflow Model Registry.

Per-class F1: Corto 0.4247 · Mediano 0.5757 · Largo 0.3999

### Notebooks

**Parte 1** (`TPMAQ2_VML_mlflow_parte1.ipynb`): Baseline Trivial, Logistic Regression, KNN, LinearSVC y Random Forest — todos con optimización Optuna + tracking MLflow.

**Parte 2** (`TPMAQ2_VML_mlflow_parte2.ipynb`): XGBoost, CatBoost, comparación final de los 7 modelos consultando MLflow, y registro del modelo campeón en el Model Registry.

### Decisiones técnicas

**F1-macro como métrica.** Un modelo trivial alcanza 55 % de accuracy prediciendo siempre Mediano — F1-macro neutraliza esto penalizando por igual el error en las tres clases.

**Optuna para búsqueda de hiperparámetros.** 20–25 trials con optimización bayesiana y 3-fold cross-validation. La búsqueda se hace sobre submuestras (200–500 k filas) y el modelo final se entrena sobre el dataset completo (2.27 M filas).

**KNN descartado para producción.** 816 s de inferencia sobre 50 k muestras lo hace inviable a escala.

**Gap de performance (0.70 → 0.47).** En el EDA el RF alcanzó F1-macro 0.70 incluyendo features de destino (distancia Haversine, coordenadas de destino). En el escenario real esas features no están disponibles al inicio del viaje, lo que explica la caída.

### Tracking con MLflow

Todos los experimentos están en `Entrega_Aprendizaje_Maq/mlflow.db` (SQLite). Por cada modelo se loguean: hiperparámetros, F1-macro por fold, métricas en test, artefactos (confusion matrix, curvas de aprendizaje, feature importance) y el modelo serializado en formato `.skops`.

---

## TP3 · Arquitectura ML — Ambiente Productivo

Implementación del modelo EcoBici en un ambiente productivo containerizado con Docker Compose.

### Servicios

| Servicio | Rol | Puerto |
|---|---|---|
| Apache Airflow | Orquestación de DAGs | 8080 |
| MLflow | Tracking de experimentos + Model Registry | 5001 |
| FastAPI | API REST para servir predicciones | 8800 |
| MinIO | Data Lake S3 local (datasets y artefactos) | 9000/9001 |
| PostgreSQL | Backend de Airflow y MLflow | 5432 |

### DAG de entrenamiento (`train_ecobici`)

Pipeline MLOps completo orquestado con Airflow dentro de Docker:

1. **`validate_data`** — verifica que los splits existen en MinIO
2. **`train_baseline_trivial`** — DummyClassifier (clase mayoritaria)
3. **`train_logistic_regression`** — LR multinomial sobre 1.5 M muestras
4. **`train_random_forest`** — RF con mejores hiperparámetros de Optuna
5. **`train_xgboost`** — XGBoost con mejores hiperparámetros de Optuna
6. **`train_catboost`** — CatBoost con parámetros default
7. **`select_champion`** — compara por F1-macro y registra el mejor en el Model Registry con alias `champion`

Los modelos 2–6 corren en paralelo. Todos loguean en el MLflow containerizado (`http://mlflow:5000`).

### API de predicción (`POST /predict`)

FastAPI expone un endpoint REST que carga el modelo `champion` desde el MLflow Model Registry al arrancar y sirve predicciones en tiempo real:

```bash
curl -X POST http://localhost:8800/predict \
  -H "Content-Type: application/json" \
  -d '{"hora_sin": 0.5, "hora_cos": 0.866, "mes_sin": 0.5, "mes_cos": 0.866,
       "es_fin_de_semana": 0, "lat_estacion_origen": -34.603,
       "long_estacion_origen": -58.381, "modelo_bicicleta": 1,
       "genero_MALE": 1, "genero_OTHER": 0,
       "dia_semana_Monday": 1, "dia_semana_Saturday": 0, "dia_semana_Sunday": 0,
       "dia_semana_Thursday": 0, "dia_semana_Tuesday": 0, "dia_semana_Wednesday": 0,
       "turno_mañana": 0, "turno_noche": 0, "turno_tarde": 1}'
# → {"tipo_viaje": "Mediano", "descripcion": "Entre 10 y 30 minutos (tramo gratuito del sistema)"}
```

Documentación interactiva disponible en http://localhost:8800/docs.

---

## Cómo ejecutar

### Notebooks de ML (local)

```bash
# Instalar dependencias
pip install scikit-learn xgboost catboost optuna mlflow pandas numpy matplotlib seaborn jupyter

# Ver experimentos en MLflow UI (opcional)
mlflow ui --backend-store-uri sqlite:///Entrega_Aprendizaje_Maq/mlflow.db
# Acceder en http://localhost:5000

# Ejecutar los notebooks en orden
jupyter notebook Entrega_Aprendizaje_Maq/TPMAQ2_VML_mlflow_parte1.ipynb
jupyter notebook Entrega_Aprendizaje_Maq/TPMAQ2_VML_mlflow_parte2.ipynb
```

Ambos notebooks cargan los CSV desde `EDA/dataset/` y deben compartir el mismo `mlflow.db` para que la comparación final de la Parte 2 incluya todos los modelos.

### Ambiente productivo (Docker)

```bash
cd Entrega_Operaciones

# Configurar variables de entorno
cp .env.example .env
# En .env, ajustar AIRFLOW_UID al resultado de: id -u

# Crear carpetas de Airflow
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/config

# Levantar todos los servicios
docker compose --profile all up -d

# Verificar estado
docker ps -a
```

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | — |
| MinIO | http://localhost:9001 | minio / minio123 |
| API | http://localhost:8800 | — |
| Docs API | http://localhost:8800/docs | — |

```bash
# Detener servicios
docker compose --profile all down
```

---

## Tecnologías

**ML:** Python 3.10+ · scikit-learn · XGBoost · CatBoost · Optuna · MLflow · pandas · numpy · matplotlib · seaborn · Jupyter

**Producción:** Docker · Apache Airflow (CeleryExecutor) · MLflow 2.10 · FastAPI · MinIO · PostgreSQL · Valkey

## Licencia

Uso académico. Dataset bajo licencia del Gobierno de la Ciudad Autónoma de Buenos Aires (Datos Abiertos · CC-BY 4.0).
