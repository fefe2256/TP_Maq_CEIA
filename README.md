# TP Final · EcoBici CABA 2024 · Aprendizaje de Máquina

**Carrera de Especialización en Inteligencia Artificial · FIUBA · 1B 2026 · Grupo 7**

Trabajo práctico final de la materia *Aprendizaje de Máquina*. Se entrena, optimiza y compara un conjunto de clasificadores supervisados para predecir la categoría de duración de un recorrido de EcoBici antes de que comience, usando únicamente información disponible al momento de inicio del viaje. El proyecto se construye sobre el pipeline de datos preparado en el TP de *Análisis de Datos* (carpeta `EDA/`).

## Integrantes

- Carmen María Rodríguez Pastrano
- Federico Agustín Fernández
- Claudio Marcelo Imberlina
- María Teresa Mallaupoma León
- Matías Guido Bovio

## Descripción del problema

A partir del dataset de recorridos de EcoBici durante 2024 (3,25 M de filas post-limpieza), se plantea un problema de **clasificación multiclase supervisada**: predecir la categoría de duración (`tipo_viaje`) usando 16 features disponibles en el momento de sacar la bici.

| Clase | Rango | Criterio |
|---|---|---|
| **Corto** | < 10 min | Uso rápido intra-barrial |
| **Mediano** | 10 – 30 min | Dentro del tramo gratuito del sistema |
| **Largo** | > 30 min | Excede el tramo gratuito · recreativo o largo recorrido |

**Valor operativo:** el sistema puede anticipar si un viaje excederá el período gratuito, recomendar el tipo de bici adecuada (mecánica vs eléctrica) y distribuir la flota en función de la demanda predicha.

## Resultados: comparación de modelos

La métrica principal es **F1-macro**, que pondera por igual las tres clases y penaliza modelos que ignoran la clase minoritaria (`Largo`).

| Modelo | Accuracy | F1-macro | F1-weighted | Tiempo de inferencia |
|---|---|---|---|---|
| Baseline Trivial (Mediano siempre) | 0.5565 | 0.2383 | 0.3979 | — |
| Baseline LR (Logistic Regression) | 0.3858 | 0.3827 | 0.3834 | — |
| KNN (Optuna, muestra 50 k) | 0.4540 | 0.3891 | 0.4531 | 816 s* |
| LinearSVC (Optuna) | 0.5440 | 0.3958 | 0.4845 | — |
| XGBoost (Optuna) | 0.5809 | 0.4182 | 0.5168 | 20 s |
| CatBoost (default) | 0.4547 | 0.4414 | 0.4634 | 24 s |
| **Random Forest (Optuna)** | **0.4912** | **0.4663** | **0.5001** | **3.7 s** |

*KNN se evaluó sobre una muestra de 50 k registros por limitaciones de complejidad O(n).

**Random Forest** es el modelo ganador con el mejor F1-macro (0.4663) y el menor tiempo de inferencia entre los modelos de ensamble (3.7 s sobre 975 k muestras). Fue registrado como modelo `champion` en el MLflow Model Registry.

**Per-class F1 del modelo ganador:**

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Corto | — | — | 0.4247 |
| Mediano | — | — | 0.5757 |
| Largo | — | — | 0.3999 |

## Estructura del repositorio

```
TP_Maq_CEIA/
├── README.md                                              ← este archivo
│
├── EDA/                                                   ← pipeline de preparación de datos (TP Análisis de Datos)
│   ├── README.md
│   ├── requirements.txt
│   ├── notebook/
│   │   └── TP_Grp7_V3_ecobici_presentation_ready.ipynb   ← EDA completo (12 secciones)
│   └── dataset/
│       ├── README.md
│       ├── X_train.csv  ·  X_test.csv                    ← features escaladas (no versionadas)
│       └── y_train.csv  ·  y_test.csv                    ← targets (no versionados)
│
├── Entrega_Aprendizaje_Maq/                               ← modelado ML (TP Aprendizaje de Máquina)
│   ├── TPMAQ2_VML_mlflow_parte1.ipynb                     ← Baselines, KNN, LinearSVC, Random Forest
│   ├── TPMAQ2_VML_mlflow_parte2.ipynb                     ← XGBoost, CatBoost, comparación final
│   └── Ecobici_ML_V2.pptx                                 ← presentación de defensa
│
└── Entrega_AMq2/                                          ← ambiente productivo (TP Arquitectura ML)
    ├── docker-compose.yaml                                ← orquestación de todos los servicios
    ├── .env.example                                       ← variables de entorno (copiar como .env)
    ├── airflow/
    │   ├── dags/                                          ← DAGs de Airflow (ETL, entrenamiento)
    │   └── secrets/                                       ← variables y conexiones de Airflow
    └── dockerfiles/
        ├── airflow/                                       ← imagen custom de Airflow
        ├── fastapi/                                       ← API REST para servir el modelo
        ├── mlflow/                                        ← servidor MLflow
        └── postgres/                                      ← base de datos PostgreSQL
```

Los CSV del dataset no se versionan (demasiado pesados, ~1.5 GB total). Se generan ejecutando el notebook del EDA.

## Notebooks de modelado

### Parte 1 · `TPMAQ2_VML_mlflow_parte1.ipynb`

1. Setup: carga de datos (X/y train/test del EDA), inicialización de MLflow, callback de Optuna para logging por trial
2. **Baseline Trivial** — predice siempre la clase mayoritaria (Mediano) — F1-macro: 0.2383
3. **Baseline ML** — Logistic Regression multinomial — F1-macro: 0.3827
4. **KNN** con Optuna (n_neighbors=6) — F1-macro: 0.3891 (evaluado sobre 50 k)
5. **LinearSVC** con Optuna (C=0.68) — F1-macro: 0.3958
6. **Random Forest** con Optuna — F1-macro: 0.4663
7. Curva de aprendizaje del Random Forest (análisis de overfitting: train 0.70 vs val 0.45)

### Parte 2 · `TPMAQ2_VML_mlflow_parte2.ipynb`

1. Recarga de datos (kernel independiente de la Parte 1)
2. **XGBoost** con Optuna (198 estimators, depth=13, lr=0.141) — F1-macro: 0.4182
3. **CatBoost** con parámetros default (iterations=300, lr=0.1, depth=8) — F1-macro: 0.4414
4. Comparación final: consulta los 7 modelos desde MLflow y presenta el ranking
5. Registro del modelo campeón (Random Forest) en el MLflow Model Registry
6. Conclusiones y análisis del gap de performance

## Tracking de experimentos con MLflow

Todos los experimentos están registrados en `mlflow.db` (SQLite local). La UI se puede levantar con:

```bash
mlflow ui --backend-store-uri sqlite:///Entrega_Aprendizaje_Maq/mlflow.db
# Acceder en http://localhost:5000
```

**Estructura del experimento:** un único experimento llamado `EcoBici_Clasificacion` con un run por modelo. Los runs de búsqueda de hiperparámetros con Optuna generan runs anidados (un run hijo por trial).

**Qué se loguea por cada modelo:**

| Categoría | Detalle |
|---|---|
| Parámetros | Hiperparámetros del modelo (los del mejor trial de Optuna) |
| Métricas de CV | F1-macro por fold durante la búsqueda |
| Métricas en test | Accuracy, F1-macro, F1-weighted, Precision-macro, Recall-macro |
| Artefactos | Confusion matrix, curva de aprendizaje (RF), feature importance (RF, XGBoost), gráfico de evolución de Optuna |
| Modelo | Serializado en formato `.skops` (scikit-learn secure format) |

**Model Registry:** el Random Forest (modelo campeón) fue registrado en el MLflow Model Registry con el alias `champion`. Esto permite cargarlo en producción con:

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/EcoBici_RF_Champion@champion")
```

## Decisiones técnicas destacadas

**Métrica principal: F1-macro.** Se eligió sobre accuracy porque las clases están desbalanceadas (26 / 56 / 18 %) y un modelo trivial puede alcanzar 55 % de accuracy prediciendo siempre Mediano. F1-macro penaliza igual el error en las tres clases.

**Optimización con Optuna.** Se utilizó Optuna (optimización bayesiana) con 20–25 trials y cross-validation de 3 folds para KNN, LinearSVC, Random Forest y XGBoost. Cada trial queda registrado como un run anidado en MLflow.

**Submuestreo para búsqueda, dataset completo para entrenamiento final.** La búsqueda de hiperparámetros se realiza sobre muestras de 200–500 k registros para acotar el tiempo de cómputo; el modelo final se entrena sobre el 70 % completo (~2.27 M de filas).

**KNN inviable a escala.** Con 2.27 M de filas de entrenamiento, KNN tardó 816 s en inferencia sobre 50 k muestras. Se documenta el resultado como referencia pero se descarta para producción.

**Gap de performance (0.70 → 0.47).** El Random Forest tenía un F1-macro de 0.70 en el EDA cuando se incluían features de destino (distancia Haversine, coordenadas de destino). En el escenario realista de predicción al inicio del viaje esas features no están disponibles, lo que explica la caída a 0.47.

**MLflow como sistema de tracking.** Todos los experimentos quedan registrados en `mlflow.db` (SQLite). Se logean hiperparámetros, métricas por fold, métricas en test, artefactos (confusion matrices, curvas de aprendizaje, feature importances) y el modelo serializado.

## Tecnologías

Python 3.10+ · scikit-learn · XGBoost · CatBoost · Optuna · MLflow · pandas · numpy · matplotlib · seaborn · Jupyter

## Cómo ejecutar

### Notebooks de ML (local)

**Requisitos:** tener los 4 CSV del dataset en `EDA/dataset/` (generados por el notebook de EDA). Ver `EDA/README.md`.

```bash
# Instalar dependencias
pip install scikit-learn xgboost catboost optuna mlflow pandas numpy matplotlib seaborn jupyter

# Lanzar MLflow UI para inspeccionar los experimentos (opcional)
mlflow ui --backend-store-uri sqlite:///Entrega_Aprendizaje_Maq/mlflow.db

# Ejecutar los notebooks en orden
jupyter notebook Entrega_Aprendizaje_Maq/TPMAQ2_VML_mlflow_parte1.ipynb
jupyter notebook Entrega_Aprendizaje_Maq/TPMAQ2_VML_mlflow_parte2.ipynb
```

Los notebooks son independientes entre sí (cada uno carga los CSV desde `EDA/dataset/`) pero deben ejecutarse sobre el mismo `mlflow.db` para que la comparación final de la Parte 2 incluya los modelos de la Parte 1.

### Ambiente productivo (Docker)

**Requisitos:** tener [Docker](https://docs.docker.com/engine/install/) instalado y corriendo.

```bash
cd Entrega_AMq2

# Copiar y configurar variables de entorno
cp .env.example .env
# Editar .env y ajustar AIRFLOW_UID al resultado de: id -u

# Crear carpetas necesarias para Airflow
mkdir -p airflow/dags airflow/logs airflow/plugins airflow/config

# Levantar todos los servicios
docker compose --profile all up -d

# Verificar que todos estén healthy
docker ps -a
```

Una vez levantado, acceder a:

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | — |
| MinIO | http://localhost:9001 | minio / minio123 |
| API | http://localhost:8800 | — |
| Docs API | http://localhost:8800/docs | — |

Para detener los servicios:

```bash
docker compose --profile all down
```

## Dataset

**Fuente:** Portal de Datos Abiertos de Buenos Aires — EcoBici Recorridos Realizados 2024

| Métrica | Valor |
|---|---|
| Filas crudas | 3.559.283 |
| Filas post-limpieza | 3.250.461 |
| Train / Test | 2.275.322 / 975.139 (split 70/30 stratificado) |
| Features | 16 (temporales, geográficas de origen, perfil de usuario) |
| Target | `tipo_viaje` ∈ {Corto, Mediano, Largo} |

## Licencia

Uso académico. Dataset bajo licencia del Gobierno de la Ciudad Autónoma de Buenos Aires (Datos Abiertos · CC-BY 4.0).
