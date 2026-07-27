# TP Final · EcoBici CABA 2024

**Análisis de Datos · Carrera de Especialización en Inteligencia Artificial · FIUBA · 1B 2026 · Grupo 7**

Trabajo práctico integrador de la materia *Análisis de Datos*. Se realiza el análisis exploratorio completo del dataset público de EcoBici (sistema de bicicletas compartidas de la Ciudad Autónoma de Buenos Aires) correspondiente al año 2024, se plantea un problema de Machine Learning supervisado, y se deja el dataset preparado para la etapa de modelado.

## 📓 Visualización del notebook

- **Render completo con mapa interactivo y todos los gráficos** → [Ver en nbviewer](https://nbviewer.org/github/cimberlina/grupo7-eda-ecobici/blob/main/notebook/TP_Grp7_V3_ecobici_presentation_ready.ipynb)
- **Render estático en GitHub** (sin mapa de Folium, por bloqueo de JS) → [notebook/TP_Grp7_V3_ecobici_presentation_ready.ipynb](notebook/TP_Grp7_V3_ecobici_presentation_ready.ipynb)

> El mapa de estaciones de la Sección 4.1 usa Folium (JavaScript) y **GitHub no lo renderiza** por razones de seguridad — aparece el mensaje *"Make this Notebook Trusted to load map"*. Para ver el mapa sin ejecutar el notebook localmente, usar el link de nbviewer de arriba.

## Integrantes

- Carmen María Rodríguez Pastrano
- Federico Agustín Fernández
- Claudio Marcelo Imberlina
- María Teresa Mallaupoma León
- Matías Guido Bovio

## Descripción del problema

A partir del dataset de recorridos realizados en EcoBici durante 2024, se plantea un problema de **clasificación multiclase supervisada**: predecir la categoría de duración de un recorrido *antes de que comience*, usando únicamente información disponible al momento de inicio del viaje.

**Variable objetivo** `tipo_viaje`, obtenida discretizando la duración en tres clases con fundamento operativo:

| Clase | Rango | Criterio |
|---|---|---|
| **Corto** | < 10 min | Uso rápido intra-barrial |
| **Mediano** | 10 – 30 min | Dentro del tramo gratuito del sistema |
| **Largo** | > 30 min | Excede el tramo gratuito · recreativo o largo recorrido |

**Valor operativo:** el sistema puede anticipar si un viaje va a exceder el período gratuito, recomendar el tipo de bici adecuado (mecánica vs eléctrica) y distribuir la flota en función de la demanda predicha.

## Dataset

**Fuente:** [Portal de Datos Abiertos de Buenos Aires](https://data.buenosaires.gob.ar/dataset/bicicletas-publicas) — EcoBici Recorridos Realizados 2024

| Métrica | Valor |
|---|---|
| Filas crudas | 3.559.283 |
| Columnas originales | 17 |
| Filas post-filtro | 3.250.461 |
| Rango temporal | Enero – Diciembre 2024 |
| Estaciones únicas | ~395 |

El CSV crudo (~800 MB) y los 4 CSV exportados del pipeline no se incluyen en el repo (están en `.gitignore`). El notebook descarga el archivo crudo automáticamente mediante `gdown` desde un mirror de Google Drive, y exporta los 4 CSV finales en la carpeta `dataset/`. Ver `dataset/README.md` para detalles.

## Estructura del repositorio

```
TP_Final_Grupo7_EcoBici/
├── README.md                                                   ← este archivo
├── requirements.txt                                            ← dependencias Python
├── .gitignore                                                  ← ignora CSVs y artefactos
├── notebook/
│   └── TP_Grp7_V3_ecobici_presentation_ready.ipynb             ← notebook principal
├── presentacion/
│   └── Presentacion_Grupo7_Final.pptx                          ← presentación de defensa (6 slides)
└── dataset/
    └── README.md                                               ← cómo se puebla esta carpeta
```

Las carpetas `notebook/`, `presentacion/` y `dataset/` separan el código, la entrega visual y los datos respectivamente. Los CSV del dataset no se versionan — se generan ejecutando el notebook.

## Cómo ejecutar

**Requisitos:** Python 3.10 o superior y `pip` instalado.

```bash
# Clonar el repositorio
git clone https://github.com/cimberlina/grupo7-eda-ecobici.git
cd grupo7-eda-ecobici

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt

# Abrir el notebook
jupyter notebook notebook/TP_Grp7_V3_ecobici_presentation_ready.ipynb
```

Importante: **el notebook debe ejecutarse desde la carpeta `notebook/`** (es el comportamiento por defecto de Jupyter si lo abrís desde ahí). Al correr la Sección 1, el CSV crudo se descarga en `dataset/`; al correr la Sección 12, los 4 CSV finales se exportan en `dataset/`.

Tiempo de ejecución aproximado en una laptop estándar: **5 – 8 minutos** (descarga inicial del dataset incluida).

## Contenido del notebook

El notebook está organizado en 12 secciones que reflejan el pipeline completo de preparación del dataset:

| § | Sección | Cubre |
|---|---|---|
| 1 | Carga del dataset | Descarga con `gdown` · `read_csv` con manejo de CSV malformado |
| 1.1 | Primer vistazo | `head` · `describe` · clasificación de las 17 variables en 4 familias funcionales |
| 2 | Limpieza inicial | Conversión de tipos · filtro de duración (3 métodos evaluados: Dominio / IQR / Winsor) · boxplot antes/después |
| 2.3 | Análisis integral de faltantes | Tipificación MCAR/MAR/MNAR por variable · decisiones de tratamiento |
| 3 | Feature engineering | Encoding cíclico (hora, mes) · distancia Haversine · discretización del target |
| 4 | EDA visual complementario | Mapa de estaciones · distribución de distancias · top 15 estaciones · distribución del target · duración por hora |
| 5 | Selección de features | Mutual Information como filtro · planteo del problema de ML supervisado |
| 6 | Split | 70/30 stratified con `stratify=y` |
| 7 | Distribución del target | Verificación del balance de clases (26 / 56 / 18 %) · decisión razonada de no rebalancear |
| 8 | Encoding de categóricas | Label Encoding · One-Hot Encoding |
| 9 | Escalado | StandardScaler con `fit` solo en train (sin data leakage) |
| 10 | Verificación final | Shapes · tipos · nulos |
| 11 | Reducción de dimensionalidad | Consolidación: MI como filtro + cíclico y Haversine como extracción · evaluación de PCA como alternativa general |
| 12 | Exportación | Generación de los 4 CSV del dataset final en `../dataset/` |

## Decisiones técnicas destacadas

**Problema de ML.** Se plantea como **clasificación multiclase** en lugar de regresión sobre la duración cruda. La discretización neutraliza los outliers extremos (bicis no devueltas) sin pérdida de información relevante, y produce un modelo más robusto y operativamente útil.

**Tratamiento de outliers.** Se evaluaron tres enfoques (criterio de dominio, IQR de Tukey, Winsor p5/p95) y se aplicó el filtro por dominio (30 s < duración < 24 h) porque los umbrales son físicamente significativos. IQR y Winsor habrían eliminado viajes largos válidos que son justamente la clase minoritaria `Largo`.

**Valores faltantes.** Tipificados según Little & Rubin:
- `genero` (0,34 %) como MNAR → eliminación de filas (imputar introduciría sesgo).
- `fecha_destino_recorrido` (0,10 %) como MAR → no se usa como feature (se calcula después del viaje, sería data leakage).
- `modelo_bicicleta` y coordenadas → MCAR, sin tratamiento especial.

**Feature engineering.** Encoding cíclico sin/cos para `hora` y `mes` (preserva topología circular); distancia Haversine como feature geométrica que condensa 4 coordenadas en 1 escalar interpretable.

**Balance de clases.** Se documenta la decisión de **no aplicar SMOTE ni otros rebalanceos**: el desbalance es moderado (ratio 3:1), `stratify=y` preserva proporciones en el split, y los modelos candidatos downstream (RF, XGBoost, LogReg) soportan `class_weight='balanced'` que es más eficiente que generar datos sintéticos.

**Reducción de dimensionalidad.** Se aplican dos técnicas de extracción de features con justificación de dominio — encoding cíclico (23→2, 11→2) y Haversine (4→1) — más Mutual Information como método de filtro. Se evalúa PCA como alternativa general y se concluye justificadamente no aplicarlo (dimensionalidad final ya baja, features continuas ya ortogonales, binarias violan supuestos de PCA, modelos de árbol downstream no se benefician).

## Resultados del pipeline

**Dataset exportado** a la carpeta `dataset/` (ver `dataset/README.md`):

| Archivo | Contenido | Shape aprox. |
|---|---|---|
| `dataset/X_train.csv` | Features de entrenamiento escaladas | 2.275.322 × 16 |
| `dataset/X_test.csv` | Features de test escaladas | 975.139 × 16 |
| `dataset/y_train.csv` | Target de entrenamiento | 2.275.322 × 1 |
| `dataset/y_test.csv` | Target de test | 975.139 × 1 |

Las features exportadas incluyen: encoding cíclico de tiempo (4 columnas), `es_fin_de_semana` (1), distancia Haversine (1), coordenadas de origen y destino (4), `modelo_bicicleta` label-encoded (1), y dummies de género y día de semana (5 columnas post-OHE con `drop_first=True`).

## Próximos pasos (fuera del alcance del TP)

El modelado en sí queda fuera del alcance del TP de Análisis de Datos. La etapa siguiente incluiría:

- Entrenamiento y comparación de modelos base: **Logistic Regression**, **Random Forest** y **XGBoost** con `class_weight='balanced'`.
- Métrica principal: **F1-score macro** para priorizar la clase minoritaria `Largo`.
- Análisis de *feature importance* para interpretabilidad.
- Evaluación del impacto operativo: ¿con qué precisión puede el sistema anticipar viajes que exceden el tramo gratuito?

## Tecnologías

Python 3.10+ · pandas · numpy · matplotlib · seaborn · scikit-learn · scipy · folium · gdown · Jupyter

## Licencia

Uso académico. Dataset bajo licencia del Gobierno de la Ciudad Autónoma de Buenos Aires (Datos Abiertos).
