# Dataset · EcoBici CABA 2024

Esta carpeta **no contiene los archivos CSV** (están listados en `.gitignore`) porque son demasiado pesados para versionar en git (~1,5 GB en total).

---

## Opción A — Correr el notebook de EDA (recomendada)

Al correr el notebook completo (`EDA/notebook/TP_Grp7_V3_ecobici_presentation_ready.ipynb`), la Sección 1 descarga el dataset crudo automáticamente desde Google Drive usando `gdown`, y la Sección 12 exporta los 4 splits procesados. Esta carpeta queda con todos los archivos listos.

```bash
pip install gdown folium jupyter
jupyter notebook EDA/notebook/TP_Grp7_V3_ecobici_presentation_ready.ipynb
```

---

## Opción B — Descarga manual

Si solo necesitás los archivos sin correr el notebook completo, podés descargarlos directamente.

### Dataset crudo (`ecobici_data.csv`)

**Link de descarga directa (Google Drive):**

```
https://drive.google.com/uc?id=1t-QLtl__u1JCIXtjXEMUy_VfIttB0Kbp
```

O usando `gdown` desde la terminal:

```bash
pip install gdown
gdown 1t-QLtl__u1JCIXtjXEMUy_VfIttB0Kbp -O EDA/dataset/ecobici_data.csv
```

**Fuente original:** [Portal de Datos Abiertos GCBA — Bicicletas Públicas 2024](https://data.buenosaires.gob.ar/dataset/bicicletas-publicas) · Licencia CC-BY 4.0.

### Dónde poner los archivos

Todos los archivos deben quedar en esta misma carpeta (`EDA/dataset/`):

```
EDA/dataset/
├── ecobici_data.csv   ← dataset crudo (765 MB)
├── X_train.csv        ← generado por el notebook EDA (316 MB)
├── X_test.csv         ← generado por el notebook EDA (135 MB)
├── y_train.csv        ← generado por el notebook EDA  (32 MB)
└── y_test.csv         ← generado por el notebook EDA  (14 MB)
```

Los splits (`X_train`, `X_test`, `y_train`, `y_test`) se generan corriendo el notebook de EDA sobre el raw. Si los recibís por otro canal (de un compañero del grupo, por ejemplo), también van en esta carpeta.

---

## Relación con el pipeline Docker

Una vez que los archivos están en `EDA/dataset/`, el siguiente paso es subirlos a MinIO ejecutando la notebook `Entrega_Operaciones/notebook_example/test.ipynb` con el stack Docker levantado. El DAG de Airflow los leerá desde ahí para entrenar los modelos.

```
EDA/dataset/  →  test.ipynb  →  MinIO s3://data/ecobici/  →  DAG train_ecobici
```

Ver `Entrega_Operaciones/INSTRUCTIVO.md` para el paso a paso completo.

---

## Especificación del dataset crudo

| Propiedad | Valor |
|---|---|
| Filas | 3.559.283 |
| Columnas | 17 |
| Rango temporal | Enero – Diciembre 2024 |
| Formato | CSV con separador `,` y encoding UTF-8 |
| Particularidad | Las primeras 2 filas no son datos (título `Tabla 1` + headers con `;`); el notebook las salta con `skiprows=2` |

## Especificación de los splits exportados

| Archivo | Shape | Descripción |
|---|---|---|
| `X_train.csv` | ~2.275.322 × 19 | Features de entrenamiento |
| `X_test.csv`  | ~975.139 × 19   | Features de evaluación |
| `y_train.csv` | ~2.275.322 × 1  | Target de entrenamiento (`tipo_viaje` ∈ {Corto, Mediano, Largo}) |
| `y_test.csv`  | ~975.139 × 1    | Target de evaluación |
