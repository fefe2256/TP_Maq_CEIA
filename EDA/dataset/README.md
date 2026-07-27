# Dataset · EcoBici CABA 2024

Esta carpeta **no contiene los archivos CSV** (están listados en `.gitignore`) porque son demasiado pesados para versionar en git (~1,5 GB en total).

## Cómo se pueblan los archivos

Al correr el notebook de principio a fin (`notebook/TP_Grp7_V3_ecobici_presentation_ready.ipynb`) esta carpeta queda con los siguientes archivos:

| Archivo | Origen | Tamaño aprox. |
|---|---|---|
| `ecobici_data.csv` | Descargado automáticamente vía `gdown` en la Sección 1 del notebook | ~800 MB |
| `X_train.csv` | Exportado al final del notebook (Sección 12) | ~465 MB |
| `X_test.csv` | Exportado al final del notebook (Sección 12) | ~200 MB |
| `y_train.csv` | Exportado al final del notebook (Sección 12) | ~34 MB |
| `y_test.csv` | Exportado al final del notebook (Sección 12) | ~14 MB |

## Fuente original

El dataset crudo proviene del **Portal de Datos Abiertos del Gobierno de la Ciudad Autónoma de Buenos Aires**:

- Dataset: [Bicicletas Públicas · Recorridos Realizados 2024](https://data.buenosaires.gob.ar/dataset/bicicletas-publicas)
- Licencia: CC-BY 4.0 según los términos del Portal GCBA.

El notebook usa un mirror en Google Drive (accesible vía `gdown`) para acelerar la descarga y garantizar reproducibilidad.

## Especificación del dataset crudo

| Propiedad | Valor |
|---|---|
| Filas | 3.559.283 |
| Columnas | 17 |
| Rango temporal | Enero – Diciembre 2024 |
| Formato | CSV con separador `,` y encoding UTF-8 |
| Particularidad | Las primeras 2 filas no son datos (título `Tabla 1` + headers con `;`); el notebook las salta con `skiprows=2` |

## Especificación del dataset exportado

Tras correr el pipeline, los 4 CSV finales tienen la siguiente forma:

| Archivo | Shape | Columnas |
|---|---|---|
| `X_train.csv` | ~2.275.322 × 16 | 10 numéricas + 1 label-encoded + 5 one-hot (tras `drop_first=True`) |
| `X_test.csv`  | ~975.139 × 16   | (mismas columnas) |
| `y_train.csv` | ~2.275.322 × 1  | `tipo_viaje` ∈ {Corto, Mediano, Largo} |
| `y_test.csv`  | ~975.139 × 1    | (mismas clases) |

Los índices se preservan en todos los archivos para poder hacer trazabilidad fila a fila contra el dataset crudo si hace falta auditar.
