"""Lógica del procesamiento por lotes de la demo.

Todo lo que ocurre entre "el usuario sube un archivo" y "el usuario se descarga las
predicciones", sin una sola llamada a Streamlit.

Criterio de diseño
------------------
La interfaz de Streamlit no se puede probar con pytest sin levantar un navegador, así
que aquí vive **toda la lógica** —lectura, validación, predicción, resumen,
serialización— y en la página sólo quedan los widgets. La consecuencia práctica es que
las pruebas del entregable 16 cubren el comportamiento real del lote (qué pasa con un
archivo sin columnas, con uno vacío, con uno de 500 filas) y no sólo el dibujo.

Como en el resto del proyecto, **las transformaciones no se reimplementan**: se importa
`transformar` del `inference_pipeline`, que a su vez importa el `feature_pipeline`. Una
sola ruta de transformación para el entrenamiento, la inferencia por archivo, la demo
individual y este lote.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# `src` al path: la página de Streamlit puede arrancarse directamente y en ese caso
# el paquete `pipelines` no sería importable.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipelines.inference_pipeline.inference_pipeline import (  # noqa: E402
    BANDAS_RIESGO,
    COLS_REQUERIDAS,
    ErrorDeInferencia,
    ModeloCargado,
    alinear_columnas,
    predecir,
    transformar,
)

#: Formatos que acepta el cargador de archivos.
EXTENSIONES_ACEPTADAS = ["csv", "xlsx", "parquet"]

#: Columnas que añade la predicción a las del archivo original.
COLS_PREDICCION = [
    "probabilidad_enfermedad",
    "prediccion",
    "diagnostico",
    "nivel_riesgo",
]

#: Tope de filas por archivo. No es una limitación del modelo —el pipeline predice
#: sobre cualquier tamaño— sino de la interfaz: una tabla de 100 000 filas en el
#: navegador no la lee nadie y hace que la página parezca colgada. Para volúmenes
#: mayores está el `inference_pipeline` por línea de comandos.
MAX_FILAS = 5_000

#: Nombres habituales de columna identificadora. Si el archivo trae alguna, se usa
#: para etiquetar cada fila del resultado; si no, se numeran las filas.
COLS_ID_CANDIDATAS = ["id", "id_paciente", "paciente", "patient_id", "historia"]

#: Nombre de la columna identificadora en la salida.
COL_ID = "id_paciente"


class ErrorDeLote(ErrorDeInferencia):
    """El archivo subido no se puede procesar.

    Hereda de `ErrorDeInferencia` para conservar la misma regla del entregable 14: si
    los datos no encajan con lo que el modelo espera, **no se predice nada**. Un lote
    a medias es peor que ninguno, porque el usuario se descarga un CSV que parece
    completo.
    """


# --------------------------------------------------------------------------- #
# 1. Lectura del archivo subido
# --------------------------------------------------------------------------- #


def leer_archivo(nombre: str, contenido: bytes) -> pd.DataFrame:
    """Lee el archivo subido en un DataFrame de texto.

    Se lee **todo como cadena**, igual que `leer_datos_nuevos` del inference pipeline
    y por el mismo motivo: si pandas infiere tipos, un valor corrupto en una columna
    numérica convierte la columna entera en texto sin avisar, y el saneamiento se
    encuentra un problema distinto del que hay. Aquí decide el feature pipeline qué es
    interpretable y qué no.
    """
    extension = Path(nombre).suffix.lower().lstrip(".")
    buffer = io.BytesIO(contenido)

    if extension not in EXTENSIONES_ACEPTADAS:
        msg = (
            f"Formato no soportado: '.{extension}'. "
            f"Formatos aceptados: {', '.join('.' + e for e in EXTENSIONES_ACEPTADAS)}."
        )
        raise ErrorDeLote(msg)

    try:
        if extension == "csv":
            datos = pd.read_csv(buffer, dtype=str, keep_default_na=True)
        elif extension == "xlsx":
            datos = pd.read_excel(buffer, dtype=str)
        else:
            datos = pd.read_parquet(buffer).astype("string")
    except ImportError as error:
        # pandas delega .xlsx en openpyxl y .parquet en pyarrow, y sólo se entera al
        # abrir el archivo. Sin esto el usuario vería un ModuleNotFoundError crudo,
        # que no dice qué hacer.
        msg = (
            f"Falta la librería con la que pandas lee los .{extension} ({error}). "
            "Instala las dependencias del proyecto con `uv sync`, o guarda el archivo "
            "como .csv."
        )
        raise ErrorDeLote(msg) from error
    except (ValueError, OSError) as error:
        # Archivo corrupto, truncado o con otra extensión de la que dice. Se traduce
        # a `ErrorDeLote` para que la página lo muestre como un mensaje y no como una
        # traza de Python.
        msg = f"No se pudo leer '{nombre}' como .{extension}: {error}"
        raise ErrorDeLote(msg) from error

    if datos.empty:
        msg = f"El archivo '{nombre}' no contiene ninguna fila de datos."
        raise ErrorDeLote(msg)

    if len(datos) > MAX_FILAS:
        msg = (
            f"El archivo trae {len(datos):,} filas y el máximo de la interfaz es "
            f"{MAX_FILAS:,}. Para volúmenes mayores usa el inference pipeline por "
            "línea de comandos."
        )
        raise ErrorDeLote(msg)

    datos.columns = [str(col).strip() for col in datos.columns]
    return datos


def validar_columnas(datos: pd.DataFrame) -> list[str]:
    """Comprueba que estén las 13 variables obligatorias y devuelve las ignoradas.

    Falla en lugar de rellenar lo que falte: una columna ausente que se imputa en
    silencio produce una probabilidad con la misma pinta que las demás y ninguna
    forma de saber que está mal.
    """
    faltantes = [col for col in COLS_REQUERIDAS if col not in datos.columns]
    if faltantes:
        msg = (
            f"Al archivo le faltan {len(faltantes)} columnas obligatorias: "
            f"{', '.join(faltantes)}.\n\n"
            f"El archivo debe traer estas 13: {', '.join(COLS_REQUERIDAS)}."
        )
        raise ErrorDeLote(msg)

    return [col for col in datos.columns if col not in COLS_REQUERIDAS]


def identificadores(datos: pd.DataFrame) -> pd.Series:
    """Devuelve la etiqueta de cada fila: la columna id del archivo, o su número.

    El usuario sube un archivo y se descarga otro; si las filas no van etiquetadas, la
    única forma de casarlas es el orden, y basta con que alguien ordene la tabla en
    Excel para que el resultado se atribuya al paciente equivocado.
    """
    for candidata in COLS_ID_CANDIDATAS:
        if candidata in datos.columns:
            return datos[candidata].astype("string").reset_index(drop=True).rename(COL_ID)

    return pd.Series(range(1, len(datos) + 1), name=COL_ID, dtype="int64")


# --------------------------------------------------------------------------- #
# 2. Predicción del lote
# --------------------------------------------------------------------------- #


def predecir_lote(modelo: ModeloCargado, datos: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Predice sobre todas las filas y devuelve el archivo original con 4 columnas más.

    Se conserva el archivo tal cual llegó —incluidas las columnas que el modelo no
    usa— para que la salida sea auditable por sí sola: quien la reciba ve a qué
    paciente corresponde cada probabilidad sin cruzarla con la entrada.

    Ninguna fila se pierde por el camino: `transformar` falla si el número de filas
    cambia, así que un archivo de 40 pacientes devuelve 40 predicciones o ninguna.
    """
    validar_columnas(datos)

    # Sólo las 13 obligatorias entran al modelo. Si el archivo trae la etiqueta real
    # (`disease`), queda fuera de la predicción pero se conserva en la salida, que es
    # justo lo que hace falta para comparar predicho contra observado.
    crudo = datos[COLS_REQUERIDAS].copy()

    features = alinear_columnas(transformar(crudo), modelo)
    predicciones = predecir(modelo, features, umbral=umbral)

    salida = pd.concat(
        [
            identificadores(datos),
            datos.reset_index(drop=True),
            predicciones.reset_index(drop=True),
        ],
        axis=1,
    )
    # Si el archivo ya traía una columna con el nombre del identificador, la
    # duplicaríamos; se queda la primera.
    return salida.loc[:, ~salida.columns.duplicated()]


# --------------------------------------------------------------------------- #
# 3. Resumen y serialización
# --------------------------------------------------------------------------- #


def resumen_lote(predicciones: pd.DataFrame, umbral: float) -> dict[str, Any]:
    """Cifras agregadas del lote, para leer el resultado sin recorrer la tabla."""
    total = len(predicciones)
    positivos = int(predicciones["prediccion"].sum())
    probabilidades = predicciones["probabilidad_enfermedad"].astype(float)

    conteo = predicciones["nivel_riesgo"].value_counts()
    por_banda = {etiqueta: int(conteo.get(etiqueta, 0)) for _, etiqueta in BANDAS_RIESGO}

    return {
        "filas": total,
        "umbral": round(float(umbral), 4),
        "casos_probables": positivos,
        "casos_descartados": total - positivos,
        "proporcion_positivos": round(positivos / total, 4) if total else 0.0,
        "probabilidad_media": round(float(probabilidades.mean()), 4) if total else 0.0,
        "probabilidad_maxima": round(float(probabilidades.max()), 4) if total else 0.0,
        "por_banda_de_riesgo": por_banda,
    }


def a_csv(predicciones: pd.DataFrame) -> bytes:
    """Serializa el resultado para el botón de descarga.

    UTF-8 con BOM: sin él, Excel en Windows abre el CSV en la codificación del sistema
    y las tildes de 'nivel_riesgo' salen rotas. El destinatario típico de este archivo
    lo abre en Excel.
    """
    # Anotación explícita: el hook de mypy corre sin pandas instalado y ve el retorno
    # de `to_csv` como `Any`, lo que dispararía `no-any-return`.
    texto: str = predicciones.to_csv(index=False)
    return texto.encode("utf-8-sig")


def plantilla() -> pd.DataFrame:
    """Archivo de entrada mínimo, con las 13 columnas y una fila de ejemplo.

    Se ofrece para descargar desde la propia interfaz: es más rápido que leer la
    documentación para averiguar cómo se llama cada columna.
    """
    fila = {
        "id_paciente": "P-001",
        "age": 55,
        "sex": "male",
        "chest_pain": "nontypical",
        "rest_bp": 138,
        "chol": 245,
        "fbs": 0,
        "rest_ecg": "normal",
        "max_hr": 150,
        "exang": 0,
        "old_peak": 1.2,
        "slope": 2,
        "ca": 1,
        "thal": "normal",
    }
    return pd.DataFrame([fila])
