"""Feature pipeline del proyecto Heart_project (predicción de enfermedad cardíaca).

Lee los datos crudos de `data/01_raw/corazon.csv`, los sanea, deriva los atributos
clínicos definidos en el notebook de ingeniería de características y almacena la
tabla de features en `data/04_feature/corazon_features.parquet`.

El script es autónomo: se ejecuta sin argumentos y resuelve por sí mismo las rutas
del proyecto.

    python src/pipelines/feature_pipeline/feature_pipeline.py

Criterio de diseño
------------------
Aquí sólo se aplican transformaciones **deterministas y sin ajuste** (saneamiento de
tipos, deduplicación, codificación con vocabulario fijo y atributos derivados fila a
fila). Las transformaciones que aprenden parámetros de los datos —imputación,
recorte de atípicos, escalado, discretización y selección de atributos— permanecen
dentro del pipeline de entrenamiento de scikit-learn para evitar fuga de información
(*data leakage*) entre train y test. Por eso la tabla de features conserva los
valores faltantes tal cual.

Validación de datos
-------------------
El pipeline valida en tres puntos y **no persiste nada si alguna validación falla**:

1. `validar_entrada`     — esquema del archivo crudo (columnas y contenido mínimo).
2. `validar_intermedio`  — tipos, rangos, categorías válidas, porcentaje máximo de
   nulos, formato de fechas y unicidad de registros sobre el dataset saneado.
3. `validar_features`    — tipos, ausencia de infinitos, integridad entre campos
   (atributos derivados y codificación one-hot) e integridad entre datasets
   (intermedio vs. features).

Las reglas de columna se declaran con **Pandera** (`pandera.pandas.DataFrameSchema`);
las reglas que cruzan campos, registros o datasets se implementan como funciones
explícitas para poder emitir un mensaje de error preciso. Cualquier incumplimiento
lanza `ErrorDeValidacion`, el script registra el detalle y termina con código 1 sin
haber escrito ningún archivo.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

# --------------------------------------------------------------------------- #
# Configuración del dominio
# --------------------------------------------------------------------------- #

OBJETIVO = "disease"

#: Columnas que deben acabar siendo numéricas; lo no convertible pasa a NaN.
COLS_NUMERICAS: list[str] = ["age", "rest_bp", "chol", "max_hr", "old_peak", "ca", "fbs"]

#: Catálogo de categorías admitidas por columna. Todo lo demás es basura -> NaN.
CATEGORIAS_VALIDAS: dict[str, list[str]] = {
    "sex": ["female", "male"],
    "chest_pain": ["asymptomatic", "nonanginal", "nontypical", "typical"],
    "rest_ecg": ["left ventricular hypertrophy", "normal", "st-t wave abnormality"],
    "exang": ["0", "1"],
    "slope": ["1", "2", "3"],
    "thal": ["fixed", "normal", "reversable"],
    OBJETIVO: ["0", "1"],
}

#: Columnas categóricas binarias y su codificación explícita.
MAPA_BINARIAS: dict[str, dict[str, int]] = {
    "sex": {"female": 0, "male": 1},
    "exang": {"0": 0, "1": 1},
}

#: Columnas ordinales: el orden de la lista define el código 0, 1, 2, ...
ORDEN_ORDINALES: dict[str, list[str]] = {"slope": ["1", "2", "3"]}

#: Columnas nominales que se codifican en one-hot con vocabulario fijo.
COLS_NOMINALES: list[str] = ["chest_pain", "rest_ecg", "thal"]

#: Columnas numéricas que pasan sin transformar a la tabla de features.
COLS_PASO_DIRECTO: list[str] = ["age", "rest_bp", "chol", "max_hr", "old_peak", "ca", "fbs"]

#: Atributos derivados del conocimiento clínico, en el orden en que se generan.
NOMBRES_DERIVADOS: list[str] = [
    "fc_maxima_teorica",
    "reserva_cardiaca",
    "pct_fc_alcanzada",
    "ratio_chol_edad",
    "presion_x_chol",
    "indice_riesgo_st",
]

# --------------------------------------------------------------------------- #
# Configuración de las validaciones
# --------------------------------------------------------------------------- #

#: Rango admitido por columna numérica, según plausibilidad clínica.
#: Un valor fuera de rango no es un error de tipo sino un dato imposible
#: (una presión de 900 mm Hg, una edad de 300 años) y debe detener el pipeline.
RANGOS_VALIDOS: dict[str, tuple[float, float]] = {
    "age": (18.0, 120.0),  # años
    "rest_bp": (60.0, 260.0),  # mm Hg en reposo
    "chol": (80.0, 700.0),  # mg/dl
    "max_hr": (50.0, 220.0),  # lpm; 220 es el máximo teórico absoluto
    "old_peak": (0.0, 10.0),  # depresión del ST en mm
    "ca": (0.0, 3.0),  # nº de vasos coloreados por fluoroscopia
    "fbs": (0.0, 1.0),  # indicador binario
}

#: Proporción máxima de nulos tolerada por columna tras el saneamiento.
#: El dataset real llega al 24 % en `rest_ecg`; por encima de 35 % la columna deja
#: de ser informativa y es preferible detener el pipeline a entrenar con ruido.
MAX_PROPORCION_NULOS = 0.35

#: Columnas de fecha y su formato esperado. Este dataset clínico no contiene
#: ninguna, pero la regla queda implementada y parametrizada: basta añadir aquí
#: una entrada (p. ej. {"fecha_examen": "%Y-%m-%d"}) para que se valide.
COLS_FECHA: dict[str, str] = {}

#: Campos clave cuya combinación debe ser única. El dataset no trae identificador
#: de paciente, así que la unicidad se comprueba sobre el registro completo: tras
#: la deduplicación no puede quedar ninguna fila repetida.
CLAVE_UNICIDAD: list[str] = []

#: Tolerancia al comparar atributos derivados recalculados (aritmética de punto
#: flotante).
TOLERANCIA = 1e-9

# --------------------------------------------------------------------------- #
# Rutas por defecto (relativas a la raíz del proyecto)
# --------------------------------------------------------------------------- #

RUTA_ENTRADA = Path("data") / "01_raw" / "corazon.csv"
RUTA_INTERMEDIA = Path("data") / "02_intermediate" / "corazon.parquet"
RUTA_SALIDA = Path("data") / "04_feature" / "corazon_features.parquet"
RUTA_METADATOS = Path("data") / "04_feature" / "corazon_features_metadata.json"

logger = logging.getLogger("feature_pipeline")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def localizar_raiz(inicio: Path | None = None) -> Path:
    """Devuelve la raíz del proyecto: el primer directorio con `pyproject.toml`.

    Si no encuentra ninguno (por ejemplo al ejecutar el script fuera del
    repositorio) devuelve el directorio de trabajo actual.
    """
    actual = (inicio or Path(__file__)).resolve()
    for candidato in (actual, *actual.parents):
        if (candidato / "pyproject.toml").is_file():
            return candidato
    return Path.cwd()


def normalizar_nombre(valor: str) -> str:
    """Convierte una categoría en un sufijo de columna válido.

    >>> normalizar_nombre("st-t wave abnormality")
    'st_t_wave_abnormality'
    """
    limpio = valor.strip().lower()
    for caracter in (" ", "-", "/"):
        limpio = limpio.replace(caracter, "_")
    while "__" in limpio:
        limpio = limpio.replace("__", "_")
    return limpio


# --------------------------------------------------------------------------- #
# 1. Lectura
# --------------------------------------------------------------------------- #


def leer_datos_crudos(ruta: Path) -> pd.DataFrame:
    """Lee el CSV crudo como texto para no perder los valores no interpretables."""
    if not ruta.is_file():
        msg = f"No se encuentra el archivo de datos crudos: {ruta}"
        raise FileNotFoundError(msg)

    datos = pd.read_csv(ruta, dtype=str, keep_default_na=True)
    logger.info("Datos crudos leídos desde %s -> %s filas x %s columnas", ruta, *datos.shape)

    esperadas = set(COLS_NUMERICAS) | set(CATEGORIAS_VALIDAS)
    faltantes = esperadas - set(datos.columns)
    if faltantes:
        msg = f"Al archivo de entrada le faltan columnas esperadas: {sorted(faltantes)}"
        raise ValueError(msg)

    return datos


# --------------------------------------------------------------------------- #
# 2. Saneamiento
# --------------------------------------------------------------------------- #


def sanear_dataset(datos: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos y marca como NaN todo valor no interpretable.

    No elimina filas: las columnas numéricas se fuerzan a número y las categóricas
    se normalizan (minúsculas, espacios colapsados) contra el catálogo del dominio.
    """
    saneado = datos.copy()

    for col in COLS_NUMERICAS:
        # `astype(float)` fija el tipo pase lo que pase: sin él, un lote sin
        # faltantes daría int64 y otro con faltantes float64, y el esquema de
        # validación fallaría por una diferencia que no es un problema de datos.
        saneado[col] = pd.to_numeric(saneado[col], errors="coerce").astype("float64")

    for col, validas in CATEGORIAS_VALIDAS.items():
        serie = saneado[col].astype("string").str.strip().str.lower()
        serie = serie.str.replace(r"\s+", " ", regex=True)
        saneado[col] = serie.where(serie.isin(validas))  # lo no válido queda como faltante

    convertidos = int(saneado.isna().sum().sum() - datos.isna().sum().sum())
    logger.info("Saneamiento de tipos: %s valores inválidos convertidos a NaN", convertidos)
    return saneado


def depurar_filas(saneado: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos y filas sin variable objetivo."""
    antes = len(saneado)
    depurado = saneado.drop_duplicates().reset_index(drop=True)
    logger.info(
        "Duplicados eliminados: %s filas (%s -> %s)", antes - len(depurado), antes, len(depurado)
    )

    sin_objetivo = int(depurado[OBJETIVO].isna().sum())
    depurado = depurado.dropna(subset=[OBJETIVO]).reset_index(drop=True)
    depurado[OBJETIVO] = depurado[OBJETIVO].astype(int)

    # Al eliminar filas pueden reaparecer duplicados: se vuelve a comprobar.
    depurado = depurado.drop_duplicates().reset_index(drop=True)
    logger.info(
        "Filas sin etiqueta eliminadas: %s -> dataset final %s filas", sin_objetivo, len(depurado)
    )
    return depurado


# --------------------------------------------------------------------------- #
# 3. Ingeniería de características
# --------------------------------------------------------------------------- #


def codificar_categoricas(depurado: pd.DataFrame) -> pd.DataFrame:
    """Codifica binarias, ordinales y nominales con vocabularios fijos.

    Al ser vocabularios conocidos del dominio, la codificación es determinista y no
    depende de los datos observados: no introduce fuga de información. Los valores
    faltantes se propagan como NaN (también en las columnas one-hot).
    """
    codificado = pd.DataFrame(index=depurado.index)

    for col, mapa in MAPA_BINARIAS.items():
        codificado[col] = pd.to_numeric(depurado[col].map(mapa), errors="coerce").astype(float)

    for col, orden in ORDEN_ORDINALES.items():
        mapa_orden = {categoria: posicion for posicion, categoria in enumerate(orden)}
        codificado[col] = pd.to_numeric(depurado[col].map(mapa_orden), errors="coerce").astype(
            float
        )

    for col in COLS_NOMINALES:
        valores = depurado[col]
        ausente = valores.isna().to_numpy(dtype=bool)
        for categoria in CATEGORIAS_VALIDAS[col]:
            nombre = f"{col}_{normalizar_nombre(categoria)}"
            indicador = valores.eq(categoria).fillna(value=False).to_numpy(dtype=bool).astype(float)
            indicador[ausente] = np.nan
            codificado[nombre] = indicador

    return codificado


def generar_atributos_clinicos(depurado: pd.DataFrame) -> pd.DataFrame:
    """Genera los atributos derivados del conocimiento del dominio cardiológico.

    - `fc_maxima_teorica`: frecuencia cardíaca máxima esperada por edad (220 - edad).
    - `reserva_cardiaca`: diferencia entre la FC alcanzada y la teórica.
    - `pct_fc_alcanzada`: proporción de la FC teórica que alcanzó el paciente.
    - `ratio_chol_edad`: colesterol por año de edad.
    - `presion_x_chol`: interacción presión arterial x colesterol (escalada /1000).
    - `indice_riesgo_st`: depresión del ST ponderada por la pendiente del segmento.
    """

    def num(columna: str) -> pd.Series:
        return pd.to_numeric(depurado[columna], errors="coerce")

    edad, fc_max = num("age"), num("max_hr")
    fc_teorica = 220 - edad

    derivados = pd.DataFrame(index=depurado.index)
    derivados["fc_maxima_teorica"] = fc_teorica
    derivados["reserva_cardiaca"] = fc_max - fc_teorica
    derivados["pct_fc_alcanzada"] = fc_max / fc_teorica.replace(0, np.nan)
    derivados["ratio_chol_edad"] = num("chol") / edad.replace(0, np.nan)
    derivados["presion_x_chol"] = num("rest_bp") * num("chol") / 1000.0
    derivados["indice_riesgo_st"] = num("old_peak") * (num("slope") - 1)

    return derivados.astype(float)


def construir_features(crudo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orquesta la transformación completa de datos crudos a tabla de features.

    Devuelve la tabla intermedia (saneada y depurada) y la tabla de features final.
    """
    saneado = sanear_dataset(crudo)
    depurado = depurar_filas(saneado)

    features = pd.concat(
        [
            depurado[COLS_PASO_DIRECTO].astype(float),
            codificar_categoricas(depurado),
            generar_atributos_clinicos(depurado),
            depurado[[OBJETIVO]].astype(int),
        ],
        axis=1,
    )

    logger.info(
        "Features construidos: %s columnas a partir de %s columnas originales",
        features.shape[1],
        crudo.shape[1],
    )
    return depurado, features


# --------------------------------------------------------------------------- #
# 4. Validación de datos
# --------------------------------------------------------------------------- #


class ErrorDeValidacion(RuntimeError):
    """Una regla de calidad o integridad no se cumple.

    Se lanza siempre **antes** de escribir ningún archivo, de modo que un dataset
    que no supera las validaciones nunca llega a persistirse.
    """


def es_finito(serie: pd.Series) -> pd.Series:
    """Comprueba elemento a elemento que no haya infinitos.

    Pandera descarta los nulos antes de aplicar el check (`ignore_na`), así que
    aquí sólo llegan valores presentes: un `inf` sólo puede venir de una división
    por cero mal controlada.
    """
    return pd.Series(np.isfinite(serie.to_numpy(dtype="float64")), index=serie.index)


def esquema_crudo() -> pa.DataFrameSchema:
    """Esquema del archivo crudo: presencia de columnas y contenido mínimo.

    En esta etapa todo es texto (aún no se han convertido los tipos), así que sólo
    se comprueba que estén todas las columnas esperadas y que el archivo no venga
    vacío. `strict=False` permite columnas extra sin romper el pipeline.
    """
    columnas = {
        nombre: pa.Column(str, nullable=True, coerce=True)
        for nombre in [*COLS_NUMERICAS, *CATEGORIAS_VALIDAS]
    }
    return pa.DataFrameSchema(
        columns=columnas,
        strict=False,
        name="datos_crudos",
        checks=pa.Check(
            lambda df: len(df) > 0,
            error="el archivo de entrada no contiene ninguna fila",
        ),
    )


def esquema_intermedio() -> pa.DataFrameSchema:
    """Esquema del dataset saneado: tipos, rangos, categorías y nulos.

    - Las columnas numéricas deben ser `float` y caer dentro del rango clínico.
    - Las categóricas sólo admiten valores del catálogo del dominio.
    - La variable objetivo debe ser entera, binaria y **sin faltantes**: una fila
      sin etiqueta no sirve para entrenar y ya debió eliminarse en la depuración.
    """
    columnas: dict[str, pa.Column] = {}

    for nombre, (minimo, maximo) in RANGOS_VALIDOS.items():
        columnas[nombre] = pa.Column(
            float,
            checks=pa.Check.in_range(minimo, maximo, include_min=True, include_max=True),
            nullable=True,
            description=f"numérica en [{minimo}, {maximo}]",
        )

    for nombre, validas in CATEGORIAS_VALIDAS.items():
        if nombre == OBJETIVO:
            continue
        columnas[nombre] = pa.Column(
            None,
            checks=pa.Check.isin(validas),
            nullable=True,
            description=f"categórica en {validas}",
        )

    columnas[OBJETIVO] = pa.Column(
        int,
        checks=pa.Check.isin([0, 1]),
        nullable=False,
        description="variable objetivo binaria, sin faltantes",
    )

    return pa.DataFrameSchema(columns=columnas, strict=False, name="dataset_intermedio")


def esquema_features() -> pa.DataFrameSchema:
    """Esquema de la tabla de features: todo numérico, finito y en rango.

    Los faltantes siguen permitidos (los imputa el pipeline de entrenamiento), pero
    un infinito sí es un error: sólo puede venir de una división por cero mal
    controlada en los atributos derivados.
    """
    sin_infinitos = pa.Check(es_finito, error="contiene valores infinitos")

    columnas: dict[str, pa.Column] = {
        nombre: pa.Column(float, checks=sin_infinitos, nullable=True)
        for nombre in [*COLS_PASO_DIRECTO, *MAPA_BINARIAS, *ORDEN_ORDINALES]
    }

    for col in COLS_NOMINALES:
        for categoria in CATEGORIAS_VALIDAS[col]:
            nombre = f"{col}_{normalizar_nombre(categoria)}"
            columnas[nombre] = pa.Column(
                float,
                checks=pa.Check.isin([0.0, 1.0]),
                nullable=True,
                description="indicador one-hot",
            )

    for nombre in NOMBRES_DERIVADOS:
        columnas[nombre] = pa.Column(float, checks=sin_infinitos, nullable=True)

    columnas[OBJETIVO] = pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False)

    return pa.DataFrameSchema(columns=columnas, strict=True, name="features")


def validar_esquema(datos: pd.DataFrame, esquema: pa.DataFrameSchema, etapa: str) -> None:
    """Aplica un esquema de Pandera y traduce el fallo a `ErrorDeValidacion`.

    `lazy=True` recoge **todas** las infracciones en una sola pasada, en vez de
    detenerse en la primera: el mensaje resultante muestra el cuadro completo de
    problemas, que es lo útil cuando llega un archivo nuevo con varios defectos.
    """
    try:
        esquema.validate(datos, lazy=True)
    except (SchemaError, SchemaErrors) as error:
        detalle = getattr(error, "failure_cases", None)
        resumen = (
            detalle.head(20).to_string(index=False)
            if isinstance(detalle, pd.DataFrame)
            else str(error)
        )
        msg = f"[{etapa}] el dataset no cumple el esquema '{esquema.name}':\n{resumen}"
        raise ErrorDeValidacion(msg) from error

    logger.info("[%s] esquema '%s': OK", etapa, esquema.name)


def validar_proporcion_nulos(
    datos: pd.DataFrame, etapa: str, umbral: float = MAX_PROPORCION_NULOS
) -> None:
    """Ninguna columna puede superar la proporción de nulos admitida."""
    proporciones = datos.isna().mean()
    excedidas = proporciones[proporciones > umbral]
    if not excedidas.empty:
        detalle = ", ".join(f"{col} ({pct:.1%})" for col, pct in excedidas.items())
        msg = f"[{etapa}] columnas por encima del {umbral:.0%} de nulos: {detalle}"
        raise ErrorDeValidacion(msg)

    logger.info(
        "[%s] proporción de nulos: OK (máximo %.1f%%, umbral %.0f%%)",
        etapa,
        proporciones.max() * 100,
        umbral * 100,
    )


def validar_unicidad(datos: pd.DataFrame, etapa: str, clave: list[str] | None = None) -> None:
    """No puede haber registros repetidos según la clave configurada.

    Con `clave` vacía la unicidad se evalúa sobre la fila completa, que es el
    criterio aplicable a este dataset por no tener identificador de paciente.
    """
    columnas = clave if clave else list(datos.columns)
    repetidos = int(datos.duplicated(subset=columnas).sum())
    if repetidos:
        etiqueta = f"la clave {columnas}" if clave else "el registro completo"
        msg = f"[{etapa}] hay {repetidos} filas duplicadas según {etiqueta}"
        raise ErrorDeValidacion(msg)

    logger.info("[%s] unicidad de registros: OK", etapa)


def validar_formato_fechas(
    datos: pd.DataFrame, etapa: str, columnas: dict[str, str] | None = None
) -> None:
    """Cada columna de fecha debe respetar exactamente su formato declarado."""
    columnas = COLS_FECHA if columnas is None else columnas
    if not columnas:
        logger.info("[%s] formato de fechas: no aplica (el dataset no tiene fechas)", etapa)
        return

    for columna, formato in columnas.items():
        if columna not in datos.columns:
            msg = f"[{etapa}] falta la columna de fecha '{columna}'"
            raise ErrorDeValidacion(msg)

        serie = datos[columna]
        convertidas = pd.to_datetime(serie, format=formato, errors="coerce")
        invalidas = int((convertidas.isna() & serie.notna()).sum())
        if invalidas:
            msg = (
                f"[{etapa}] la columna '{columna}' tiene {invalidas} valores que no "
                f"respetan el formato {formato}"
            )
            raise ErrorDeValidacion(msg)

    logger.info("[%s] formato de fechas: OK", etapa)


def validar_integridad_derivados(features: pd.DataFrame, etapa: str) -> None:
    """Los atributos derivados deben ser coherentes con las columnas que los originan.

    Es una comprobación de integridad **entre campos**: recalcula las fórmulas a
    partir de las columnas originales y exige que coincidan. Detecta un desalineado
    de filas o un cambio en la fórmula que no se propagó.
    """
    esperada_fc = 220 - features["age"]
    diferencia_fc = (features["fc_maxima_teorica"] - esperada_fc).abs()
    if (diferencia_fc > TOLERANCIA).any():
        msg = f"[{etapa}] 'fc_maxima_teorica' no coincide con 220 - age"
        raise ErrorDeValidacion(msg)

    esperada_reserva = features["max_hr"] - features["fc_maxima_teorica"]
    diferencia_reserva = (features["reserva_cardiaca"] - esperada_reserva).abs()
    if (diferencia_reserva > TOLERANCIA).any():
        msg = f"[{etapa}] 'reserva_cardiaca' no coincide con max_hr - fc_maxima_teorica"
        raise ErrorDeValidacion(msg)

    logger.info("[%s] integridad de atributos derivados: OK", etapa)


def validar_integridad_onehot(features: pd.DataFrame, etapa: str) -> None:
    """Cada grupo one-hot debe sumar exactamente 1, o ser NaN por completo.

    Integridad **entre campos**: una fila con dos categorías activas —o con ninguna
    sin ser faltante— significa que la codificación se corrompió.
    """
    for col in COLS_NOMINALES:
        columnas = [f"{col}_{normalizar_nombre(c)}" for c in CATEGORIAS_VALIDAS[col]]
        bloque = features[columnas]
        todo_nulo = bloque.isna().all(axis=1)
        suma = bloque.sum(axis=1)

        incoherentes = int((~todo_nulo & (suma != 1)).sum())
        parcialmente_nulo = int((bloque.isna().any(axis=1) & ~todo_nulo).sum())

        if incoherentes or parcialmente_nulo:
            msg = (
                f"[{etapa}] la codificación one-hot de '{col}' es inconsistente: "
                f"{incoherentes} filas no suman 1 y {parcialmente_nulo} filas tienen "
                f"faltantes parciales"
            )
            raise ErrorDeValidacion(msg)

    logger.info("[%s] integridad de la codificación one-hot: OK", etapa)


def validar_consistencia_datasets(intermedio: pd.DataFrame, features: pd.DataFrame) -> None:
    """Integridad **entre datasets**: intermedio y features deben corresponderse.

    La transformación no elimina ni añade filas, así que ambos deben tener el mismo
    número de registros y la misma variable objetivo, fila a fila.
    """
    if len(intermedio) != len(features):
        msg = (
            f"[features] el número de filas no coincide: intermedio {len(intermedio)} "
            f"vs. features {len(features)}"
        )
        raise ErrorDeValidacion(msg)

    if (
        not intermedio[OBJETIVO]
        .reset_index(drop=True)
        .equals(features[OBJETIVO].reset_index(drop=True))
    ):
        msg = "[features] la variable objetivo no coincide entre intermedio y features"
        raise ErrorDeValidacion(msg)

    logger.info("[features] consistencia entre datasets: OK")


def validar_entrada(crudo: pd.DataFrame) -> None:
    """Valida el archivo crudo antes de transformarlo."""
    validar_esquema(crudo, esquema_crudo(), "entrada")


def validar_intermedio(depurado: pd.DataFrame) -> None:
    """Valida calidad, consistencia y formato del dataset saneado."""
    validar_esquema(depurado, esquema_intermedio(), "intermedio")
    validar_proporcion_nulos(depurado, "intermedio")
    validar_formato_fechas(depurado, "intermedio")
    validar_unicidad(depurado, "intermedio", CLAVE_UNICIDAD)


def validar_features(features: pd.DataFrame, intermedio: pd.DataFrame) -> None:
    """Valida la tabla de features y su integridad frente al dataset intermedio."""
    validar_esquema(features, esquema_features(), "features")
    validar_integridad_derivados(features, "features")
    validar_integridad_onehot(features, "features")
    validar_consistencia_datasets(intermedio, features)


#: Reglas aplicadas, en orden. Se registra en el manifiesto para trazabilidad.
REGLAS_APLICADAS: list[str] = [
    "entrada: esquema de columnas y archivo no vacío",
    "intermedio: tipos esperados por columna",
    "intermedio: rangos de plausibilidad clínica",
    "intermedio: categorías válidas del dominio",
    "intermedio: variable objetivo binaria y sin faltantes",
    f"intermedio: proporción de nulos <= {MAX_PROPORCION_NULOS:.0%} por columna",
    "intermedio: formato de las columnas de fecha declaradas",
    "intermedio: unicidad de registros",
    "features: todo numérico, finito y one-hot en {0, 1}",
    "features: integridad entre campos (atributos derivados)",
    "features: integridad entre campos (codificación one-hot)",
    "features: integridad entre datasets (intermedio vs. features)",
]


# --------------------------------------------------------------------------- #
# 5. Escritura
# --------------------------------------------------------------------------- #


def guardar_parquet(datos: pd.DataFrame, ruta: Path) -> Path:
    """Escribe un DataFrame en formato parquet creando los directorios necesarios."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos.to_parquet(ruta, index=False)
    logger.info("Archivo escrito: %s (%s filas x %s columnas)", ruta, *datos.shape)
    return ruta


def guardar_metadatos(features: pd.DataFrame, entrada: Path, ruta: Path) -> Path:
    """Guarda un pequeño manifiesto JSON para trazabilidad de la ejecución."""
    metadatos = {
        "generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuente": str(entrada),
        "n_filas": int(features.shape[0]),
        "n_columnas": int(features.shape[1]),
        "objetivo": OBJETIVO,
        "columnas": list(features.columns),
        "faltantes_por_columna": {
            col: int(features[col].isna().sum())
            for col in features.columns
            if int(features[col].isna().sum()) > 0
        },
        "validaciones_superadas": REGLAS_APLICADAS,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadatos escritos: %s", ruta)
    return ruta


# --------------------------------------------------------------------------- #
# 6. Punto de entrada
# --------------------------------------------------------------------------- #


def parsear_argumentos(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Define y analiza los argumentos de línea de comandos."""
    raiz = localizar_raiz()
    parser = argparse.ArgumentParser(
        description="Feature pipeline del proyecto Heart_project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--entrada", type=Path, default=raiz / RUTA_ENTRADA, help="CSV crudo")
    parser.add_argument(
        "--intermedio", type=Path, default=raiz / RUTA_INTERMEDIA, help="parquet saneado"
    )
    parser.add_argument(
        "--salida", type=Path, default=raiz / RUTA_SALIDA, help="parquet de features"
    )
    parser.add_argument(
        "--metadatos", type=Path, default=raiz / RUTA_METADATOS, help="manifiesto JSON"
    )
    parser.add_argument(
        "--nivel-log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="verbosidad del registro",
    )
    return parser.parse_args(argv)


def ejecutar_pipeline(
    entrada: Path, intermedio: Path, salida: Path, metadatos: Path
) -> pd.DataFrame:
    """Ejecuta el pipeline completo y devuelve la tabla de features.

    El orden importa: **todas** las validaciones se ejecutan antes de la primera
    escritura, de modo que un fallo de calidad deja el directorio de salida
    intacto en lugar de dejar a medias un dataset inválido.
    """
    crudo = leer_datos_crudos(entrada)
    validar_entrada(crudo)

    depurado, features = construir_features(crudo)
    validar_intermedio(depurado)
    validar_features(features, depurado)

    logger.info("Todas las validaciones superadas; se procede a persistir")
    guardar_parquet(depurado, intermedio)
    guardar_parquet(features, salida)
    guardar_metadatos(features, entrada, metadatos)
    return features


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el pipeline de forma autónoma. Devuelve 0 si todo fue bien."""
    args = parsear_argumentos(argv)
    logging.basicConfig(
        level=getattr(logging, args.nivel_log),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=== Feature pipeline: inicio ===")
    try:
        features = ejecutar_pipeline(args.entrada, args.intermedio, args.salida, args.metadatos)
    except ErrorDeValidacion as error:
        # TRY400 se silencia a propósito: aquí no queremos la traza de
        # `logger.exception`. El fallo es de datos, no de código, y una traza
        # sólo enterraría el mensaje que necesita leer quien corrige el dataset.
        logger.error(  # noqa: TRY400
            "VALIDACIÓN FALLIDA - no se persistió ningún archivo\n%s", error
        )
        return 1
    except (FileNotFoundError, ValueError, OSError):
        logger.exception("El feature pipeline falló")
        return 1

    logger.info("=== Feature pipeline: fin (%s filas listas) ===", len(features))
    return 0


if __name__ == "__main__":
    sys.exit(main())
