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
        saneado[col] = pd.to_numeric(saneado[col], errors="coerce")

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
# 4. Escritura
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
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadatos escritos: %s", ruta)
    return ruta


# --------------------------------------------------------------------------- #
# 5. Punto de entrada
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
    """Ejecuta el pipeline completo y devuelve la tabla de features."""
    crudo = leer_datos_crudos(entrada)
    depurado, features = construir_features(crudo)
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
    except (FileNotFoundError, ValueError, OSError):
        logger.exception("El feature pipeline falló")
        return 1

    logger.info("=== Feature pipeline: fin (%s filas listas) ===", len(features))
    return 0


if __name__ == "__main__":
    sys.exit(main())
