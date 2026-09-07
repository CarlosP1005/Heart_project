"""Inference pipeline del proyecto Heart_project (predicción de enfermedad cardíaca).

Carga el modelo entrenado, lee datos de pacientes nuevos desde un archivo, les aplica
**exactamente las mismas transformaciones** que se usaron al entrenar, y genera,
almacena y visualiza las predicciones.

El script es autónomo: se ejecuta sin argumentos y resuelve por sí mismo las rutas
del proyecto.

    python src/pipelines/inference_pipeline/inference_pipeline.py

Entradas y salidas
------------------
    data/06_models/modelo_corazon_completo.joblib -> modelo + umbral calibrado
    data/01_raw/pacientes_nuevos.csv           -> datos nuevos (sin etiqueta)
    data/07_model_output/predicciones_nuevas.csv -> predicciones por paciente
    data/08_reporting/resumen_inferencia.json  -> manifiesto de la inferencia
    data/08_reporting/figuras/distribucion_predicciones.png -> visualización

Criterio de diseño
------------------
La regla que gobierna este script es que **las transformaciones no se reimplementan**:
se importan del feature pipeline (`sanear_dataset`, `codificar_categoricas`,
`generar_atributos_clinicos`) y las que aprenden de los datos —imputación y escalado—
viajan dentro del `Pipeline` serializado, ya ajustadas con el conjunto de
entrenamiento. Reescribir aquí la lógica de transformación sería la forma más rápida
de introducir un *training-serving skew*: dos implementaciones que empiezan iguales y
divergen en cuanto alguien toca una sola.

Hay dos diferencias deliberadas respecto al feature pipeline:

1. **No se elimina ninguna fila.** En entrenamiento se deduplica y se descartan las
   filas sin etiqueta; aquí cada paciente que entra tiene que salir con su predicción,
   en el mismo orden. Perder una fila en silencio sería devolver el resultado de un
   paciente a otro.
2. **No hay variable objetivo.** Los datos nuevos no traen `disease`; si el archivo la
   incluye, se ignora y se avisa.

Antes de predecir se comprueba que las columnas generadas coincidan exactamente, en
nombre y en orden, con las que vio el modelo al entrenar (`feature_names_in_`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.pipeline import Pipeline

# Backend sin ventana: el script corre en terminal y en CI, sin display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# El script debe poder ejecutarse directamente (`python src/pipelines/...`), en cuyo
# caso `src` no está en el path y el paquete `pipelines` no es importable.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipelines.feature_pipeline.feature_pipeline import (  # noqa: E402
    COLS_PASO_DIRECTO,
    OBJETIVO,
    ErrorDeValidacion,
    codificar_categoricas,
    esquema_features,
    generar_atributos_clinicos,
    localizar_raiz,
    sanear_dataset,
    validar_esquema,
)

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

#: Umbral de decisión de respaldo, si el modelo cargado no trae uno calibrado.
UMBRAL_POR_DEFECTO = 0.5

#: Bandas de riesgo para acompañar la probabilidad con una lectura interpretable.
#: No son una decisión clínica: son una ayuda de lectura para quien revisa el CSV.
BANDAS_RIESGO: list[tuple[float, str]] = [
    (0.20, "muy bajo"),
    (0.40, "bajo"),
    (0.60, "medio"),
    (0.80, "alto"),
    (1.01, "muy alto"),
]

#: Columnas que deben venir en el archivo de entrada (las mismas del crudo,
#: sin la variable objetivo).
COLS_REQUERIDAS: list[str] = [
    "age",
    "sex",
    "chest_pain",
    "rest_bp",
    "chol",
    "fbs",
    "rest_ecg",
    "max_hr",
    "exang",
    "old_peak",
    "slope",
    "ca",
    "thal",
]

# --------------------------------------------------------------------------- #
# Rutas por defecto (relativas a la raíz del proyecto)
# --------------------------------------------------------------------------- #

RUTA_MODELO = Path("data") / "06_models" / "modelo_corazon_completo.joblib"
RUTA_MODELO_SIMPLE = Path("data") / "06_models" / "modelo_corazon.joblib"
RUTA_ENTRADA = Path("data") / "01_raw" / "pacientes_nuevos.csv"
RUTA_PREDICCIONES = Path("data") / "07_model_output" / "predicciones_nuevas.csv"
RUTA_RESUMEN = Path("data") / "08_reporting" / "resumen_inferencia.json"
RUTA_FIGURA = Path("data") / "08_reporting" / "figuras" / "distribucion_predicciones.png"

logger = logging.getLogger("inference_pipeline")


class ErrorDeInferencia(ErrorDeValidacion):
    """No se puede predecir con seguridad sobre estos datos o con este modelo.

    Hereda de `ErrorDeValidacion` para que el manejador de `main` la trate igual:
    mensaje claro, sin traza, y ninguna predicción escrita. Una predicción sobre
    datos que no encajan con lo que el modelo espera es peor que ninguna, porque
    parece válida.
    """


@dataclass(frozen=True)
class ModeloCargado:
    """Modelo listo para predecir, junto con el contexto con el que se entrenó."""

    pipeline: Pipeline
    umbral: float
    origen: Path
    metadatos: dict[str, Any]

    @property
    def atributos_esperados(self) -> list[str]:
        """Columnas que el modelo vio al entrenar, en su orden original."""
        nombres = getattr(self.pipeline, "feature_names_in_", None)
        return [] if nombres is None else [str(n) for n in nombres]


# --------------------------------------------------------------------------- #
# 1. Carga del modelo
# --------------------------------------------------------------------------- #


def cargar_modelo(ruta: Path) -> ModeloCargado:
    """Carga el modelo entrenado desde el almacenamiento del proyecto.

    Acepta las dos formas que produce el training pipeline: el artefacto completo
    (un diccionario con el pipeline y sus metadatos) o el `.joblib` con sólo el
    pipeline. El artefacto es preferible porque trae el **umbral calibrado**: sin
    él habría que asumir 0.5, que no es el umbral con el que se evaluó el modelo.
    """
    if not ruta.is_file():
        msg = f"No se encuentra el modelo: {ruta}. Ejecuta antes el training pipeline."
        raise FileNotFoundError(msg)

    contenido = joblib.load(ruta)

    if isinstance(contenido, dict):
        pipeline = contenido.get("pipeline")
        umbral = float(contenido.get("umbral_optimo", UMBRAL_POR_DEFECTO))
        metadatos = {k: v for k, v in contenido.items() if k != "pipeline"}
    else:
        pipeline = contenido
        umbral = UMBRAL_POR_DEFECTO
        metadatos = {}
        logger.warning(
            "El modelo no trae metadatos: se usa el umbral por defecto %.2f",
            UMBRAL_POR_DEFECTO,
        )

    if not hasattr(pipeline, "predict_proba"):
        msg = (
            f"El objeto cargado desde {ruta} no es un modelo con `predict_proba`: "
            f"es {type(pipeline).__name__}"
        )
        raise ErrorDeInferencia(msg)

    logger.info("Modelo cargado desde %s", ruta)
    logger.info(
        "  tipo: %s | umbral: %.2f | atributos esperados: %s",
        type(pipeline).__name__,
        umbral,
        len(getattr(pipeline, "feature_names_in_", [])),
    )
    if metadatos.get("generado_en"):
        logger.info("  entrenado el %s con %s", metadatos["generado_en"], metadatos.get("modelo"))

    return ModeloCargado(pipeline=pipeline, umbral=umbral, origen=ruta, metadatos=metadatos)


def resolver_ruta_modelo(preferida: Path, alternativa: Path) -> Path:
    """Devuelve el artefacto completo si existe, y si no el pipeline suelto."""
    if preferida.is_file():
        return preferida
    if alternativa.is_file():
        logger.warning("No está el artefacto completo; se usa %s", alternativa)
        return alternativa
    return preferida  # deja que `cargar_modelo` emita el error con la ruta esperada


# --------------------------------------------------------------------------- #
# 2. Lectura de los datos nuevos
# --------------------------------------------------------------------------- #


def leer_datos_nuevos(ruta: Path) -> pd.DataFrame:
    """Lee el archivo de pacientes nuevos como texto, sin inferir tipos.

    Se lee todo como cadena por la misma razón que en el feature pipeline: si pandas
    infiere tipos, un valor corrupto en una columna numérica puede convertir la
    columna entera en texto sin avisar. El saneamiento posterior decide qué es
    interpretable y qué no.
    """
    if not ruta.is_file():
        msg = f"No se encuentra el archivo de datos nuevos: {ruta}"
        raise FileNotFoundError(msg)

    datos = pd.read_csv(ruta, dtype=str, keep_default_na=True)
    logger.info("Datos nuevos leídos desde %s -> %s filas x %s columnas", ruta, *datos.shape)

    if datos.empty:
        msg = f"El archivo {ruta} no contiene ninguna fila"
        raise ErrorDeInferencia(msg)

    faltantes = set(COLS_REQUERIDAS) - set(datos.columns)
    if faltantes:
        msg = f"Al archivo de entrada le faltan columnas obligatorias: {sorted(faltantes)}"
        raise ErrorDeInferencia(msg)

    if OBJETIVO in datos.columns:
        logger.warning(
            "El archivo trae la columna '%s': se ignora, la inferencia no usa etiquetas",
            OBJETIVO,
        )
        datos = datos.drop(columns=[OBJETIVO])

    return datos[COLS_REQUERIDAS]


# --------------------------------------------------------------------------- #
# 3. Transformación (la misma del entrenamiento)
# --------------------------------------------------------------------------- #


def esquema_features_inferencia() -> pa.DataFrameSchema:
    """El esquema de features del entrenamiento, menos la variable objetivo.

    Se deriva del esquema real en lugar de reescribirlo: si mañana el feature
    pipeline añade una columna, esta validación se entera sola.
    """
    original = esquema_features()
    columnas = {nombre: col for nombre, col in original.columns.items() if nombre != OBJETIVO}
    return pa.DataFrameSchema(columns=columnas, strict=True, name="features_inferencia")


def transformar(crudo: pd.DataFrame) -> pd.DataFrame:
    """Aplica las mismas transformaciones deterministas que en el entrenamiento.

    Reutiliza las funciones del feature pipeline en el mismo orden. La diferencia
    es que aquí **no se elimina ni se reordena ninguna fila**: la salida tiene tantas
    filas como la entrada y en la misma posición, para poder devolver cada predicción
    a su paciente.
    """
    saneado = sanear_dataset(crudo)

    features = pd.concat(
        [
            saneado[COLS_PASO_DIRECTO].astype(float),
            codificar_categoricas(saneado),
            generar_atributos_clinicos(saneado),
        ],
        axis=1,
    )

    if len(features) != len(crudo):
        msg = (
            f"La transformación cambió el número de filas: {len(crudo)} -> {len(features)}. "
            "En inferencia cada fila de entrada debe producir exactamente una salida."
        )
        raise ErrorDeInferencia(msg)

    validar_esquema(features, esquema_features_inferencia(), "inferencia")
    logger.info("Features generados: %s filas x %s columnas", *features.shape)
    return features


def alinear_columnas(features: pd.DataFrame, modelo: ModeloCargado) -> pd.DataFrame:
    """Comprueba y ordena las columnas según lo que el modelo vio al entrenar.

    Es la comprobación que separa una predicción válida de una silenciosamente
    equivocada: si el orden de las columnas no coincide, muchos estimadores predicen
    igualmente, pero interpretando el colesterol como si fuera la edad.
    """
    esperadas = modelo.atributos_esperados
    if not esperadas:
        logger.warning("El modelo no declara `feature_names_in_`: no se puede alinear")
        return features

    faltan = [c for c in esperadas if c not in features.columns]
    sobran = [c for c in features.columns if c not in esperadas]
    if faltan or sobran:
        msg = (
            "Las columnas generadas no coinciden con las del entrenamiento.\n"
            f"  faltan: {faltan}\n"
            f"  sobran: {sobran}"
        )
        raise ErrorDeInferencia(msg)

    if list(features.columns) != esperadas:
        logger.info("Reordenando columnas para respetar el orden del entrenamiento")

    logger.info("Columnas alineadas con el modelo: %s atributos", len(esperadas))
    return features[esperadas]


# --------------------------------------------------------------------------- #
# 4. Predicción
# --------------------------------------------------------------------------- #


def clasificar_riesgo(probabilidad: float) -> str:
    """Traduce una probabilidad a una banda de riesgo legible."""
    for limite, etiqueta in BANDAS_RIESGO:
        if probabilidad < limite:
            return etiqueta
    return BANDAS_RIESGO[-1][1]


def predecir(
    modelo: ModeloCargado, features: pd.DataFrame, umbral: float | None = None
) -> pd.DataFrame:
    """Genera las predicciones con la probabilidad, la decisión y su lectura.

    Se devuelve la probabilidad además de la etiqueta porque una decisión binaria
    pierde toda la información de confianza: 0.51 y 0.99 predicen lo mismo y no
    significan lo mismo para quien tiene que actuar.
    """
    umbral_efectivo = modelo.umbral if umbral is None else umbral
    probabilidades = modelo.pipeline.predict_proba(features)[:, 1]
    predicciones = (probabilidades >= umbral_efectivo).astype(int)

    resultado = pd.DataFrame(
        {
            "probabilidad_enfermedad": probabilidades.round(4),
            "prediccion": predicciones,
            "diagnostico": np.where(predicciones == 1, "enfermo", "sano"),
            "nivel_riesgo": [clasificar_riesgo(p) for p in probabilidades],
        },
        index=features.index,
    )

    logger.info(
        "Predicciones generadas con umbral %.2f: %s enfermos, %s sanos",
        umbral_efectivo,
        int(predicciones.sum()),
        int(len(predicciones) - predicciones.sum()),
    )
    return resultado


# --------------------------------------------------------------------------- #
# 5. Almacenamiento y visualización
# --------------------------------------------------------------------------- #


def guardar_predicciones(crudo: pd.DataFrame, predicciones: pd.DataFrame, ruta: Path) -> Path:
    """Guarda las predicciones junto a los datos originales de cada paciente.

    Se adjuntan los datos crudos para que el CSV sea auditable por sí solo: quien lo
    reciba puede ver a qué paciente corresponde cada probabilidad sin cruzarlo con
    otro archivo.
    """
    salida = pd.concat([crudo.reset_index(drop=True), predicciones.reset_index(drop=True)], axis=1)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(ruta, index_label="paciente")
    logger.info("Predicciones guardadas: %s (%s filas)", ruta, len(salida))
    return ruta


def guardar_resumen(resumen: dict[str, Any], ruta: Path) -> Path:
    """Escribe el manifiesto JSON de la inferencia."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resumen guardado: %s", ruta)
    return ruta


def graficar_predicciones(predicciones: pd.DataFrame, umbral: float, ruta: Path) -> Path:
    """Dibuja la distribución de probabilidades con el umbral de decisión marcado.

    Es la visualización que pide el requerimiento y la que más dice de un lote: si
    las probabilidades se agolpan junto al umbral, las decisiones son frágiles y
    conviene revisarlas a mano.
    """
    azul, naranja = "#2a78d6", "#eb6834"
    ruta.parent.mkdir(parents=True, exist_ok=True)

    fig, (izquierda, derecha) = plt.subplots(1, 2, figsize=(11.0, 3.8))

    izquierda.hist(
        predicciones["probabilidad_enfermedad"],
        bins=20,
        range=(0, 1),
        color=azul,
        alpha=0.8,
        edgecolor="white",
    )
    izquierda.axvline(umbral, color=naranja, linestyle="--", label=f"umbral {umbral:.2f}")
    izquierda.set_xlabel("probabilidad de enfermedad")
    izquierda.set_ylabel("pacientes")
    izquierda.set_title("Distribución de las probabilidades")
    izquierda.legend(frameon=False)
    izquierda.grid(alpha=0.2)

    orden = [etiqueta for _, etiqueta in BANDAS_RIESGO]
    conteo = predicciones["nivel_riesgo"].value_counts().reindex(orden, fill_value=0)
    derecha.bar(conteo.index, conteo.to_numpy(), color=azul, alpha=0.85)
    derecha.set_xlabel("nivel de riesgo")
    derecha.set_ylabel("pacientes")
    derecha.set_title("Pacientes por banda de riesgo")
    derecha.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info("Figura guardada: %s", ruta)
    return ruta


# --------------------------------------------------------------------------- #
# 6. Orquestación
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RutasInferencia:
    """Destinos de los archivos que produce la inferencia."""

    predicciones: Path
    resumen: Path
    figura: Path

    @classmethod
    def por_defecto(cls, raiz: Path) -> RutasInferencia:
        """Rutas estándar dentro de las capas de datos del proyecto."""
        return cls(
            predicciones=raiz / RUTA_PREDICCIONES,
            resumen=raiz / RUTA_RESUMEN,
            figura=raiz / RUTA_FIGURA,
        )


def ejecutar_pipeline(
    modelo_ruta: Path,
    entrada: Path,
    rutas: RutasInferencia,
    umbral: float | None = None,
) -> dict[str, Any]:
    """Ejecuta la inferencia completa y devuelve el resumen de la ejecución."""
    modelo = cargar_modelo(modelo_ruta)
    crudo = leer_datos_nuevos(entrada)

    features = transformar(crudo)
    features = alinear_columnas(features, modelo)

    umbral_efectivo = modelo.umbral if umbral is None else umbral
    predicciones = predecir(modelo, features, umbral_efectivo)

    guardar_predicciones(crudo, predicciones, rutas.predicciones)
    graficar_predicciones(predicciones, umbral_efectivo, rutas.figura)

    resumen: dict[str, Any] = {
        "generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": str(modelo_ruta),
        "modelo_entrenado_en": modelo.metadatos.get("generado_en"),
        "tipo_modelo": modelo.metadatos.get("modelo"),
        "entrada": str(entrada),
        "n_pacientes": len(predicciones),
        "n_atributos": features.shape[1],
        "umbral": umbral_efectivo,
        "n_predichos_enfermos": int(predicciones["prediccion"].sum()),
        "n_predichos_sanos": int(len(predicciones) - predicciones["prediccion"].sum()),
        "probabilidad_media": float(predicciones["probabilidad_enfermedad"].mean().round(4)),
        "por_nivel_riesgo": predicciones["nivel_riesgo"].value_counts().to_dict(),
        "faltantes_en_entrada": int(crudo.isna().sum().sum()),
    }
    guardar_resumen(resumen, rutas.resumen)
    return resumen


# --------------------------------------------------------------------------- #
# 7. Punto de entrada
# --------------------------------------------------------------------------- #


def parsear_argumentos(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Define y analiza los argumentos de línea de comandos."""
    raiz = localizar_raiz()
    defecto = RutasInferencia.por_defecto(raiz)
    parser = argparse.ArgumentParser(
        description="Inference pipeline del proyecto Heart_project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=resolver_ruta_modelo(raiz / RUTA_MODELO, raiz / RUTA_MODELO_SIMPLE),
        help="modelo entrenado (.joblib)",
    )
    parser.add_argument("--entrada", type=Path, default=raiz / RUTA_ENTRADA)
    parser.add_argument("--predicciones", type=Path, default=defecto.predicciones)
    parser.add_argument("--resumen", type=Path, default=defecto.resumen)
    parser.add_argument("--figura", type=Path, default=defecto.figura)
    parser.add_argument(
        "--umbral",
        type=float,
        default=None,
        help="umbral de decisión; por defecto el calibrado del modelo",
    )
    parser.add_argument(
        "--nivel-log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="verbosidad del registro",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la inferencia de forma autónoma. Devuelve 0 si todo fue bien."""
    args = parsear_argumentos(argv)
    logging.basicConfig(
        level=getattr(logging, args.nivel_log),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=== Inference pipeline: inicio ===")
    try:
        rutas = RutasInferencia(
            predicciones=args.predicciones, resumen=args.resumen, figura=args.figura
        )
        resumen = ejecutar_pipeline(args.modelo, args.entrada, rutas, umbral=args.umbral)
    except ErrorDeValidacion as error:
        # El problema son los datos o el modelo, no el código: mensaje sin traza.
        logger.error("INFERENCIA FALLIDA - no se generó ninguna predicción\n%s", error)  # noqa: TRY400
        return 1
    except (FileNotFoundError, ValueError, OSError):
        logger.exception("El inference pipeline falló")
        return 1

    logger.info(
        "=== Inference pipeline: fin (%s pacientes, %s predichos enfermos) ===",
        resumen["n_pacientes"],
        resumen["n_predichos_enfermos"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
