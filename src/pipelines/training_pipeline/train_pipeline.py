"""Training pipeline del proyecto Heart_project (predicción de enfermedad cardíaca).

Lee la tabla de features producida por el *feature pipeline*, separa train/test,
entrena el modelo, lo evalúa con métricas apropiadas para el problema y almacena
tanto el modelo entrenado como los resultados de la evaluación.

El script es autónomo: se ejecuta sin argumentos y resuelve por sí mismo las rutas
del proyecto.

    python src/pipelines/training_pipeline/train_pipeline.py

Entradas y salidas
------------------
    data/04_feature/corazon_features.parquet   -> entrada (features + objetivo)
    data/06_models/modelo_corazon.joblib       -> pipeline entrenado
    data/06_models/modelo_corazon_completo.joblib -> pipeline + metadatos
    data/07_model_output/predicciones_test.csv -> predicciones sobre el test
    data/08_reporting/metricas_entrenamiento.csv  -> métricas por conjunto
    data/08_reporting/metricas_entrenamiento.json -> manifiesto de la ejecución
    data/08_reporting/checks_separacion.csv    -> comprobaciones de la separación

Criterio de diseño
------------------
Aquí viven las transformaciones que **aprenden de los datos** —imputación y
escalado— y por eso van dentro de un `Pipeline` de scikit-learn que se ajusta
únicamente con el conjunto de entrenamiento. El *feature pipeline* dejó los
faltantes sin imputar precisamente para que esa decisión se tomara en este punto:
imputar antes del `train_test_split` habría filtrado información del test hacia el
entrenamiento (*data leakage*).

El modelo por defecto es el que ganó la comparación del notebook `5-models`: un
*gradient boosting* regularizado. Los otros dos candidatos siguen disponibles con
`--modelo` para poder reproducir la comparación.

Verificación de la separación
-----------------------------
Antes de entrenar, `validar_separacion_train_test` comprueba que la partición sea
utilizable. Distingue dos niveles:

- **Errores** (detienen el pipeline): índices compartidos, filas idénticas en ambos
  conjuntos, una clase ausente, un conjunto demasiado pequeño o con otras columnas.
  Cualquiera de ellos invalida la evaluación por fuga de información.
- **Advertencias** (se registran y dejan continuar): el reparto se desvía de lo
  configurado, la estratificación no cuadra, algún atributo se distribuye distinto
  entre train y test (Kolmogorov-Smirnov) o el patrón de faltantes difiere.

Con `--estricto` las advertencias también detienen la ejecución.

La métrica principal es **F1**. En un problema clínico con clases equilibradas
(≈48 % de positivos) la exactitud sola es engañosa: importa tanto no dejar pasar
un enfermo (sensibilidad) como no alarmar a un sano (precisión). Se reportan
además ROC AUC y PR AUC, que evalúan el ranking de probabilidades con
independencia del umbral.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# El script debe poder ejecutarse directamente (`python src/pipelines/...`), en cuyo
# caso `src` no está en el path y el paquete `pipelines` no es importable. Se añade
# antes de importar el feature pipeline, del que se reutiliza el contrato de datos.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipelines.feature_pipeline.feature_pipeline import (  # noqa: E402
    OBJETIVO,
    ErrorDeValidacion,
    esquema_features,
    localizar_raiz,
    validar_esquema,
)

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

#: Semilla única para todo el script: split, modelos y validación cruzada.
SEMILLA = 42

#: Proporción del dataset reservada para la evaluación final.
PROPORCION_TEST = 0.20

#: Particiones de la validación cruzada sobre el conjunto de entrenamiento.
N_PARTICIONES = 5

#: Métricas reportadas. F1 es la principal; el resto da el contexto necesario
#: para interpretarla (¿falla por sensibilidad o por precisión?).
#: `zero_division=0` evita un aviso ruidoso cuando un modelo trivial no predice
#: ningún positivo: en ese caso la precisión no está definida y vale 0, no NaN.
METRICAS: dict[str, Any] = {
    "f1": partial(f1_score, zero_division=0),
    "sensibilidad": partial(recall_score, zero_division=0),
    "precision": partial(precision_score, zero_division=0),
    "exactitud_balanceada": balanced_accuracy_score,
    "exactitud": accuracy_score,
}

#: Métricas que necesitan probabilidades en lugar de etiquetas.
METRICAS_PROBABILIDAD: dict[str, Any] = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
}

METRICA_PRINCIPAL = "f1"

#: Umbral de decisión por defecto sobre la probabilidad de la clase positiva.
UMBRAL_DEFECTO = 0.5

# --------------------------------------------------------------------------- #
# Configuración de los checks de la separación train/test
# --------------------------------------------------------------------------- #

#: Desviación máxima admitida entre la prevalencia de train y la de test.
#: Con estratificación la diferencia debería ser casi nula; 3 puntos deja margen
#: para el redondeo al repartir filas y nada más.
TOLERANCIA_ESTRATIFICACION = 0.03

#: Desviación máxima admitida entre la proporción de test real y la configurada.
TOLERANCIA_PROPORCION = 0.02

#: Nivel de significación del test de Kolmogorov-Smirnov por columna. Se usa 0.01
#: y no 0.05 porque se comparan 26 columnas: con 0.05 cabría esperar más de una
#: "diferencia significativa" por puro azar.
ALFA_DISTRIBUCION = 0.01

#: Diferencia máxima admitida en la proporción de nulos de una columna entre
#: train y test. Un desbalance mayor apunta a que la partición no fue aleatoria.
TOLERANCIA_NULOS = 0.10

#: Filas mínimas que debe tener cada conjunto para que las métricas signifiquen algo.
MIN_FILAS_CONJUNTO = 20

#: Modelos disponibles. El primero es el ganador de la comparación del notebook
#: `5-models`; los otros dos permiten reproducir esa comparación desde el script.
MODELOS: dict[str, BaseEstimator] = {
    "gradient_boosting": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=SEMILLA,
    ),
    "regresion_logistica": LogisticRegression(C=1.0, max_iter=5000, random_state=SEMILLA),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=SEMILLA,
        n_jobs=-1,
    ),
}

MODELO_POR_DEFECTO = "gradient_boosting"

# --------------------------------------------------------------------------- #
# Rutas por defecto (relativas a la raíz del proyecto)
# --------------------------------------------------------------------------- #

RUTA_FEATURES = Path("data") / "04_feature" / "corazon_features.parquet"
RUTA_MODELO = Path("data") / "06_models" / "modelo_corazon.joblib"
RUTA_ARTEFACTO = Path("data") / "06_models" / "modelo_corazon_completo.joblib"
RUTA_PREDICCIONES = Path("data") / "07_model_output" / "predicciones_test.csv"
RUTA_METRICAS = Path("data") / "08_reporting" / "metricas_entrenamiento.csv"
RUTA_CHECKS_SPLIT = Path("data") / "08_reporting" / "checks_separacion.csv"
RUTA_MANIFIESTO = Path("data") / "08_reporting" / "metricas_entrenamiento.json"


@dataclass(frozen=True)
class RutasSalida:
    """Destinos de los cinco artefactos que produce el entrenamiento.

    Van agrupados en un objeto y no como cinco parámetros sueltos: siempre viajan
    juntos, y así `ejecutar_pipeline` mantiene una firma legible.
    """

    modelo: Path
    artefacto: Path
    predicciones: Path
    metricas: Path
    manifiesto: Path
    checks_split: Path

    @classmethod
    def por_defecto(cls, raiz: Path) -> RutasSalida:
        """Rutas estándar dentro de las capas de datos del proyecto."""
        return cls(
            modelo=raiz / RUTA_MODELO,
            artefacto=raiz / RUTA_ARTEFACTO,
            predicciones=raiz / RUTA_PREDICCIONES,
            metricas=raiz / RUTA_METRICAS,
            manifiesto=raiz / RUTA_MANIFIESTO,
            checks_split=raiz / RUTA_CHECKS_SPLIT,
        )


logger = logging.getLogger("train_pipeline")


# --------------------------------------------------------------------------- #
# 1. Lectura de datos
# --------------------------------------------------------------------------- #


def leer_features(ruta: Path, validar: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Lee la tabla de features y la separa en atributos (X) y objetivo (y).

    Por defecto revalida el archivo contra el mismo esquema que declaró el feature
    pipeline. No es redundante: entre la generación y el entrenamiento pueden pasar
    días, y el archivo puede haber sido regenerado por otra versión del código o
    editado a mano. El contrato se comprueba en ambos extremos.
    """
    if not ruta.is_file():
        msg = f"No se encuentra la tabla de features: {ruta}. Ejecuta antes el feature pipeline."
        raise FileNotFoundError(msg)

    datos = pd.read_parquet(ruta)
    logger.info("Features leídos desde %s -> %s filas x %s columnas", ruta, *datos.shape)

    if OBJETIVO not in datos.columns:
        msg = f"La tabla de features no contiene la variable objetivo '{OBJETIVO}'"
        raise ErrorDeValidacion(msg)

    if validar:
        validar_esquema(datos, esquema_features(), "entrenamiento")

    atributos = datos.drop(columns=[OBJETIVO])
    objetivo = datos[OBJETIVO]
    logger.info("Atributos: %s | positivos: %.1f%%", atributos.shape[1], objetivo.mean() * 100)
    return atributos, objetivo


def separar_train_test(
    atributos: pd.DataFrame,
    objetivo: pd.Series,
    proporcion_test: float = PROPORCION_TEST,
    semilla: int = SEMILLA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separación estratificada train/test.

    La estratificación mantiene la proporción de enfermos en ambos conjuntos: sin
    ella, con 480 filas, el azar puede dejar un test con una prevalencia muy
    distinta y las métricas dejarían de ser comparables.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        atributos,
        objetivo,
        test_size=proporcion_test,
        random_state=semilla,
        stratify=objetivo,
    )
    logger.info(
        "Train: %s filas (%.1f%% positivos) | Test: %s filas (%.1f%% positivos)",
        len(x_train),
        y_train.mean() * 100,
        len(x_test),
        y_test.mean() * 100,
    )
    return x_train, x_test, y_train, y_test


# --------------------------------------------------------------------------- #
# 2. Verificación de la separación train/test
# --------------------------------------------------------------------------- #


class ErrorDeSeparacion(ErrorDeValidacion):
    """La separación train/test no es utilizable.

    Hereda de `ErrorDeValidacion` para que el manejador de `main` la trate igual:
    mensaje claro, sin traza, y ningún modelo persistido.
    """


@dataclass(frozen=True)
class Comprobacion:
    """Resultado de una única comprobación sobre la separación."""

    nombre: str
    #: "ok" | "advertencia" | "error". Un error impide entrenar; una advertencia
    #: se registra y deja continuar, porque describe un riesgo, no un defecto.
    severidad: str
    detalle: str

    @property
    def es_error(self) -> bool:
        return self.severidad == "error"

    @property
    def es_advertencia(self) -> bool:
        return self.severidad == "advertencia"


@dataclass(frozen=True)
class ResultadoSeparacion:
    """Conjunto de comprobaciones aplicadas a una separación train/test."""

    comprobaciones: list[Comprobacion]

    @property
    def errores(self) -> list[Comprobacion]:
        return [c for c in self.comprobaciones if c.es_error]

    @property
    def advertencias(self) -> list[Comprobacion]:
        return [c for c in self.comprobaciones if c.es_advertencia]

    @property
    def valida(self) -> bool:
        """La separación es utilizable: ninguna comprobación falló como error."""
        return not self.errores

    def tabla(self) -> pd.DataFrame:
        """Los resultados como DataFrame, para inspeccionar o guardar en CSV."""
        return pd.DataFrame(
            [
                {"comprobacion": c.nombre, "severidad": c.severidad, "detalle": c.detalle}
                for c in self.comprobaciones
            ]
        )

    def resumen(self) -> str:
        """Una línea con el recuento por severidad."""
        return (
            f"{len(self.comprobaciones)} comprobaciones: "
            f"{len(self.errores)} errores, {len(self.advertencias)} advertencias"
        )


def _huellas(atributos: pd.DataFrame) -> pd.Series:
    """Huella textual de cada fila, para detectar filas repetidas entre conjuntos.

    Comparar índices no basta: dos filas con índices distintos pueden ser el mismo
    paciente duplicado en el origen. Si una de esas copias cae en train y la otra
    en test, el modelo ya vio la respuesta y el test deja de medir generalización.
    """
    # Los faltantes se representan con un marcador explícito: dos filas con NaN en
    # la misma columna deben considerarse iguales, y `NaN != NaN` lo impediría.
    texto = atributos.astype("string").fillna("<NA>")
    return texto.agg("|".join, axis=1)


def _comprobar_tamanos(
    x_train: pd.DataFrame, x_test: pd.DataFrame, proporcion_test: float
) -> list[Comprobacion]:
    """Los dos conjuntos existen, no están vacíos y reparten según lo configurado."""
    comprobaciones = []
    total = len(x_train) + len(x_test)

    if len(x_train) < MIN_FILAS_CONJUNTO or len(x_test) < MIN_FILAS_CONJUNTO:
        comprobaciones.append(
            Comprobacion(
                "tamaño mínimo",
                "error",
                f"train={len(x_train)}, test={len(x_test)}; se exigen "
                f"{MIN_FILAS_CONJUNTO} filas en cada uno",
            )
        )
    else:
        comprobaciones.append(
            Comprobacion(
                "tamaño mínimo", "ok", f"train={len(x_train)} filas, test={len(x_test)} filas"
            )
        )

    real = len(x_test) / total if total else 0.0
    desviacion = abs(real - proporcion_test)
    severidad = "ok" if desviacion <= TOLERANCIA_PROPORCION else "advertencia"
    comprobaciones.append(
        Comprobacion(
            "proporción de test",
            severidad,
            f"real {real:.1%} vs. configurada {proporcion_test:.0%} (desviación {desviacion:.1%})",
        )
    )
    return comprobaciones


def _comprobar_solape(x_train: pd.DataFrame, x_test: pd.DataFrame) -> list[Comprobacion]:
    """Ninguna fila puede estar en los dos conjuntos: es la fuga más directa."""
    comprobaciones = []

    indices_compartidos = set(x_train.index) & set(x_test.index)
    comprobaciones.append(
        Comprobacion(
            "índices disjuntos",
            "error" if indices_compartidos else "ok",
            f"{len(indices_compartidos)} índices en ambos conjuntos"
            if indices_compartidos
            else "ningún índice compartido",
        )
    )

    repetidas = len(set(_huellas(x_train)) & set(_huellas(x_test)))
    comprobaciones.append(
        Comprobacion(
            "filas duplicadas entre conjuntos",
            "error" if repetidas else "ok",
            f"{repetidas} filas idénticas presentes en train y en test"
            if repetidas
            else "ninguna fila de test aparece en train",
        )
    )
    return comprobaciones


def _comprobar_columnas(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Comprobacion:
    """Ambos conjuntos deben tener las mismas columnas, en el mismo orden."""
    if list(x_train.columns) == list(x_test.columns):
        return Comprobacion("columnas", "ok", f"{x_train.shape[1]} columnas idénticas")

    faltan = set(x_train.columns) ^ set(x_test.columns)
    detalle = f"difieren en {sorted(faltan)}" if faltan else "mismas columnas en distinto orden"
    return Comprobacion("columnas", "error", detalle)


def _comprobar_clases(y_train: pd.Series, y_test: pd.Series) -> list[Comprobacion]:
    """Las dos clases deben aparecer en ambos conjuntos y en proporción similar."""
    comprobaciones = []

    ausentes_train = {0, 1} - set(y_train.unique())
    ausentes_test = {0, 1} - set(y_test.unique())
    if ausentes_train or ausentes_test:
        comprobaciones.append(
            Comprobacion(
                "cobertura de clases",
                "error",
                f"faltan clases en train: {sorted(ausentes_train)}, "
                f"en test: {sorted(ausentes_test)}",
            )
        )
    else:
        comprobaciones.append(
            Comprobacion("cobertura de clases", "ok", "ambas clases presentes en train y test")
        )

    diferencia = abs(float(y_train.mean()) - float(y_test.mean()))
    severidad = "ok" if diferencia <= TOLERANCIA_ESTRATIFICACION else "advertencia"
    comprobaciones.append(
        Comprobacion(
            "estratificación",
            severidad,
            f"positivos train {y_train.mean():.1%} vs. test {y_test.mean():.1%} "
            f"(diferencia {diferencia:.1%})",
        )
    )
    return comprobaciones


def _comprobar_distribuciones(
    x_train: pd.DataFrame, x_test: pd.DataFrame, alfa: float = ALFA_DISTRIBUCION
) -> Comprobacion:
    """Cada atributo debe distribuirse igual en train y en test.

    Se aplica el test de Kolmogorov-Smirnov de dos muestras columna a columna. Es
    no paramétrico, así que sirve igual para las continuas que para los indicadores
    one-hot, y no asume normalidad —que estos datos no tienen—.

    Un p-valor bajo significa que las dos muestras vienen de distribuciones
    distintas: la partición sesgó ese atributo y el test dejó de representar el
    mismo problema. Es una advertencia y no un error porque con 26 columnas y una
    muestra pequeña conviene mirarlo antes que abortar automáticamente.
    """
    sospechosas = []
    # Sólo las columnas comunes: si los esquemas difieren, eso ya lo reporta
    # `_comprobar_columnas` como error y aquí no debe reventar.
    comunes = [c for c in x_train.columns if c in x_test.columns]
    for columna in comunes:
        muestra_train = x_train[columna].dropna().to_numpy(dtype="float64")
        muestra_test = x_test[columna].dropna().to_numpy(dtype="float64")
        if len(muestra_train) < 2 or len(muestra_test) < 2:  # noqa: PLR2004
            continue
        p_valor = float(ks_2samp(muestra_train, muestra_test).pvalue)
        if p_valor < alfa:
            sospechosas.append(f"{columna} (p={p_valor:.4f})")

    if sospechosas:
        return Comprobacion(
            "distribución de atributos (KS)",
            "advertencia",
            f"{len(sospechosas)} de {len(comunes)} atributos difieren "
            f"(alfa={alfa}): {', '.join(sospechosas[:5])}",
        )
    return Comprobacion(
        "distribución de atributos (KS)",
        "ok",
        f"ningún atributo difiere de forma significativa (alfa={alfa})",
    )


def _comprobar_nulos(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Comprobacion:
    """La proporción de faltantes debe ser parecida en ambos conjuntos.

    Un test con muchos más nulos que el train mediría al modelo en condiciones que
    no son las del entrenamiento, y el imputador —ajustado sólo con train— tendría
    que rellenar mucho más de lo previsto.
    """
    comunes = [c for c in x_train.columns if c in x_test.columns]
    nulos_train = x_train[comunes].isna().mean()
    nulos_test = x_test[comunes].isna().mean()
    diferencias = (nulos_train - nulos_test).abs()
    excedidas = diferencias[diferencias > TOLERANCIA_NULOS]

    if not excedidas.empty:
        detalle = ", ".join(f"{col} ({dif:.1%})" for col, dif in excedidas.items())
        return Comprobacion(
            "faltantes comparables",
            "advertencia",
            f"diferencia mayor que {TOLERANCIA_NULOS:.0%} en: {detalle}",
        )
    return Comprobacion(
        "faltantes comparables",
        "ok",
        f"diferencia máxima {diferencias.max():.1%} (tolerancia {TOLERANCIA_NULOS:.0%})",
    )


def validar_separacion_train_test(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    proporcion_test: float = PROPORCION_TEST,
) -> ResultadoSeparacion:
    """Verifica que la separación train/test es utilizable y representativa.

    Es una función independiente y sin efectos secundarios: recibe los cuatro
    conjuntos y devuelve el resultado. No escribe archivos, no lanza excepciones y
    no depende del resto del pipeline, así que puede probarse aislada y reutilizarse
    con cualquier otra partición.

    Comprueba dos cosas distintas:

    1. **Ausencia de fuga** (errores): índices disjuntos y ninguna fila idéntica en
       ambos conjuntos. Si el modelo ya vio una fila del test, el test no mide nada.
    2. **Representatividad** (advertencias): la proporción del reparto, la
       estratificación de la clase, la distribución de cada atributo y el patrón de
       faltantes. Aquí no hay nada roto, pero el test puede haber dejado de
       representar el problema.
    """
    comprobaciones: list[Comprobacion] = []
    comprobaciones.extend(_comprobar_tamanos(x_train, x_test, proporcion_test))
    comprobaciones.append(_comprobar_columnas(x_train, x_test))
    comprobaciones.extend(_comprobar_solape(x_train, x_test))
    comprobaciones.extend(_comprobar_clases(y_train, y_test))
    comprobaciones.append(_comprobar_distribuciones(x_train, x_test))
    comprobaciones.append(_comprobar_nulos(x_train, x_test))
    return ResultadoSeparacion(comprobaciones)


#: Alias en inglés: el requerimiento nombra la función `validate_train_test_split()`.
#: El resto del código está en español, así que ese es el nombre principal y este
#: es sólo un puente para quien busque el nombre del enunciado.
validate_train_test_split = validar_separacion_train_test


def aplicar_resultado_separacion(resultado: ResultadoSeparacion, estricto: bool = False) -> None:
    """Registra las comprobaciones y decide si el pipeline puede continuar.

    Separada a propósito de `validar_separacion_train_test`: una función calcula y
    la otra decide. Así la primera se puede usar para inspeccionar una partición sin
    que aborte nada, y la política —qué es fatal y qué es un aviso— queda en un solo
    sitio y es configurable con `--estricto`.
    """
    for comprobacion in resultado.comprobaciones:
        if comprobacion.es_error:
            logger.error("[split] %s: %s", comprobacion.nombre, comprobacion.detalle)
        elif comprobacion.es_advertencia:
            logger.warning("[split] %s: %s", comprobacion.nombre, comprobacion.detalle)
        else:
            logger.info("[split] %s: %s", comprobacion.nombre, comprobacion.detalle)

    problemas = list(resultado.errores)
    if estricto:
        problemas += resultado.advertencias

    if problemas:
        detalle = "\n".join(f"  - {c.nombre}: {c.detalle}" for c in problemas)
        modo = " (modo estricto: las advertencias cuentan como errores)" if estricto else ""
        msg = f"La separación train/test no superó las comprobaciones{modo}:\n{detalle}"
        raise ErrorDeSeparacion(msg)

    logger.info("[split] %s", resultado.resumen())


def guardar_checks_separacion(resultado: ResultadoSeparacion, ruta: Path) -> Path:
    """Escribe el resultado de las comprobaciones en CSV."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    resultado.tabla().to_csv(ruta, index=False)
    logger.info("Comprobaciones de la separación guardadas: %s", ruta)
    return ruta


# --------------------------------------------------------------------------- #
# 3. Construcción y entrenamiento del modelo
# --------------------------------------------------------------------------- #


def construir_pipeline(nombre_modelo: str = MODELO_POR_DEFECTO) -> Pipeline:
    """Devuelve el pipeline completo: imputación + escalado + estimador.

    Ambos pasos previos aprenden de los datos (la mediana, la media y la desviación)
    y por eso viven aquí dentro: al estar en el `Pipeline`, `cross_validate` los
    reajusta en cada partición y nunca ven el pliegue de validación.
    """
    if nombre_modelo not in MODELOS:
        msg = f"Modelo desconocido: '{nombre_modelo}'. Opciones: {sorted(MODELOS)}"
        raise ValueError(msg)

    return Pipeline(
        [
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
            ("modelo", clone(MODELOS[nombre_modelo])),
        ]
    )


def entrenar(pipeline: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Ajusta el pipeline sobre el conjunto de entrenamiento."""
    pipeline.fit(x_train, y_train)
    logger.info("Modelo entrenado sobre %s filas", len(x_train))
    return pipeline


def validar_cruzado(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_particiones: int = N_PARTICIONES,
) -> dict[str, float]:
    """Estima el rendimiento con validación cruzada estratificada sobre el train.

    Es la estimación honesta que se usa para decidir: el conjunto de test se
    reserva para una única medición final y no participa en ninguna elección.
    """
    cv = StratifiedKFold(n_splits=n_particiones, shuffle=True, random_state=SEMILLA)
    puntuaciones = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring=[
            "f1",
            "recall",
            "precision",
            "balanced_accuracy",
            "accuracy",
            "roc_auc",
            "average_precision",
        ],
        n_jobs=None,
    )

    resultado = {
        "f1": float(puntuaciones["test_f1"].mean()),
        "f1_std": float(puntuaciones["test_f1"].std()),
        "sensibilidad": float(puntuaciones["test_recall"].mean()),
        "precision": float(puntuaciones["test_precision"].mean()),
        "exactitud_balanceada": float(puntuaciones["test_balanced_accuracy"].mean()),
        "exactitud": float(puntuaciones["test_accuracy"].mean()),
        "roc_auc": float(puntuaciones["test_roc_auc"].mean()),
        "pr_auc": float(puntuaciones["test_average_precision"].mean()),
    }
    logger.info(
        "Validación cruzada (%s particiones): F1 = %.4f ± %.4f",
        n_particiones,
        resultado["f1"],
        resultado["f1_std"],
    )
    return resultado


def optimizar_umbral(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_particiones: int = N_PARTICIONES,
) -> float:
    """Busca el umbral que maximiza F1, usando predicciones fuera de muestra.

    El umbral es un hiperparámetro más: elegirlo mirando el test lo contaminaría.
    Por eso se barre sobre las probabilidades de `cross_val_predict`, que para cada
    fila proceden de un modelo que no la vio al entrenar.
    """
    cv = StratifiedKFold(n_splits=n_particiones, shuffle=True, random_state=SEMILLA)
    probabilidades = cross_val_predict(
        clone(pipeline), x_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    candidatos = np.linspace(0.05, 0.95, 91)
    puntuaciones = [
        f1_score(y_train, (probabilidades >= umbral).astype(int)) for umbral in candidatos
    ]
    mejor = float(candidatos[int(np.argmax(puntuaciones))])

    logger.info(
        "Umbral óptimo por F1 fuera de muestra: %.2f (F1 = %.4f)",
        mejor,
        max(puntuaciones),
    )
    return mejor


# --------------------------------------------------------------------------- #
# 4. Evaluación
# --------------------------------------------------------------------------- #


def evaluar(
    pipeline: Pipeline,
    atributos: pd.DataFrame,
    objetivo: pd.Series,
    umbral: float = UMBRAL_DEFECTO,
) -> dict[str, float]:
    """Calcula todas las métricas sobre un conjunto, al umbral indicado."""
    probabilidades = pipeline.predict_proba(atributos)[:, 1]
    predicciones = (probabilidades >= umbral).astype(int)

    resultado = {
        nombre: float(funcion(objetivo, predicciones)) for nombre, funcion in METRICAS.items()
    }
    resultado.update(
        {
            nombre: float(funcion(objetivo, probabilidades))
            for nombre, funcion in METRICAS_PROBABILIDAD.items()
        }
    )
    return resultado


def matriz_confusion(
    pipeline: Pipeline,
    atributos: pd.DataFrame,
    objetivo: pd.Series,
    umbral: float = UMBRAL_DEFECTO,
) -> dict[str, int]:
    """Devuelve la matriz de confusión como diccionario legible.

    En un contexto clínico los dos errores no pesan igual: un falso negativo es un
    enfermo que se va a casa sin diagnóstico.
    """
    probabilidades = pipeline.predict_proba(atributos)[:, 1]
    predicciones = (probabilidades >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(objetivo, predicciones, labels=[0, 1]).ravel()
    return {
        "verdaderos_negativos": int(vn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(vp),
    }


def evaluar_modelo_trivial(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Línea base: predecir siempre la clase mayoritaria.

    Sirve de suelo. Un modelo que no supere claramente esta referencia no está
    aprendiendo nada útil de los atributos.
    """
    trivial = DummyClassifier(strategy="most_frequent")
    trivial.fit(x_train, y_train)
    return evaluar(trivial, x_test, y_test)


def construir_tabla_metricas(
    metricas_cv: dict[str, float],
    metricas_train: dict[str, float],
    metricas_test: dict[str, float],
    metricas_test_umbral: dict[str, float],
    metricas_trivial: dict[str, float],
) -> pd.DataFrame:
    """Reúne todas las evaluaciones en una tabla comparable."""
    tabla = pd.DataFrame(
        {
            "entrenamiento": metricas_train,
            "validacion_cruzada": {k: v for k, v in metricas_cv.items() if k != "f1_std"},
            "test": metricas_test,
            "test_umbral_optimo": metricas_test_umbral,
            "base_trivial": metricas_trivial,
        }
    )
    return tabla.round(4)


# --------------------------------------------------------------------------- #
# 5. Almacenamiento
# --------------------------------------------------------------------------- #


def guardar_modelo(pipeline: Pipeline, ruta: Path) -> Path:
    """Serializa el pipeline entrenado."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ruta)
    logger.info("Modelo guardado: %s", ruta)
    return ruta


def guardar_artefacto(artefacto: dict[str, Any], ruta: Path) -> Path:
    """Serializa el pipeline junto con sus metadatos.

    Un `.joblib` con sólo el estimador no dice con qué umbral se calibró ni con qué
    versión de scikit-learn se entrenó, y ambos datos hacen falta para servirlo sin
    sorpresas.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, ruta)
    logger.info("Artefacto guardado: %s", ruta)
    return ruta


def guardar_metricas(tabla: pd.DataFrame, ruta: Path) -> Path:
    """Escribe la tabla de métricas en CSV."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=True, index_label="metrica")
    logger.info("Métricas guardadas: %s", ruta)
    return ruta


def guardar_predicciones(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    umbral: float,
    ruta: Path,
) -> Path:
    """Guarda las predicciones del test para poder auditar caso por caso."""
    probabilidades = pipeline.predict_proba(x_test)[:, 1]
    predicciones = pd.DataFrame(
        {
            "real": y_test.to_numpy(),
            "probabilidad": probabilidades,
            "prediccion_umbral_defecto": (probabilidades >= UMBRAL_DEFECTO).astype(int),
            "prediccion_umbral_optimo": (probabilidades >= umbral).astype(int),
        },
        index=y_test.index,
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    predicciones.to_csv(ruta, index_label="indice")
    logger.info("Predicciones guardadas: %s", ruta)
    return ruta


def guardar_manifiesto(manifiesto: dict[str, Any], ruta: Path) -> Path:
    """Escribe el manifiesto JSON de la ejecución."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifiesto guardado: %s", ruta)
    return ruta


# --------------------------------------------------------------------------- #
# 6. Orquestación
# --------------------------------------------------------------------------- #


def ejecutar_pipeline(
    features: Path,
    rutas: RutasSalida,
    nombre_modelo: str = MODELO_POR_DEFECTO,
    estricto: bool = False,
) -> dict[str, Any]:
    """Ejecuta el entrenamiento completo y devuelve el resumen de la ejecución.

    La separación se verifica antes de construir nada: si la partición tiene fuga
    de información, entrenar sobre ella sólo produciría métricas infladas.
    """
    atributos, objetivo = leer_features(features)
    x_train, x_test, y_train, y_test = separar_train_test(atributos, objetivo)

    checks_split = validar_separacion_train_test(x_train, x_test, y_train, y_test)
    guardar_checks_separacion(checks_split, rutas.checks_split)
    aplicar_resultado_separacion(checks_split, estricto=estricto)

    pipeline = construir_pipeline(nombre_modelo)
    metricas_cv = validar_cruzado(pipeline, x_train, y_train)
    umbral = optimizar_umbral(pipeline, x_train, y_train)

    entrenar(pipeline, x_train, y_train)

    metricas_train = evaluar(pipeline, x_train, y_train)
    metricas_test = evaluar(pipeline, x_test, y_test)
    metricas_test_umbral = evaluar(pipeline, x_test, y_test, umbral=umbral)
    metricas_trivial = evaluar_modelo_trivial(x_train, y_train, x_test, y_test)

    tabla = construir_tabla_metricas(
        metricas_cv, metricas_train, metricas_test, metricas_test_umbral, metricas_trivial
    )
    logger.info(
        "Test - F1: %.4f | sensibilidad: %.4f | ROC AUC: %.4f",
        metricas_test["f1"],
        metricas_test["sensibilidad"],
        metricas_test["roc_auc"],
    )

    resumen: dict[str, Any] = {
        "generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuente": str(features),
        "modelo": nombre_modelo,
        "n_filas": len(atributos),
        "n_atributos": int(atributos.shape[1]),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "semilla": SEMILLA,
        "metrica_principal": METRICA_PRINCIPAL,
        "umbral_defecto": UMBRAL_DEFECTO,
        "umbral_optimo": umbral,
        "metricas": {
            "validacion_cruzada": metricas_cv,
            "entrenamiento": metricas_train,
            "test": metricas_test,
            "test_umbral_optimo": metricas_test_umbral,
            "base_trivial": metricas_trivial,
        },
        "matriz_confusion_test": matriz_confusion(pipeline, x_test, y_test),
        "brecha_train_test_f1": round(metricas_train["f1"] - metricas_test["f1"], 4),
        "checks_separacion": {
            "resumen": checks_split.resumen(),
            "advertencias": [f"{c.nombre}: {c.detalle}" for c in checks_split.advertencias],
        },
        "version_sklearn": sklearn.__version__,
    }

    guardar_modelo(pipeline, rutas.modelo)
    guardar_artefacto({"pipeline": pipeline, **resumen}, rutas.artefacto)
    guardar_predicciones(pipeline, x_test, y_test, umbral, rutas.predicciones)
    guardar_metricas(tabla, rutas.metricas)
    guardar_manifiesto(resumen, rutas.manifiesto)

    return resumen


# --------------------------------------------------------------------------- #
# 7. Punto de entrada
# --------------------------------------------------------------------------- #


def parsear_argumentos(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Define y analiza los argumentos de línea de comandos."""
    raiz = localizar_raiz()
    parser = argparse.ArgumentParser(
        description="Training pipeline del proyecto Heart_project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defecto = RutasSalida.por_defecto(raiz)
    parser.add_argument("--features", type=Path, default=raiz / RUTA_FEATURES)
    parser.add_argument("--modelo-salida", type=Path, default=defecto.modelo)
    parser.add_argument("--artefacto", type=Path, default=defecto.artefacto)
    parser.add_argument("--predicciones", type=Path, default=defecto.predicciones)
    parser.add_argument("--metricas", type=Path, default=defecto.metricas)
    parser.add_argument("--manifiesto", type=Path, default=defecto.manifiesto)
    parser.add_argument("--checks-split", type=Path, default=defecto.checks_split)
    parser.add_argument(
        "--modelo",
        default=MODELO_POR_DEFECTO,
        choices=sorted(MODELOS),
        help="estimador a entrenar",
    )
    parser.add_argument(
        "--estricto",
        action="store_true",
        help="trata las advertencias de la separación como errores",
    )
    parser.add_argument(
        "--nivel-log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="verbosidad del registro",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el pipeline de forma autónoma. Devuelve 0 si todo fue bien."""
    args = parsear_argumentos(argv)
    logging.basicConfig(
        level=getattr(logging, args.nivel_log),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=== Training pipeline: inicio (modelo: %s) ===", args.modelo)
    try:
        rutas = RutasSalida(
            modelo=args.modelo_salida,
            artefacto=args.artefacto,
            predicciones=args.predicciones,
            metricas=args.metricas,
            manifiesto=args.manifiesto,
            checks_split=args.checks_split,
        )
        resumen = ejecutar_pipeline(
            args.features, rutas, nombre_modelo=args.modelo, estricto=args.estricto
        )
    except ErrorDeValidacion as error:
        # Cubre también `ErrorDeSeparacion`, que hereda de ella. El problema son
        # los datos, no el código: mensaje claro y sin traza.
        logger.error(  # noqa: TRY400
            "VALIDACIÓN FALLIDA - no se entrenó ningún modelo\n%s", error
        )
        return 1
    except (FileNotFoundError, ValueError, OSError):
        logger.exception("El training pipeline falló")
        return 1

    logger.info(
        "=== Training pipeline: fin (F1 test = %.4f) ===", resumen["metricas"]["test"]["f1"]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
