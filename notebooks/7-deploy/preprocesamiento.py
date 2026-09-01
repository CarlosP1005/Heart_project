"""Transformadores y utilidades del pipeline de preprocesamiento del proyecto.

Este módulo contiene **exactamente** las mismas clases y funciones que se
definieron en los notebooks para construir el pipeline. Existe por una razón
concreta: `joblib` no guarda el código de las clases personalizadas dentro del
`.joblib`, solo una referencia a ellas. Al cargar el modelo, Python necesita
poder encontrarlas.

Tenerlas en un módulo importable —en lugar de copiadas dentro de cada notebook—
es lo que permite que la aplicación de Streamlit, los tests y cualquier servicio
de producción carguen el mismo modelo sin duplicar código.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import FunctionTransformer

# Marca de versión del módulo. La aplicación la muestra en la barra lateral: si el
# número no coincide con el esperado, es que se está cargando una copia antigua.
VERSION_MODULO = 3

RANDOM_STATE = 42

OBJETIVO = "disease"

COLS_NUM = ["age", "rest_bp", "chol", "max_hr", "old_peak", "ca", "fbs"]

CATEGORIAS_VALIDAS: dict[str, list[str]] = {
    "sex": ["female", "male"],
    "chest_pain": ["asymptomatic", "nonanginal", "nontypical", "typical"],
    "rest_ecg": ["left ventricular hypertrophy", "normal", "st-t wave abnormality"],
    "exang": ["0", "1"],
    "slope": ["1", "2", "3"],
    "thal": ["fixed", "normal", "reversable"],
    "disease": ["0", "1"],
}


def sanear_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte tipos y transforma en NaN todo valor no interpretable."""
    df = df.copy()
    for col in COLS_NUM:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, validas in CATEGORIAS_VALIDAS.items():
        if col in df.columns:
            serie = df[col].astype("string").str.strip().str.lower()
            serie = serie.str.replace(r"\s+", " ", regex=True)
            df[col] = serie.where(serie.isin(validas), other=pd.NA)
    return df


def a_numerico(X: Any) -> pd.DataFrame:
    """Convierte todas las columnas a float; lo no convertible queda como NaN."""
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").astype(float)


def a_texto(X: Any) -> pd.DataFrame:
    """Convierte a texto normalizado usando np.nan para los faltantes."""
    df = pd.DataFrame(X).astype(object)

    def normalizar(valor: Any) -> Any:
        return np.nan if pd.isna(valor) else str(valor).strip().lower()

    aplicar = df.map if hasattr(pd.DataFrame, "map") else df.applymap
    return pd.DataFrame(aplicar(normalizar))


def nombres_log(transformador: Any, nombres: Any) -> NDArray[np.str_]:
    """Nombres de salida de la rama logarítmica."""
    return np.array([f"log_{n}" for n in nombres])


def nombres_raiz(transformador: Any, nombres: Any) -> NDArray[np.str_]:
    """Nombres de salida de la rama de raíz cuadrada."""
    return np.array([f"sqrt_{n}" for n in nombres])


def informacion_mutua(X: Any, y: Any) -> NDArray[np.float64]:
    """Información mutua con semilla fija, para que la selección sea reproducible."""
    resultado = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    return np.asarray(resultado, dtype=np.float64)


A_NUMERICO = FunctionTransformer(a_numerico, feature_names_out="one-to-one")
A_TEXTO = FunctionTransformer(a_texto, feature_names_out="one-to-one")


class SaneadorTipos(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Convierte a numérico o a categoría válida, marcando como NaN lo no interpretable."""

    def __init__(
        self,
        cols_numericas: list[str] | None = None,
        categorias_validas: dict[str, list[str]] | None = None,
    ) -> None:
        self.cols_numericas = cols_numericas
        self.categorias_validas = categorias_validas

    def fit(self, X: pd.DataFrame, y: Any = None) -> SaneadorTipos:
        """Registra los nombres de las columnas de entrada."""
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Aplica el saneamiento columna a columna."""
        X = pd.DataFrame(X).copy()
        for col in self.cols_numericas or []:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        for col, validas in (self.categorias_validas or {}).items():
            if col in X.columns:
                serie = X[col].astype("string").str.strip().str.lower()
                serie = serie.str.replace(r"\s+", " ", regex=True)
                X[col] = serie.where(serie.isin(validas), other=pd.NA)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Devuelve los nombres de las columnas de salida."""
        return np.asarray(self.feature_names_in_ if input_features is None else input_features)


class RecortadorAtipicos(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Winsoriza cada columna numérica a los límites de Tukey aprendidos en fit."""

    def __init__(self, factor: float = 1.5, metodo: str = "iqr") -> None:
        self.factor = factor
        self.metodo = metodo

    def fit(self, X: Any, y: Any = None) -> RecortadorAtipicos:
        """Aprende los límites de recorte a partir del conjunto de entrenamiento."""
        X = pd.DataFrame(X)
        self.nombres_entrada_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]
        numericas = X.apply(pd.to_numeric, errors="coerce").astype(float)
        if self.metodo == "iqr":
            q1, q3 = numericas.quantile(0.25), numericas.quantile(0.75)
            rango = q3 - q1
            self.limite_inferior_ = (q1 - self.factor * rango).to_numpy(dtype=float)
            self.limite_superior_ = (q3 + self.factor * rango).to_numpy(dtype=float)
        else:
            self.limite_inferior_ = numericas.quantile(0.01).to_numpy(dtype=float)
            self.limite_superior_ = numericas.quantile(0.99).to_numpy(dtype=float)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Recorta los valores a los límites aprendidos."""
        numericas = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").astype(float)
        return numericas.clip(lower=self.limite_inferior_, upper=self.limite_superior_, axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Devuelve los nombres de las columnas de salida."""
        return np.asarray(self.nombres_entrada_ if input_features is None else input_features)


class AtributosClinicos(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Genera atributos derivados a partir del conocimiento del dominio clínico."""

    def __init__(self, incluir_interacciones: bool = True) -> None:
        self.incluir_interacciones = incluir_interacciones

    def fit(self, X: Any, y: Any = None) -> AtributosClinicos:
        """Registra los nombres de las columnas de entrada."""
        self.nombres_entrada_ = np.asarray(pd.DataFrame(X).columns)
        self.n_features_in_ = len(self.nombres_entrada_)
        return self

    def transform(self, X: Any) -> NDArray[np.float64]:
        """Calcula los atributos clínicos derivados."""
        X = pd.DataFrame(X).copy()

        def num(columna: str) -> pd.Series:
            """Devuelve una columna convertida a número."""
            return pd.to_numeric(X[columna], errors="coerce")

        edad, fc_max = num("age"), num("max_hr")
        fc_teorica = 220 - edad

        nuevas = pd.DataFrame(index=X.index)
        nuevas["fc_maxima_teorica"] = fc_teorica
        nuevas["reserva_cardiaca"] = fc_max - fc_teorica
        nuevas["pct_fc_alcanzada"] = fc_max / fc_teorica.replace(0, np.nan)

        if self.incluir_interacciones:
            nuevas["ratio_chol_edad"] = num("chol") / edad.replace(0, np.nan)
            nuevas["presion_x_chol"] = num("rest_bp") * num("chol") / 1000.0
            nuevas["indice_riesgo_st"] = num("old_peak") * (num("slope") - 1)

        return np.asarray(nuevas.to_numpy(dtype=float), dtype=np.float64)

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[Any]:
        """Devuelve los nombres de los atributos generados."""
        base = ["fc_maxima_teorica", "reserva_cardiaca", "pct_fc_alcanzada"]
        if self.incluir_interacciones:
            base += ["ratio_chol_edad", "presion_x_chol", "indice_riesgo_st"]
        return np.asarray(base)


OBJETOS_DEL_PIPELINE = (
    SaneadorTipos,
    RecortadorAtipicos,
    AtributosClinicos,
    a_numerico,
    a_texto,
    nombres_log,
    nombres_raiz,
    informacion_mutua,
    sanear_dataset,
)


def registrar_en_main() -> None:
    """Publica las clases del pipeline en el módulo `__main__`.

    El modelo se serializó desde un notebook, así que dentro del `.joblib` las
    referencias apuntan a `__main__.SaneadorTipos`, `__main__.a_texto`, etc. Al
    cargarlo desde una aplicación esos nombres no existen en `__main__` y
    `joblib.load` falla con `AttributeError`.

    Esta función los registra antes de la carga. Es la forma de reutilizar el
    modelo tal cual salió del notebook, sin reentrenarlo ni volver a guardarlo.

    El módulo se busca en `sys.modules` **en el momento de la llamada**, no con un
    `import __main__` al principio del archivo. La diferencia importa: Streamlit
    ejecuta el script de la aplicación como `__main__` y sustituye la entrada de
    `sys.modules` por ese módulo nuevo. Un `import` hecho antes se quedaría
    apuntando al módulo antiguo, y `pickle` —que resuelve por `sys.modules`—
    seguiría sin encontrar las clases.
    """
    modulo_main = sys.modules.get("__main__")
    if modulo_main is None:  # pragma: no cover - situación anómala
        return
    for objeto in OBJETOS_DEL_PIPELINE:
        setattr(modulo_main, objeto.__name__, objeto)


def _cargar_suplantando_main(ruta: Any) -> Any:
    """Carga un `.joblib` haciendo que este módulo ocupe el lugar de `__main__`.

    Es la red de seguridad de `cargar_pipeline`. Durante la deserialización,
    `sys.modules["__main__"]` apunta temporalmente a este módulo, que sí contiene
    todas las clases del pipeline. Funcione como funcione el entorno de ejecución,
    `pickle` las encuentra.
    """
    original = sys.modules.get("__main__")
    sys.modules["__main__"] = sys.modules[__name__]
    try:
        return joblib.load(ruta)
    finally:
        if original is not None:
            sys.modules["__main__"] = original


def alias_modulos_compilados() -> None:
    """Registra con su nombre corto los módulos compilados de scikit-learn.

    Algunas versiones de scikit-learn traen extensiones de Cython cuyas clases
    declaran un `__module__` sin el prefijo del paquete: `CyHalfBinomialLoss`, por
    ejemplo, dice pertenecer a `_loss` y no a `sklearn._loss._loss`. Cuando eso
    ocurre, el `.joblib` guarda la referencia corta, y al abrirlo en una versión que
    no publica ese alias `pickle` falla con
    `ModuleNotFoundError: No module named '_loss'`.

    Registrar los alias antes de cargar evita el problema. No sustituye a usar la
    misma versión de scikit-learn con la que se entrenó —esa sigue siendo la
    recomendación—, pero permite abrir modelos de versiones vecinas.
    """
    candidatos = [
        ("_loss", "sklearn._loss._loss"),
        ("_criterion", "sklearn.tree._criterion"),
        ("_splitter", "sklearn.tree._splitter"),
        ("_tree", "sklearn.tree._tree"),
    ]
    for nombre_corto, ruta_completa in candidatos:
        if nombre_corto in sys.modules:
            continue
        try:
            modulo = importlib.import_module(ruta_completa)
        except ImportError:
            continue
        sys.modules[nombre_corto] = modulo


def cargar_pipeline(ruta: str | Path) -> Any:
    """Carga un pipeline serializado desde un notebook.

    Prepara el entorno en dos frentes antes de deserializar: publica las clases del
    pipeline en `__main__` y registra los alias de los módulos compilados de
    scikit-learn. Si aun así el entorno hace algo inesperado con `__main__`, recurre
    a suplantarlo durante la carga.
    """
    registrar_en_main()
    alias_modulos_compilados()
    try:
        return joblib.load(ruta)
    except AttributeError:
        return _cargar_suplantando_main(ruta)
