"""Pruebas de la lógica de la demo de Streamlit.

Se prueba lo que se puede probar sin levantar la interfaz: la localización del
modelo, la conversión del formulario en una predicción y las señales clínicas. Los
widgets de Streamlit no se prueban aquí — para eso está la evidencia de la app
funcionando en el notebook 15.

Como en el inference pipeline, todo va con **modelo dummy y datos sintéticos**: la
suite no depende de que exista un modelo entrenado en el repositorio.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from demo import app
from pipelines.feature_pipeline.feature_pipeline import CATEGORIAS_VALIDAS, RANGOS_VALIDOS
from pipelines.inference_pipeline.inference_pipeline import (
    BANDAS_RIESGO,
    COLS_REQUERIDAS,
    alinear_columnas,
    cargar_modelo,
    predecir,
    transformar,
)

N_SENALES = 4


@pytest.fixture(scope="module")
def modelo_dummy_en_disco(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Artefacto dummy con la misma forma que el real, guardado en disco."""
    crudo = pd.DataFrame(
        [{col: str(valor) for col, valor in ejemplo.items()} for ejemplo in app.EJEMPLOS.values()]
    )[COLS_REQUERIDAS]
    features = transformar(crudo)

    pipeline = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
            ("modelo", DummyClassifier(strategy="stratified", random_state=0)),
        ]
    )
    pipeline.fit(features, [0, 1, 1])

    ruta: Path = tmp_path_factory.mktemp("modelos") / "modelo_completo.joblib"
    joblib.dump({"pipeline": pipeline, "umbral_optimo": 0.31, "modelo": "dummy"}, ruta)
    return ruta


# --------------------------------------------------------------------------- #
# Localización del modelo
# --------------------------------------------------------------------------- #


def test_localizar_modelo_prefiere_models_versionado() -> None:
    """`models/` va primero: es la ruta que existe en el despliegue.

    `data/` está en el `.gitignore`, así que en Streamlit Cloud sólo llega la copia
    de `models/`. Si el orden fuera el contrario, la app funcionaría en local y
    fallaría al desplegar.
    """
    assert app.RUTAS_MODELO[0].parent.name == "models"
    assert app.RUTAS_MODELO[1].parent.name == "06_models"


def test_localizar_modelo_devuelve_una_ruta_existente_o_la_primera() -> None:
    """Con o sin modelo presente, siempre devuelve una ruta utilizable."""
    ruta = app.localizar_modelo()
    assert ruta in app.RUTAS_MODELO


# --------------------------------------------------------------------------- #
# Casos de ejemplo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("nombre", list(app.EJEMPLOS))
def test_los_ejemplos_traen_todas_las_columnas(nombre: str) -> None:
    """Cada ejemplo precargado tiene las 13 variables que pide el modelo."""
    assert set(app.EJEMPLOS[nombre]) == set(COLS_REQUERIDAS)


@pytest.mark.parametrize("nombre", list(app.EJEMPLOS))
def test_los_ejemplos_usan_categorias_validas(nombre: str) -> None:
    """Un ejemplo con una categoría inventada se convertiría en NaN sin avisar."""
    ejemplo = app.EJEMPLOS[nombre]
    for columna, validas in CATEGORIAS_VALIDAS.items():
        if columna in ejemplo:
            assert str(ejemplo[columna]) in validas


@pytest.mark.parametrize("nombre", list(app.EJEMPLOS))
def test_los_ejemplos_caen_dentro_de_los_rangos(nombre: str) -> None:
    """Los valores precargados respetan los rangos de plausibilidad clínica."""
    ejemplo = app.EJEMPLOS[nombre]
    for columna, (minimo, maximo) in RANGOS_VALIDOS.items():
        if columna in ejemplo:
            assert minimo <= float(ejemplo[columna]) <= maximo


# --------------------------------------------------------------------------- #
# Predicción desde el formulario
# --------------------------------------------------------------------------- #


def test_predecir_paciente_devuelve_las_cuatro_salidas(modelo_dummy_en_disco: Path) -> None:
    """Del formulario sale probabilidad, decisión, diagnóstico y banda de riesgo."""
    modelo = cargar_modelo(modelo_dummy_en_disco)
    resultado = app.predecir_paciente(modelo, app.EJEMPLOS["Caso intermedio"], modelo.umbral)

    assert set(resultado.index) == {
        "probabilidad_enfermedad",
        "prediccion",
        "diagnostico",
        "nivel_riesgo",
    }
    assert 0.0 <= float(resultado["probabilidad_enfermedad"]) <= 1.0


def test_predecir_paciente_usa_el_umbral_indicado(modelo_dummy_en_disco: Path) -> None:
    """La decisión cambia exactamente al cruzar la probabilidad estimada.

    Los umbrales se calculan a partir de la propia probabilidad en lugar de fijar
    0.01 y 0.99: un `DummyClassifier` puede devolver 0 o 1 exactos, y con valores
    fijos la prueba pasaría o fallaría por cómo es el dummy, no por la lógica.
    """
    modelo = cargar_modelo(modelo_dummy_en_disco)
    datos = app.EJEMPLOS["Perfil de alto riesgo"]
    probabilidad = float(
        app.predecir_paciente(modelo, datos, modelo.umbral)["probabilidad_enfermedad"]
    )

    assert app.predecir_paciente(modelo, datos, probabilidad - 0.01)["prediccion"] == 1
    assert app.predecir_paciente(modelo, datos, probabilidad + 0.01)["prediccion"] == 0


def test_predecir_paciente_pasa_por_el_inference_pipeline(
    modelo_dummy_en_disco: Path,
) -> None:
    """La app no transforma por su cuenta: el resultado es el del pipeline.

    Se compara con el camino explícito —transformar, alinear, predecir— para que la
    prueba falle si alguien mete un atajo en la app.
    """
    modelo = cargar_modelo(modelo_dummy_en_disco)
    datos = app.EJEMPLOS["Perfil de bajo riesgo"]

    crudo = pd.DataFrame([{col: str(datos[col]) for col in COLS_REQUERIDAS}])
    esperado = predecir(modelo, alinear_columnas(transformar(crudo), modelo), umbral=modelo.umbral)

    obtenido = app.predecir_paciente(modelo, datos, modelo.umbral)
    assert float(obtenido["probabilidad_enfermedad"]) == float(
        esperado.iloc[0]["probabilidad_enfermedad"]
    )


# --------------------------------------------------------------------------- #
# Señales clínicas
# --------------------------------------------------------------------------- #


def test_senales_detecta_el_perfil_de_alto_riesgo() -> None:
    """El caso de alto riesgo tiene las cuatro señales presentes."""
    senales = app.senales_de_riesgo(app.EJEMPLOS["Perfil de alto riesgo"])
    assert len(senales) == N_SENALES
    assert all(presente for _, presente in senales)


def test_senales_no_dispara_en_el_perfil_de_bajo_riesgo() -> None:
    """El caso de bajo riesgo no tiene ninguna."""
    senales = app.senales_de_riesgo(app.EJEMPLOS["Perfil de bajo riesgo"])
    assert not any(presente for _, presente in senales)


def test_bandas_de_riesgo_tienen_color() -> None:
    """Toda banda que devuelva el pipeline tiene un color asignado en la app."""
    assert {etiqueta for _, etiqueta in BANDAS_RIESGO} == set(app.COLORES_RIESGO)
