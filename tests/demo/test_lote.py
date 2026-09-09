"""Pruebas del procesamiento por lotes de la demo.

Todo va con **modelo dummy y datos sintéticos**, como en el inference pipeline: la
suite no depende de que exista un modelo entrenado en el repositorio ni de que haya un
archivo de pacientes en `data/`.

Lo que se prueba aquí es lo que decide: qué archivos se aceptan, qué pasa cuando falta
una columna, que ninguna fila se pierde y que el lote da exactamente lo mismo que
predecir paciente a paciente. Los widgets de Streamlit no se prueban —para eso está la
evidencia del notebook 16—.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from demo import app, lote
from pipelines.inference_pipeline.inference_pipeline import (
    COLS_REQUERIDAS,
    ModeloCargado,
    cargar_modelo,
    transformar,
)

N_FILAS_SINTETICAS = 12
N_COLS_PREDICCION = 4


def _fila(ejemplo: dict[str, Any]) -> dict[str, str]:
    """Un ejemplo de la app convertido en la fila de texto que traería un CSV."""
    return {col: str(ejemplo[col]) for col in COLS_REQUERIDAS}


@pytest.fixture(scope="module")
def crudo_sintetico() -> pd.DataFrame:
    """Doce pacientes construidos rotando los tres perfiles de ejemplo de la app."""
    ejemplos = list(app.EJEMPLOS.values())
    filas = [_fila(ejemplos[i % len(ejemplos)]) for i in range(N_FILAS_SINTETICAS)]
    datos = pd.DataFrame(filas)
    datos.insert(0, "id_paciente", [f"P-{i:03d}" for i in range(1, len(datos) + 1)])
    return datos


@pytest.fixture(scope="module")
def modelo(
    tmp_path_factory: pytest.TempPathFactory, crudo_sintetico: pd.DataFrame
) -> ModeloCargado:
    """Artefacto dummy con la misma forma que el real, cargado desde disco."""
    features = transformar(crudo_sintetico[COLS_REQUERIDAS])

    pipeline = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
            ("modelo", DummyClassifier(strategy="stratified", random_state=0)),
        ]
    )
    pipeline.fit(features, [i % 2 for i in range(len(features))])

    ruta: Path = tmp_path_factory.mktemp("modelos") / "modelo_completo.joblib"
    joblib.dump({"pipeline": pipeline, "umbral_optimo": 0.31, "modelo": "dummy"}, ruta)
    return cargar_modelo(ruta)


# --------------------------------------------------------------------------- #
# Lectura del archivo subido
# --------------------------------------------------------------------------- #


def test_lee_un_csv(crudo_sintetico: pd.DataFrame) -> None:
    """El caso normal: un CSV con las 13 columnas y una fila por paciente."""
    datos = lote.leer_archivo("pacientes.csv", crudo_sintetico.to_csv(index=False).encode())
    assert len(datos) == N_FILAS_SINTETICAS
    assert set(COLS_REQUERIDAS) <= set(datos.columns)


def test_lee_un_parquet(crudo_sintetico: pd.DataFrame) -> None:
    """Parquet se acepta igual que CSV: es el formato que produce el feature pipeline."""
    buffer = io.BytesIO()
    crudo_sintetico.to_parquet(buffer, index=False)
    datos = lote.leer_archivo("pacientes.parquet", buffer.getvalue())
    assert len(datos) == N_FILAS_SINTETICAS


def test_lee_un_excel(crudo_sintetico: pd.DataFrame) -> None:
    """Excel es el formato en el que llegan de verdad las tablas clínicas.

    Se salta si falta `openpyxl`: es un lector opcional —el CSV y el Parquet no lo
    necesitan— y la suite no debería fallar en rojo por una dependencia ausente
    cuando lo que hay que hacer es instalarla.
    """
    pytest.importorskip("openpyxl", reason="lector de .xlsx opcional; instálalo con `uv sync`")

    buffer = io.BytesIO()
    crudo_sintetico.to_excel(buffer, index=False)
    datos = lote.leer_archivo("pacientes.xlsx", buffer.getvalue())
    assert len(datos) == N_FILAS_SINTETICAS


def test_todo_se_lee_como_texto(crudo_sintetico: pd.DataFrame) -> None:
    """Nada de inferencia de tipos: decide el saneamiento del feature pipeline."""
    datos = lote.leer_archivo("pacientes.csv", crudo_sintetico.to_csv(index=False).encode())
    assert not any(pd.api.types.is_numeric_dtype(datos[col]) for col in COLS_REQUERIDAS)


def test_rechaza_un_formato_desconocido(crudo_sintetico: pd.DataFrame) -> None:
    """Un `.json` no se intenta adivinar: se dice qué formatos hay."""
    with pytest.raises(lote.ErrorDeLote, match="Formato no soportado"):
        lote.leer_archivo("pacientes.json", b"{}")


def test_rechaza_un_archivo_corrupto() -> None:
    """Un archivo con la extensión correcta y el contenido roto da mensaje, no traza."""
    with pytest.raises(lote.ErrorDeLote, match="No se pudo leer"):
        lote.leer_archivo("pacientes.xlsx", b"esto no es un xlsx")


def test_rechaza_un_archivo_vacio() -> None:
    """Un archivo con cabecera y sin filas no produce un lote de cero predicciones."""
    cabecera = (",".join(COLS_REQUERIDAS) + "\n").encode()
    with pytest.raises(lote.ErrorDeLote, match="ninguna fila"):
        lote.leer_archivo("vacio.csv", cabecera)


def test_rechaza_un_archivo_demasiado_grande(monkeypatch: pytest.MonkeyPatch) -> None:
    """Por encima del tope de la interfaz se remite al pipeline por línea de comandos."""
    monkeypatch.setattr(lote, "MAX_FILAS", 3)
    filas = pd.DataFrame([_fila(app.EJEMPLOS["Caso intermedio"])] * 5)
    with pytest.raises(lote.ErrorDeLote, match="máximo de la interfaz"):
        lote.leer_archivo("grande.csv", filas.to_csv(index=False).encode())


# --------------------------------------------------------------------------- #
# Validación de columnas
# --------------------------------------------------------------------------- #


def test_falla_si_falta_una_columna_obligatoria(crudo_sintetico: pd.DataFrame) -> None:
    """Se falla en lugar de imputar: una columna ausente daría una probabilidad falsa."""
    incompleto = crudo_sintetico.drop(columns=["thal", "ca"])
    with pytest.raises(lote.ErrorDeLote, match="thal"):
        lote.validar_columnas(incompleto)


def test_las_columnas_de_mas_se_reportan_pero_no_estorban(crudo_sintetico: pd.DataFrame) -> None:
    """Se devuelven para poder avisar de que no entran al modelo."""
    con_extra = crudo_sintetico.assign(comentario="revisión anual")
    ignoradas = lote.validar_columnas(con_extra)
    assert "comentario" in ignoradas
    assert "id_paciente" in ignoradas


# --------------------------------------------------------------------------- #
# Identificadores
# --------------------------------------------------------------------------- #


def test_usa_la_columna_identificadora_del_archivo(crudo_sintetico: pd.DataFrame) -> None:
    """Si el archivo trae `id_paciente`, es lo que etiqueta cada resultado."""
    ids = lote.identificadores(crudo_sintetico)
    assert list(ids) == list(crudo_sintetico["id_paciente"])


def test_numera_las_filas_si_no_hay_identificador(crudo_sintetico: pd.DataFrame) -> None:
    """Sin columna de id, la etiqueta es el número de fila empezando en 1."""
    ids = lote.identificadores(crudo_sintetico.drop(columns=["id_paciente"]))
    assert list(ids) == list(range(1, N_FILAS_SINTETICAS + 1))


# --------------------------------------------------------------------------- #
# Predicción del lote
# --------------------------------------------------------------------------- #


def test_predecir_lote_devuelve_una_fila_por_paciente(
    modelo: ModeloCargado, crudo_sintetico: pd.DataFrame
) -> None:
    """Ninguna fila se pierde por el camino: entran doce, salen doce."""
    resultado = lote.predecir_lote(modelo, crudo_sintetico, umbral=0.5)
    assert len(resultado) == len(crudo_sintetico)


def test_predecir_lote_conserva_las_columnas_originales(
    modelo: ModeloCargado, crudo_sintetico: pd.DataFrame
) -> None:
    """La salida es auditable sola: trae la entrada completa más las 4 predicciones."""
    con_extra = crudo_sintetico.assign(comentario="revisión anual")
    resultado = lote.predecir_lote(modelo, con_extra, umbral=0.5)

    assert set(con_extra.columns) <= set(resultado.columns)
    assert set(lote.COLS_PREDICCION) <= set(resultado.columns)
    assert "comentario" in resultado.columns


def test_predecir_lote_no_duplica_la_columna_de_id(
    modelo: ModeloCargado, crudo_sintetico: pd.DataFrame
) -> None:
    """El identificador se añade una sola vez aunque el archivo ya lo traiga."""
    resultado = lote.predecir_lote(modelo, crudo_sintetico, umbral=0.5)
    assert list(resultado.columns).count(lote.COL_ID) == 1


def test_el_lote_coincide_con_la_prediccion_individual(
    modelo: ModeloCargado, crudo_sintetico: pd.DataFrame
) -> None:
    """Un paciente da lo mismo suelto que dentro del lote.

    Es la prueba que fallaría si el lote tomara un atajo —normalizar por bloque,
    reordenar, deduplicar— en lugar de pasar por el mismo `inference_pipeline` que la
    demo individual.
    """
    resultado = lote.predecir_lote(modelo, crudo_sintetico, umbral=modelo.umbral)

    for posicion in (0, 5, len(crudo_sintetico) - 1):
        datos = crudo_sintetico.iloc[posicion].to_dict()
        individual = app.predecir_paciente(modelo, datos, modelo.umbral)
        assert float(resultado.iloc[posicion]["probabilidad_enfermedad"]) == pytest.approx(
            float(individual["probabilidad_enfermedad"])
        )


def test_el_umbral_solo_mueve_la_decision(
    modelo: ModeloCargado, crudo_sintetico: pd.DataFrame
) -> None:
    """Cambiar el umbral reclasifica, no vuelve a predecir: las probabilidades no cambian."""
    bajo = lote.predecir_lote(modelo, crudo_sintetico, umbral=0.05)
    alto = lote.predecir_lote(modelo, crudo_sintetico, umbral=0.95)

    assert list(bajo["probabilidad_enfermedad"]) == list(alto["probabilidad_enfermedad"])
    assert int(bajo["prediccion"].sum()) >= int(alto["prediccion"].sum())


def test_predecir_lote_falla_si_faltan_columnas(
    modelo: ModeloCargado, crudo_sintetico: pd.DataFrame
) -> None:
    """No hay lote a medias: si el archivo no sirve, no sale ninguna predicción."""
    with pytest.raises(lote.ErrorDeLote):
        lote.predecir_lote(modelo, crudo_sintetico.drop(columns=["age"]), umbral=0.5)


def test_predecir_lote_tolera_valores_ausentes(
    modelo: ModeloCargado, crudo_sintetico: pd.DataFrame
) -> None:
    """Una celda vacía se imputa dentro del pipeline; no tumba el lote entero."""
    con_huecos = crudo_sintetico.copy()
    con_huecos.loc[0, "chol"] = None
    con_huecos.loc[1, "thal"] = None

    resultado = lote.predecir_lote(modelo, con_huecos, umbral=0.5)
    assert len(resultado) == len(con_huecos)
    assert resultado["probabilidad_enfermedad"].notna().all()


# --------------------------------------------------------------------------- #
# Resumen y serialización
# --------------------------------------------------------------------------- #


def test_el_resumen_cuadra_con_la_tabla(
    modelo: ModeloCargado, crudo_sintetico: pd.DataFrame
) -> None:
    """Las cifras agregadas son las de la tabla, no un cálculo aparte."""
    predicciones = lote.predecir_lote(modelo, crudo_sintetico, umbral=0.5)
    resumen = lote.resumen_lote(predicciones, umbral=0.5)

    assert resumen["filas"] == len(predicciones)
    assert resumen["casos_probables"] == int(predicciones["prediccion"].sum())
    assert resumen["casos_probables"] + resumen["casos_descartados"] == resumen["filas"]
    assert sum(resumen["por_banda_de_riesgo"].values()) == resumen["filas"]


def test_el_csv_se_puede_volver_a_leer(
    modelo: ModeloCargado, crudo_sintetico: pd.DataFrame
) -> None:
    """Lo que se descarga vuelve a entrar: mismo número de filas y de columnas."""
    predicciones = lote.predecir_lote(modelo, crudo_sintetico, umbral=0.5)
    releido = pd.read_csv(io.BytesIO(lote.a_csv(predicciones)))

    assert len(releido) == len(predicciones)
    assert list(releido.columns) == list(predicciones.columns)


def test_la_plantilla_sirve_como_entrada(modelo: ModeloCargado) -> None:
    """La plantilla que ofrece la interfaz pasa la validación y se puede predecir.

    Sería una trampa fácil: publicar una plantilla que al subirla dé error.
    """
    datos = lote.leer_archivo("plantilla.csv", lote.a_csv(lote.plantilla()))
    resultado = lote.predecir_lote(modelo, datos, umbral=0.5)

    assert len(resultado) == 1
    assert len(lote.COLS_PREDICCION) == N_COLS_PREDICCION
