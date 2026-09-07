"""Pruebas unitarias del inference pipeline del proyecto Heart_project.

Todas las pruebas usan un **modelo dummy** y **datos sintéticos**: no dependen de
que exista un modelo entrenado en `data/06_models/` ni de los datos reales del
proyecto. Así la suite corre en un clon recién descargado y en CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipelines.feature_pipeline.feature_pipeline import OBJETIVO, ErrorDeValidacion
from pipelines.inference_pipeline.inference_pipeline import (
    BANDAS_RIESGO,
    COLS_REQUERIDAS,
    UMBRAL_POR_DEFECTO,
    ErrorDeInferencia,
    ModeloCargado,
    RutasInferencia,
    alinear_columnas,
    cargar_modelo,
    clasificar_riesgo,
    ejecutar_pipeline,
    esquema_features_inferencia,
    graficar_predicciones,
    guardar_predicciones,
    guardar_resumen,
    leer_datos_nuevos,
    main,
    predecir,
    resolver_ruta_modelo,
    transformar,
)

#: Valores esperados en las aserciones (evita "magic values" en el linter).
N_PACIENTES = 30
N_ATRIBUTOS = 26
UMBRAL_CALIBRADO = 0.31


# --------------------------------------------------------------------------- #
# Datos sintéticos y modelo dummy
# --------------------------------------------------------------------------- #


def generar_pacientes(n_filas: int = N_PACIENTES, semilla: int = 3) -> pd.DataFrame:
    """Genera un lote de pacientes nuevos sintéticos, sin variable objetivo.

    Todo se genera como texto, igual que llega un CSV real: la conversión de tipos
    es responsabilidad del saneamiento, no de quien produce el archivo.
    """
    rng = np.random.default_rng(semilla)
    filas = [
        {
            "age": str(int(rng.integers(35, 75))),
            "sex": "male" if rng.random() < 0.6 else "female",  # noqa: PLR2004
            "chest_pain": str(rng.choice(["asymptomatic", "nonanginal", "nontypical", "typical"])),
            "rest_bp": str(int(rng.integers(100, 180))),
            "chol": str(int(rng.integers(150, 350))),
            "fbs": str(int(rng.integers(0, 2))),
            "rest_ecg": str(rng.choice(["normal", "left ventricular hypertrophy"])),
            "max_hr": str(int(rng.integers(90, 195))),
            "exang": str(int(rng.integers(0, 2))),
            "old_peak": f"{rng.uniform(0, 4):.1f}",
            "slope": str(int(rng.integers(1, 4))),
            "ca": f"{float(rng.integers(0, 4)):.1f}",
            "thal": str(rng.choice(["normal", "fixed", "reversable"])),
        }
        for _ in range(n_filas)
    ]
    return pd.DataFrame(filas, columns=COLS_REQUERIDAS)


def entrenar_dummy(atributos: pd.DataFrame) -> Pipeline:
    """Construye un modelo dummy con la misma forma que el modelo real.

    Es un `Pipeline` con los mismos pasos previos (imputación y escalado) y un
    `DummyClassifier` en lugar del estimador de verdad. Interesa que **se comporte**
    como el modelo real —que exponga `predict_proba` y `feature_names_in_`— no que
    prediga bien: lo que se está probando es la mecánica de la inferencia.
    """
    modelo = Pipeline(
        [
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
            ("modelo", DummyClassifier(strategy="stratified", random_state=0)),
        ]
    )
    etiquetas = np.tile([0, 1], len(atributos) // 2 + 1)[: len(atributos)]
    modelo.fit(atributos, etiquetas)
    return modelo


@pytest.fixture(scope="module")
def pacientes() -> pd.DataFrame:
    """Lote de pacientes nuevos sintéticos."""
    return generar_pacientes()


@pytest.fixture(scope="module")
def features(pacientes: pd.DataFrame) -> pd.DataFrame:
    """Tabla de features derivada de los pacientes sintéticos."""
    return transformar(pacientes)


@pytest.fixture(scope="module")
def modelo_dummy(features: pd.DataFrame) -> Pipeline:
    """Modelo dummy ajustado sobre los features sintéticos."""
    return entrenar_dummy(features)


@pytest.fixture
def ruta_modelo(tmp_path: Path, modelo_dummy: Pipeline) -> Path:
    """Artefacto completo en disco, como el que produce el training pipeline."""
    ruta = tmp_path / "modelo_completo.joblib"
    joblib.dump(
        {
            "pipeline": modelo_dummy,
            "umbral_optimo": UMBRAL_CALIBRADO,
            "modelo": "dummy",
            "generado_en": "2026-09-07T00:00:00+00:00",
        },
        ruta,
    )
    return ruta


@pytest.fixture
def ruta_entrada(tmp_path: Path, pacientes: pd.DataFrame) -> Path:
    """CSV de pacientes nuevos en disco."""
    ruta = tmp_path / "pacientes_nuevos.csv"
    pacientes.to_csv(ruta, index=False)
    return ruta


@pytest.fixture
def rutas_salida(tmp_path: Path) -> RutasInferencia:
    """Destinos temporales para las salidas de la inferencia."""
    return RutasInferencia(
        predicciones=tmp_path / "predicciones.csv",
        resumen=tmp_path / "resumen.json",
        figura=tmp_path / "figuras" / "distribucion.png",
    )


# --------------------------------------------------------------------------- #
# Carga del modelo
# --------------------------------------------------------------------------- #


def test_carga_el_artefacto_completo_con_su_umbral(ruta_modelo: Path) -> None:
    """Del artefacto se recupera el pipeline y el umbral con el que se calibró."""
    modelo = cargar_modelo(ruta_modelo)
    assert isinstance(modelo, ModeloCargado)
    assert modelo.umbral == UMBRAL_CALIBRADO
    assert modelo.metadatos["modelo"] == "dummy"
    assert modelo.origen == ruta_modelo


def test_carga_un_pipeline_suelto_con_umbral_por_defecto(
    tmp_path: Path, modelo_dummy: Pipeline
) -> None:
    """Un `.joblib` sin metadatos también sirve, asumiendo el umbral por defecto."""
    ruta = tmp_path / "modelo_simple.joblib"
    joblib.dump(modelo_dummy, ruta)

    modelo = cargar_modelo(ruta)
    assert modelo.umbral == UMBRAL_POR_DEFECTO
    assert modelo.metadatos == {}


def test_carga_expone_los_atributos_del_entrenamiento(ruta_modelo: Path) -> None:
    """El modelo declara qué columnas vio, y en qué orden."""
    modelo = cargar_modelo(ruta_modelo)
    assert len(modelo.atributos_esperados) == N_ATRIBUTOS
    assert modelo.atributos_esperados[0] == "age"


def test_carga_falla_si_no_existe(tmp_path: Path) -> None:
    """Sin modelo el mensaje indica qué hay que ejecutar antes."""
    with pytest.raises(FileNotFoundError, match="training pipeline"):
        cargar_modelo(tmp_path / "no_existe.joblib")


def test_carga_rechaza_un_objeto_que_no_es_modelo(tmp_path: Path) -> None:
    """Un joblib con cualquier otra cosa dentro se detecta antes de predecir."""
    ruta = tmp_path / "basura.joblib"
    joblib.dump({"pipeline": "esto no es un modelo"}, ruta)

    with pytest.raises(ErrorDeInferencia, match="predict_proba"):
        cargar_modelo(ruta)


def test_resolver_ruta_prefiere_el_artefacto_completo(
    tmp_path: Path, modelo_dummy: Pipeline
) -> None:
    """Con ambos disponibles se elige el que trae el umbral calibrado."""
    completo = tmp_path / "completo.joblib"
    simple = tmp_path / "simple.joblib"
    joblib.dump({"pipeline": modelo_dummy}, completo)
    joblib.dump(modelo_dummy, simple)

    assert resolver_ruta_modelo(completo, simple) == completo
    assert resolver_ruta_modelo(tmp_path / "ausente.joblib", simple) == simple


# --------------------------------------------------------------------------- #
# Lectura de los datos nuevos
# --------------------------------------------------------------------------- #


def test_lee_los_datos_como_texto(ruta_entrada: Path, pacientes: pd.DataFrame) -> None:
    """El CSV se lee sin inferir tipos, para no perder los valores corruptos."""
    leidos = leer_datos_nuevos(ruta_entrada)
    assert len(leidos) == len(pacientes)
    assert list(leidos.columns) == COLS_REQUERIDAS
    assert not any(pd.api.types.is_numeric_dtype(leidos[c]) for c in leidos.columns)


def test_lectura_falla_si_no_existe(tmp_path: Path) -> None:
    """Un archivo inexistente se reporta con su ruta."""
    with pytest.raises(FileNotFoundError, match="datos nuevos"):
        leer_datos_nuevos(tmp_path / "no_existe.csv")


def test_lectura_falla_si_el_archivo_esta_vacio(tmp_path: Path) -> None:
    """Un CSV con cabecera pero sin filas no da nada que predecir."""
    ruta = tmp_path / "vacio.csv"
    pd.DataFrame(columns=COLS_REQUERIDAS).to_csv(ruta, index=False)

    with pytest.raises(ErrorDeInferencia, match="ninguna fila"):
        leer_datos_nuevos(ruta)


def test_lectura_falla_si_faltan_columnas(tmp_path: Path, pacientes: pd.DataFrame) -> None:
    """Sin todas las columnas de entrada no se puede construir el mismo esquema."""
    ruta = tmp_path / "incompleto.csv"
    pacientes.drop(columns=["thal", "ca"]).to_csv(ruta, index=False)

    with pytest.raises(ErrorDeInferencia, match="faltan columnas"):
        leer_datos_nuevos(ruta)


def test_lectura_ignora_la_variable_objetivo(tmp_path: Path, pacientes: pd.DataFrame) -> None:
    """Si el archivo trae etiquetas, se descartan: la inferencia no las usa."""
    ruta = tmp_path / "con_etiqueta.csv"
    pacientes.assign(**{OBJETIVO: "1"}).to_csv(ruta, index=False)

    leidos = leer_datos_nuevos(ruta)
    assert OBJETIVO not in leidos.columns
    assert len(leidos) == len(pacientes)


# --------------------------------------------------------------------------- #
# Transformación: la misma del entrenamiento
# --------------------------------------------------------------------------- #


def test_transformar_no_elimina_ninguna_fila(pacientes: pd.DataFrame) -> None:
    """Cada paciente que entra tiene que salir: no hay deduplicación en inferencia."""
    con_duplicados = pd.concat([pacientes, pacientes.head(3)], ignore_index=True)
    features = transformar(con_duplicados)
    assert len(features) == len(con_duplicados)


def test_transformar_conserva_el_orden(pacientes: pd.DataFrame) -> None:
    """La fila i de la salida corresponde a la fila i de la entrada."""
    features = transformar(pacientes)
    edades_esperadas = pd.to_numeric(pacientes["age"]).astype(float)
    np.testing.assert_allclose(features["age"].to_numpy(), edades_esperadas.to_numpy())


def test_transformar_genera_las_mismas_columnas_que_el_entrenamiento(
    features: pd.DataFrame,
) -> None:
    """El esquema de inferencia es el del entrenamiento menos la variable objetivo."""
    assert features.shape[1] == N_ATRIBUTOS
    assert OBJETIVO not in features.columns
    assert set(esquema_features_inferencia().columns) == set(features.columns)


def test_transformar_sanea_los_valores_corruptos(pacientes: pd.DataFrame) -> None:
    """Un valor no interpretable se convierte en NaN, no rompe la inferencia."""
    sucio = pacientes.copy()
    sucio.loc[0, "age"] = "no_es_una_edad"
    sucio.loc[0, "thal"] = "categoria_inventada"

    features = transformar(sucio)
    assert pd.isna(features.loc[0, "age"])
    assert features.loc[0, [c for c in features.columns if c.startswith("thal_")]].isna().all()


def test_transformar_conserva_los_faltantes(pacientes: pd.DataFrame) -> None:
    """La imputación es del modelo, no de la transformación."""
    con_huecos = pacientes.copy()
    con_huecos.loc[0, "chol"] = None

    features = transformar(con_huecos)
    assert pd.isna(features.loc[0, "chol"])


# --------------------------------------------------------------------------- #
# Alineación con el modelo
# --------------------------------------------------------------------------- #


def test_alinear_reordena_las_columnas(ruta_modelo: Path, features: pd.DataFrame) -> None:
    """Aunque lleguen desordenadas, se devuelven en el orden del entrenamiento."""
    modelo = cargar_modelo(ruta_modelo)
    desordenadas = features[list(reversed(features.columns))]

    alineadas = alinear_columnas(desordenadas, modelo)
    assert list(alineadas.columns) == modelo.atributos_esperados


def test_alinear_detecta_columnas_faltantes(ruta_modelo: Path, features: pd.DataFrame) -> None:
    """Si falta un atributo, se aborta en vez de predecir con datos incompletos."""
    modelo = cargar_modelo(ruta_modelo)
    with pytest.raises(ErrorDeInferencia, match="faltan"):
        alinear_columnas(features.drop(columns=["age"]), modelo)


def test_alinear_detecta_columnas_sobrantes(ruta_modelo: Path, features: pd.DataFrame) -> None:
    """Una columna extra también invalida la correspondencia con el modelo."""
    modelo = cargar_modelo(ruta_modelo)
    with pytest.raises(ErrorDeInferencia, match="sobran"):
        alinear_columnas(features.assign(columna_nueva=1.0), modelo)


# --------------------------------------------------------------------------- #
# Predicción
# --------------------------------------------------------------------------- #


def test_predecir_devuelve_una_fila_por_paciente(ruta_modelo: Path, features: pd.DataFrame) -> None:
    """La salida tiene probabilidad, decisión y lectura para cada paciente."""
    modelo = cargar_modelo(ruta_modelo)
    predicciones = predecir(modelo, alinear_columnas(features, modelo))

    assert len(predicciones) == len(features)
    assert list(predicciones.columns) == [
        "probabilidad_enfermedad",
        "prediccion",
        "diagnostico",
        "nivel_riesgo",
    ]


def test_probabilidades_en_rango_valido(ruta_modelo: Path, features: pd.DataFrame) -> None:
    """Toda probabilidad cae en [0, 1] y ninguna es NaN."""
    modelo = cargar_modelo(ruta_modelo)
    predicciones = predecir(modelo, alinear_columnas(features, modelo))
    probabilidades = predicciones["probabilidad_enfermedad"]

    assert probabilidades.notna().all()
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()


def test_prediccion_respeta_el_umbral(ruta_modelo: Path, features: pd.DataFrame) -> None:
    """La decisión es exactamente `probabilidad >= umbral`."""
    modelo = cargar_modelo(ruta_modelo)
    predicciones = predecir(modelo, alinear_columnas(features, modelo), umbral=0.5)

    esperada = (predicciones["probabilidad_enfermedad"] >= 0.5).astype(int)  # noqa: PLR2004
    pd.testing.assert_series_equal(
        predicciones["prediccion"], esperada, check_names=False, check_dtype=False
    )


def test_umbral_mas_bajo_no_reduce_los_positivos(ruta_modelo: Path, features: pd.DataFrame) -> None:
    """Bajar el umbral sólo puede añadir predicciones positivas, nunca quitarlas."""
    modelo = cargar_modelo(ruta_modelo)
    alineadas = alinear_columnas(features, modelo)

    estricto = predecir(modelo, alineadas, umbral=0.8)["prediccion"].sum()
    permisivo = predecir(modelo, alineadas, umbral=0.2)["prediccion"].sum()
    assert permisivo >= estricto


def test_diagnostico_coincide_con_la_prediccion(ruta_modelo: Path, features: pd.DataFrame) -> None:
    """La etiqueta legible no puede contradecir al 0/1."""
    modelo = cargar_modelo(ruta_modelo)
    predicciones = predecir(modelo, alinear_columnas(features, modelo))

    enfermos = predicciones["prediccion"] == 1
    assert (predicciones.loc[enfermos, "diagnostico"] == "enfermo").all()
    assert (predicciones.loc[~enfermos, "diagnostico"] == "sano").all()


@pytest.mark.parametrize(
    ("probabilidad", "esperado"),
    [(0.05, "muy bajo"), (0.3, "bajo"), (0.5, "medio"), (0.7, "alto"), (0.95, "muy alto")],
)
def test_bandas_de_riesgo(probabilidad: float, esperado: str) -> None:
    """Cada probabilidad cae en la banda que le corresponde."""
    assert clasificar_riesgo(probabilidad) == esperado


def test_las_bandas_cubren_todo_el_rango() -> None:
    """Ninguna probabilidad puede quedarse sin banda."""
    for probabilidad in np.linspace(0, 1, 101):
        assert clasificar_riesgo(float(probabilidad)) in {e for _, e in BANDAS_RIESGO}


# --------------------------------------------------------------------------- #
# Almacenamiento y visualización
# --------------------------------------------------------------------------- #


def test_guardar_predicciones_adjunta_los_datos_del_paciente(
    tmp_path: Path, ruta_modelo: Path, pacientes: pd.DataFrame, features: pd.DataFrame
) -> None:
    """El CSV es auditable solo: trae los datos de entrada junto al resultado."""
    modelo = cargar_modelo(ruta_modelo)
    predicciones = predecir(modelo, alinear_columnas(features, modelo))
    ruta = tmp_path / "salida" / "predicciones.csv"

    guardar_predicciones(pacientes, predicciones, ruta)
    releido = pd.read_csv(ruta)

    assert len(releido) == len(pacientes)
    assert {"age", "thal", "probabilidad_enfermedad", "diagnostico"} <= set(releido.columns)


def test_guardar_resumen_escribe_json(tmp_path: Path) -> None:
    """El manifiesto queda en JSON legible."""
    ruta = tmp_path / "resumen.json"
    guardar_resumen({"n_pacientes": 10, "umbral": 0.31}, ruta)

    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    assert contenido["n_pacientes"] == 10  # noqa: PLR2004


def test_graficar_genera_la_figura(
    tmp_path: Path, ruta_modelo: Path, features: pd.DataFrame
) -> None:
    """La visualización de las predicciones se crea y no está vacía."""
    modelo = cargar_modelo(ruta_modelo)
    predicciones = predecir(modelo, alinear_columnas(features, modelo))
    ruta = tmp_path / "figuras" / "distribucion.png"

    graficar_predicciones(predicciones, modelo.umbral, ruta)
    assert ruta.is_file()
    assert ruta.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Ejecución de extremo a extremo
# --------------------------------------------------------------------------- #


def test_pipeline_completo_genera_las_tres_salidas(
    ruta_modelo: Path, ruta_entrada: Path, rutas_salida: RutasInferencia
) -> None:
    """La ejecución deja predicciones, resumen y figura."""
    resumen = ejecutar_pipeline(ruta_modelo, ruta_entrada, rutas_salida)

    assert rutas_salida.predicciones.is_file()
    assert rutas_salida.resumen.is_file()
    assert rutas_salida.figura.is_file()
    assert resumen["n_pacientes"] == N_PACIENTES


def test_resumen_documenta_la_inferencia(
    ruta_modelo: Path, ruta_entrada: Path, rutas_salida: RutasInferencia
) -> None:
    """El resumen dice con qué modelo y con qué umbral se predijo."""
    ejecutar_pipeline(ruta_modelo, ruta_entrada, rutas_salida)
    resumen = json.loads(rutas_salida.resumen.read_text(encoding="utf-8"))

    assert resumen["umbral"] == UMBRAL_CALIBRADO
    assert resumen["tipo_modelo"] == "dummy"
    assert resumen["n_atributos"] == N_ATRIBUTOS
    assert resumen["n_predichos_enfermos"] + resumen["n_predichos_sanos"] == N_PACIENTES


def test_pipeline_usa_el_umbral_indicado(
    ruta_modelo: Path, ruta_entrada: Path, rutas_salida: RutasInferencia
) -> None:
    """`--umbral` tiene prioridad sobre el calibrado del modelo."""
    resumen = ejecutar_pipeline(ruta_modelo, ruta_entrada, rutas_salida, umbral=0.9)
    assert resumen["umbral"] == 0.9  # noqa: PLR2004


def test_main_devuelve_cero_en_ejecucion_correcta(
    ruta_modelo: Path, ruta_entrada: Path, rutas_salida: RutasInferencia
) -> None:
    """El script termina con código 0 y deja el CSV de predicciones."""
    codigo = main(
        [
            "--modelo",
            str(ruta_modelo),
            "--entrada",
            str(ruta_entrada),
            "--predicciones",
            str(rutas_salida.predicciones),
            "--resumen",
            str(rutas_salida.resumen),
            "--figura",
            str(rutas_salida.figura),
        ]
    )
    assert codigo == 0
    assert rutas_salida.predicciones.is_file()


def test_main_devuelve_uno_sin_modelo(
    tmp_path: Path, ruta_entrada: Path, rutas_salida: RutasInferencia
) -> None:
    """Sin modelo entrenado no se genera ninguna predicción."""
    codigo = main(
        [
            "--modelo",
            str(tmp_path / "no_existe.joblib"),
            "--entrada",
            str(ruta_entrada),
            "--predicciones",
            str(rutas_salida.predicciones),
            "--resumen",
            str(rutas_salida.resumen),
            "--figura",
            str(rutas_salida.figura),
        ]
    )
    assert codigo == 1
    assert not rutas_salida.predicciones.exists()


def test_main_devuelve_uno_con_datos_incompletos(
    tmp_path: Path, ruta_modelo: Path, pacientes: pd.DataFrame, rutas_salida: RutasInferencia
) -> None:
    """Un archivo al que le faltan columnas no produce predicciones a medias."""
    entrada = tmp_path / "incompleto.csv"
    pacientes.drop(columns=["chol"]).to_csv(entrada, index=False)

    codigo = main(
        [
            "--modelo",
            str(ruta_modelo),
            "--entrada",
            str(entrada),
            "--predicciones",
            str(rutas_salida.predicciones),
            "--resumen",
            str(rutas_salida.resumen),
            "--figura",
            str(rutas_salida.figura),
        ]
    )
    assert codigo == 1
    assert not rutas_salida.predicciones.exists()


def test_inferencia_es_reproducible(tmp_path: Path, ruta_modelo: Path, ruta_entrada: Path) -> None:
    """Dos ejecuciones sobre la misma entrada dan las mismas predicciones."""

    def ejecutar(sufijo: str) -> pd.DataFrame:
        rutas = RutasInferencia(
            predicciones=tmp_path / f"pred_{sufijo}.csv",
            resumen=tmp_path / f"resumen_{sufijo}.json",
            figura=tmp_path / f"figura_{sufijo}.png",
        )
        ejecutar_pipeline(ruta_modelo, ruta_entrada, rutas)
        salida: pd.DataFrame = pd.read_csv(rutas.predicciones)
        return salida

    pd.testing.assert_frame_equal(ejecutar("a"), ejecutar("b"))


def test_error_de_inferencia_hereda_de_error_de_validacion() -> None:
    """Así el manejador de `main` la trata igual: sin traza y sin predicciones."""
    assert issubclass(ErrorDeInferencia, ErrorDeValidacion)


def test_rutas_por_defecto_apuntan_a_las_capas_del_proyecto(tmp_path: Path) -> None:
    """Las salidas van a las capas de datos que corresponden."""
    rutas: Any = RutasInferencia.por_defecto(tmp_path)
    assert "07_model_output" in str(rutas.predicciones)
    assert "08_reporting" in str(rutas.resumen)
    assert rutas.figura.suffix == ".png"
