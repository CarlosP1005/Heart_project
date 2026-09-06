"""Pruebas unitarias del training pipeline del proyecto Heart_project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pipelines.feature_pipeline.feature_pipeline import ErrorDeValidacion, construir_features
from pipelines.training_pipeline.train_pipeline import (
    MODELO_POR_DEFECTO,
    MODELOS,
    PROPORCION_TEST,
    UMBRAL_DEFECTO,
    ErrorDeSeparacion,
    ResultadoSeparacion,
    RutasSalida,
    aplicar_resultado_separacion,
    construir_pipeline,
    construir_tabla_metricas,
    ejecutar_pipeline,
    entrenar,
    evaluar,
    evaluar_modelo_trivial,
    guardar_artefacto,
    guardar_checks_separacion,
    guardar_metricas,
    guardar_modelo,
    guardar_predicciones,
    leer_features,
    main,
    matriz_confusion,
    optimizar_umbral,
    separar_train_test,
    validar_cruzado,
    validar_separacion_train_test,
    validate_train_test_split,
)

#: (X_train, X_test, y_train, y_test). El hook de mypy corre en un entorno aislado
#: sin pandas, así que ve `DataFrame` como `Any`; el alias lo deja explícito y evita
#: el aviso `no-any-return` al devolver la tupla.
Conjuntos = tuple[Any, Any, Any, Any]

#: Valores esperados en las aserciones (evita "magic values" en el linter).
N_FILAS_SINTETICAS = 200
MIN_F1_ACEPTABLE = 0.6
TOLERANCIA_PROPORCION = 0.02
N_METRICAS = 7
N_COMPROBACIONES = 9


# --------------------------------------------------------------------------- #
# Datos de prueba
# --------------------------------------------------------------------------- #


def generar_crudo(n_filas: int = N_FILAS_SINTETICAS, semilla: int = 0) -> pd.DataFrame:
    """Genera datos crudos sintéticos con señal real y algo de ruido.

    Se construyen crudos y se pasan por el feature pipeline en lugar de fabricar la
    tabla de features a mano: así las pruebas usan exactamente el mismo contrato de
    datos que produce el paso anterior, y un cambio de esquema las rompe (que es lo
    que queremos que ocurra).
    """
    rng = np.random.default_rng(semilla)
    enfermo = rng.integers(0, 2, n_filas)

    filas = []
    for indice, etiqueta in enumerate(enfermo):
        # La señal: los enfermos tienen menos FC máxima y más depresión del ST.
        max_hr = int(rng.normal(140 if etiqueta else 165, 15))
        old_peak = float(np.clip(rng.normal(2.0 if etiqueta else 0.6, 0.6), 0, 6))
        filas.append(
            {
                "age": str(int(rng.normal(58 if etiqueta else 50, 8))),
                "sex": "male" if rng.random() < 0.7 else "female",  # noqa: PLR2004
                "chest_pain": "asymptomatic" if etiqueta else "nonanginal",
                "rest_bp": str(int(rng.normal(135, 15))),
                "chol": str(int(rng.normal(250, 45))),
                "fbs": str(int(rng.integers(0, 2))),
                "rest_ecg": "normal",
                "max_hr": str(max_hr),
                "exang": str(int(etiqueta) if rng.random() < 0.7 else 0),  # noqa: PLR2004
                "old_peak": f"{old_peak:.1f}",
                "slope": str(int(rng.integers(1, 4))),
                "ca": str(float(rng.integers(0, 4))),
                "thal": "reversable" if etiqueta else "normal",
                "disease": str(int(etiqueta)),
                # Una columna distinta por fila evita duplicados exactos, que el
                # feature pipeline eliminaría y dejaría el dataset demasiado corto.
                "__id": str(indice),
            }
        )

    return (
        pd.DataFrame(filas).drop(columns="__id").assign(chol=[str(200 + i) for i in range(n_filas)])
    )


@pytest.fixture(scope="module")
def tabla_features() -> pd.DataFrame:
    """Tabla de features válida, producida por el feature pipeline real."""
    return construir_features(generar_crudo())[1]


@pytest.fixture(scope="module")
def ruta_features(tmp_path_factory: pytest.TempPathFactory, tabla_features: pd.DataFrame) -> Path:
    """Parquet de features en disco, tal como lo deja el paso anterior."""
    ruta: Path = tmp_path_factory.mktemp("datos") / "corazon_features.parquet"
    tabla_features.to_parquet(ruta, index=False)
    return ruta


@pytest.fixture
def conjuntos(tabla_features: pd.DataFrame) -> Conjuntos:
    """Separación train/test lista para usar en las pruebas."""
    atributos = tabla_features.drop(columns=["disease"])
    objetivo = tabla_features["disease"]
    resultado: Conjuntos = separar_train_test(atributos, objetivo)
    return resultado


@pytest.fixture
def modelo_entrenado(conjuntos: Conjuntos) -> Pipeline:
    """Pipeline entrenado sobre los datos sintéticos."""
    x_train, _, y_train, _ = conjuntos
    return entrenar(construir_pipeline(), x_train, y_train)


# --------------------------------------------------------------------------- #
# Lectura de datos
# --------------------------------------------------------------------------- #


def test_leer_features_separa_atributos_y_objetivo(
    ruta_features: Path, tabla_features: pd.DataFrame
) -> None:
    """La lectura devuelve X sin la columna objetivo, e y con ella."""
    atributos, objetivo = leer_features(ruta_features)
    assert "disease" not in atributos.columns
    assert atributos.shape[1] == tabla_features.shape[1] - 1
    assert objetivo.name == "disease"
    assert len(atributos) == len(objetivo)


def test_leer_features_valida_el_esquema(ruta_features: Path) -> None:
    """El archivo se revalida contra el contrato del feature pipeline."""
    leer_features(ruta_features, validar=True)


def test_leer_features_falla_si_no_existe(tmp_path: Path) -> None:
    """Sin tabla de features el mensaje indica qué ejecutar antes."""
    with pytest.raises(FileNotFoundError, match="feature pipeline"):
        leer_features(tmp_path / "no_existe.parquet")


def test_leer_features_falla_sin_objetivo(tmp_path: Path, tabla_features: pd.DataFrame) -> None:
    """Una tabla sin la variable objetivo no sirve para entrenar."""
    ruta = tmp_path / "sin_objetivo.parquet"
    tabla_features.drop(columns=["disease"]).to_parquet(ruta, index=False)
    with pytest.raises(ErrorDeValidacion, match="objetivo"):
        leer_features(ruta)


def test_leer_features_falla_con_esquema_corrupto(
    tmp_path: Path, tabla_features: pd.DataFrame
) -> None:
    """Un archivo manipulado a mano se detecta antes de entrenar sobre basura."""
    corrupto = tabla_features.copy()
    corrupto["thal_normal"] = 7.0  # un one-hot sólo admite 0 y 1
    ruta = tmp_path / "corrupto.parquet"
    corrupto.to_parquet(ruta, index=False)
    with pytest.raises(ErrorDeValidacion):
        leer_features(ruta)


# --------------------------------------------------------------------------- #
# Separación train/test
# --------------------------------------------------------------------------- #


def test_separar_respeta_la_proporcion(tabla_features: pd.DataFrame) -> None:
    """El test se lleva la proporción configurada de las filas."""
    atributos = tabla_features.drop(columns=["disease"])
    objetivo = tabla_features["disease"]
    x_train, x_test, _, _ = separar_train_test(atributos, objetivo)
    assert len(x_train) + len(x_test) == len(atributos)
    assert abs(len(x_test) / len(atributos) - PROPORCION_TEST) < TOLERANCIA_PROPORCION


def test_separar_estratifica_la_clase(tabla_features: pd.DataFrame) -> None:
    """La prevalencia de enfermos se mantiene en ambos conjuntos."""
    atributos = tabla_features.drop(columns=["disease"])
    objetivo = tabla_features["disease"]
    _, _, y_train, y_test = separar_train_test(atributos, objetivo)
    assert abs(y_train.mean() - y_test.mean()) < 0.05  # noqa: PLR2004


def test_separar_es_reproducible(tabla_features: pd.DataFrame) -> None:
    """Con la misma semilla, la partición es idéntica."""
    atributos = tabla_features.drop(columns=["disease"])
    objetivo = tabla_features["disease"]
    primera = separar_train_test(atributos, objetivo)[0]
    segunda = separar_train_test(atributos, objetivo)[0]
    pd.testing.assert_frame_equal(primera, segunda)


def test_separar_no_comparte_filas_entre_conjuntos(tabla_features: pd.DataFrame) -> None:
    """Ninguna fila puede estar a la vez en train y en test."""
    atributos = tabla_features.drop(columns=["disease"])
    objetivo = tabla_features["disease"]
    x_train, x_test, _, _ = separar_train_test(atributos, objetivo)
    assert not set(x_train.index) & set(x_test.index)


# --------------------------------------------------------------------------- #
# Construcción y entrenamiento
# --------------------------------------------------------------------------- #


def test_construir_pipeline_incluye_imputacion_y_escalado() -> None:
    """Los pasos que aprenden de los datos van dentro del pipeline."""
    pipeline = construir_pipeline()
    assert list(pipeline.named_steps) == ["imputador", "escalador", "modelo"]


@pytest.mark.parametrize("nombre", sorted(MODELOS))
def test_construir_pipeline_acepta_todos_los_modelos(nombre: str) -> None:
    """Las tres opciones de `--modelo` producen un pipeline válido."""
    assert isinstance(construir_pipeline(nombre), Pipeline)


def test_construir_pipeline_rechaza_modelo_desconocido() -> None:
    """Un nombre de modelo inexistente falla con las opciones disponibles."""
    with pytest.raises(ValueError, match="Modelo desconocido"):
        construir_pipeline("perceptron_magico")


def test_entrenar_deja_el_pipeline_ajustado(conjuntos: Conjuntos) -> None:
    """Tras entrenar, el pipeline predice y expone las clases aprendidas."""
    x_train, x_test, y_train, _ = conjuntos
    pipeline = entrenar(construir_pipeline(), x_train, y_train)
    predicciones = pipeline.predict(x_test)
    assert len(predicciones) == len(x_test)
    assert set(np.unique(predicciones)) <= {0, 1}


def test_entrenar_maneja_los_faltantes(conjuntos: Conjuntos) -> None:
    """El imputador absorbe los NaN que el feature pipeline dejó a propósito."""
    x_train, x_test, y_train, _ = conjuntos
    pipeline = entrenar(construir_pipeline(), x_train, y_train)
    probabilidades = pipeline.predict_proba(x_test)[:, 1]
    assert not np.isnan(probabilidades).any()


def test_entrenamiento_es_reproducible(conjuntos: Conjuntos) -> None:
    """Dos entrenamientos con la misma semilla dan las mismas probabilidades."""
    x_train, x_test, y_train, _ = conjuntos
    primera = entrenar(construir_pipeline(), x_train, y_train).predict_proba(x_test)
    segunda = entrenar(construir_pipeline(), x_train, y_train).predict_proba(x_test)
    np.testing.assert_allclose(primera, segunda)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #


def test_evaluar_devuelve_todas_las_metricas(
    modelo_entrenado: Pipeline, conjuntos: Conjuntos
) -> None:
    """La evaluación cubre las siete métricas, todas en [0, 1]."""
    _, x_test, _, y_test = conjuntos
    metricas = evaluar(modelo_entrenado, x_test, y_test)
    assert len(metricas) == N_METRICAS
    assert all(0.0 <= valor <= 1.0 for valor in metricas.values())


def test_modelo_supera_a_la_base_trivial(modelo_entrenado: Pipeline, conjuntos: Conjuntos) -> None:
    """Con señal en los datos, el modelo debe batir a predecir la clase mayoritaria."""
    x_train, x_test, y_train, y_test = conjuntos
    metricas = evaluar(modelo_entrenado, x_test, y_test)
    trivial = evaluar_modelo_trivial(x_train, y_train, x_test, y_test)
    assert metricas["f1"] > trivial["f1"]
    assert metricas["f1"] > MIN_F1_ACEPTABLE


def test_evaluar_respeta_el_umbral(modelo_entrenado: Pipeline, conjuntos: Conjuntos) -> None:
    """Bajar el umbral no puede reducir la sensibilidad."""
    _, x_test, _, y_test = conjuntos
    por_defecto = evaluar(modelo_entrenado, x_test, y_test, umbral=UMBRAL_DEFECTO)
    permisivo = evaluar(modelo_entrenado, x_test, y_test, umbral=0.2)
    assert permisivo["sensibilidad"] >= por_defecto["sensibilidad"]


def test_matriz_confusion_suma_el_total(modelo_entrenado: Pipeline, conjuntos: Conjuntos) -> None:
    """Las cuatro celdas cubren exactamente el conjunto evaluado."""
    _, x_test, _, y_test = conjuntos
    matriz = matriz_confusion(modelo_entrenado, x_test, y_test)
    assert sum(matriz.values()) == len(y_test)


def test_validacion_cruzada_devuelve_media_y_desviacion(conjuntos: Conjuntos) -> None:
    """La validación cruzada reporta F1 medio y su dispersión."""
    x_train, _, y_train, _ = conjuntos
    resultado = validar_cruzado(construir_pipeline(), x_train, y_train, n_particiones=3)
    assert 0.0 <= resultado["f1"] <= 1.0
    assert resultado["f1_std"] >= 0.0


def test_optimizar_umbral_devuelve_un_valor_valido(conjuntos: Conjuntos) -> None:
    """El umbral óptimo es una probabilidad dentro del rango barrido."""
    x_train, _, y_train, _ = conjuntos
    umbral = optimizar_umbral(construir_pipeline(), x_train, y_train, n_particiones=3)
    assert 0.05 <= umbral <= 0.95  # noqa: PLR2004


def test_tabla_metricas_compara_los_cinco_escenarios() -> None:
    """La tabla final enfrenta train, CV, test, test con umbral y base trivial."""
    ficticias = {"f1": 0.8, "sensibilidad": 0.8}
    tabla = construir_tabla_metricas(
        {**ficticias, "f1_std": 0.05}, ficticias, ficticias, ficticias, ficticias
    )
    assert list(tabla.columns) == [
        "entrenamiento",
        "validacion_cruzada",
        "test",
        "test_umbral_optimo",
        "base_trivial",
    ]
    assert "f1_std" not in tabla.index


# --------------------------------------------------------------------------- #
# Almacenamiento
# --------------------------------------------------------------------------- #


def test_guardar_modelo_se_puede_recargar(
    tmp_path: Path, modelo_entrenado: Pipeline, conjuntos: Conjuntos
) -> None:
    """El modelo recargado reproduce exactamente las mismas predicciones."""
    _, x_test, _, _ = conjuntos
    ruta = tmp_path / "modelos" / "modelo.joblib"
    guardar_modelo(modelo_entrenado, ruta)

    assert ruta.is_file()
    recargado = joblib.load(ruta)
    np.testing.assert_array_equal(recargado.predict(x_test), modelo_entrenado.predict(x_test))


def test_guardar_artefacto_conserva_los_metadatos(
    tmp_path: Path, modelo_entrenado: Pipeline
) -> None:
    """El artefacto guarda el pipeline junto con umbral y versiones."""
    ruta = tmp_path / "artefacto.joblib"
    guardar_artefacto(
        {"pipeline": modelo_entrenado, "umbral_optimo": 0.42, "modelo": "gradient_boosting"},
        ruta,
    )
    recargado = joblib.load(ruta)
    assert isinstance(recargado["pipeline"], Pipeline)
    assert recargado["umbral_optimo"] == 0.42  # noqa: PLR2004


def test_guardar_metricas_escribe_csv_legible(tmp_path: Path) -> None:
    """El CSV de métricas se puede releer con la métrica como índice."""
    tabla = pd.DataFrame({"test": {"f1": 0.85, "sensibilidad": 0.9}})
    ruta = tmp_path / "reportes" / "metricas.csv"
    guardar_metricas(tabla, ruta)

    releido = pd.read_csv(ruta, index_col="metrica")
    assert releido.loc["f1", "test"] == 0.85  # noqa: PLR2004


def test_guardar_predicciones_incluye_ambos_umbrales(
    tmp_path: Path, modelo_entrenado: Pipeline, conjuntos: Conjuntos
) -> None:
    """El CSV de predicciones permite auditar caso por caso."""
    _, x_test, _, y_test = conjuntos
    ruta = tmp_path / "predicciones.csv"
    guardar_predicciones(modelo_entrenado, x_test, y_test, 0.3, ruta)

    releido = pd.read_csv(ruta)
    assert len(releido) == len(y_test)
    assert {
        "real",
        "probabilidad",
        "prediccion_umbral_defecto",
        "prediccion_umbral_optimo",
    } <= set(releido.columns)


# --------------------------------------------------------------------------- #
# Ejecución de extremo a extremo
# --------------------------------------------------------------------------- #


@pytest.fixture
def rutas_salida(tmp_path: Path) -> RutasSalida:
    """Rutas de salida en un directorio temporal."""
    return RutasSalida(
        modelo=tmp_path / "modelo.joblib",
        artefacto=tmp_path / "artefacto.joblib",
        predicciones=tmp_path / "predicciones.csv",
        metricas=tmp_path / "metricas.csv",
        manifiesto=tmp_path / "manifiesto.json",
        checks_split=tmp_path / "checks_separacion.csv",
    )


def test_ejecutar_pipeline_genera_todas_las_salidas(
    ruta_features: Path, rutas_salida: RutasSalida
) -> None:
    """La ejecución completa deja modelo, predicciones, métricas y manifiesto."""
    resumen = ejecutar_pipeline(ruta_features, rutas_salida)

    assert all(
        getattr(rutas_salida, campo).is_file() for campo in rutas_salida.__dataclass_fields__
    )
    assert resumen["modelo"] == MODELO_POR_DEFECTO
    assert resumen["n_train"] + resumen["n_test"] == resumen["n_filas"]


def test_manifiesto_documenta_la_ejecucion(ruta_features: Path, rutas_salida: RutasSalida) -> None:
    """El manifiesto guarda métricas, umbral, semilla y versión de sklearn."""
    ejecutar_pipeline(ruta_features, rutas_salida)
    manifiesto = json.loads(rutas_salida.manifiesto.read_text(encoding="utf-8"))

    assert manifiesto["semilla"] == 42  # noqa: PLR2004
    assert "test" in manifiesto["metricas"]
    assert 0.0 < manifiesto["metricas"]["test"]["f1"] <= 1.0
    assert manifiesto["version_sklearn"]
    assert sum(manifiesto["matriz_confusion_test"].values()) == manifiesto["n_test"]


def test_main_devuelve_cero_en_ejecucion_correcta(
    ruta_features: Path, rutas_salida: RutasSalida
) -> None:
    """El script termina con código 0 y deja el modelo escrito."""
    codigo = main(
        [
            "--features",
            str(ruta_features),
            "--modelo-salida",
            str(rutas_salida.modelo),
            "--artefacto",
            str(rutas_salida.artefacto),
            "--predicciones",
            str(rutas_salida.predicciones),
            "--metricas",
            str(rutas_salida.metricas),
            "--manifiesto",
            str(rutas_salida.manifiesto),
        ]
    )
    assert codigo == 0
    assert rutas_salida.modelo.is_file()


def test_main_devuelve_uno_sin_features(tmp_path: Path, rutas_salida: RutasSalida) -> None:
    """Sin la tabla de features el script falla de forma controlada."""
    codigo = main(
        [
            "--features",
            str(tmp_path / "no_existe.parquet"),
            "--modelo-salida",
            str(rutas_salida.modelo),
            "--artefacto",
            str(rutas_salida.artefacto),
            "--predicciones",
            str(rutas_salida.predicciones),
            "--metricas",
            str(rutas_salida.metricas),
            "--manifiesto",
            str(rutas_salida.manifiesto),
        ]
    )
    assert codigo == 1
    assert not rutas_salida.modelo.exists()


def test_main_devuelve_uno_con_features_invalidos(
    tmp_path: Path, tabla_features: pd.DataFrame, rutas_salida: RutasSalida
) -> None:
    """Unos features que no cumplen el contrato no llegan a entrenar un modelo."""
    corrupto = tabla_features.copy()
    corrupto["age"] = np.inf
    ruta = tmp_path / "corrupto.parquet"
    corrupto.to_parquet(ruta, index=False)

    codigo = main(
        [
            "--features",
            str(ruta),
            "--modelo-salida",
            str(rutas_salida.modelo),
            "--artefacto",
            str(rutas_salida.artefacto),
            "--predicciones",
            str(rutas_salida.predicciones),
            "--metricas",
            str(rutas_salida.metricas),
            "--manifiesto",
            str(rutas_salida.manifiesto),
        ]
    )
    assert codigo == 1
    assert not rutas_salida.modelo.exists()


# --------------------------------------------------------------------------- #
# Verificación de la separación train/test
# --------------------------------------------------------------------------- #


def severidades(resultado: ResultadoSeparacion) -> dict[str, str]:
    """Mapa comprobación -> severidad, para aserciones legibles."""
    return {c.nombre: c.severidad for c in resultado.comprobaciones}


# --- Caso válido -----------------------------------------------------------


def test_separacion_valida_no_reporta_errores(conjuntos: Conjuntos) -> None:
    """La partición que produce el propio pipeline supera todas las comprobaciones."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(x_train, x_test, y_train, y_test)
    assert resultado.valida
    assert resultado.errores == []


def test_separacion_valida_cubre_todas_las_comprobaciones(conjuntos: Conjuntos) -> None:
    """Se evalúan las nueve comprobaciones, no un subconjunto."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(x_train, x_test, y_train, y_test)
    assert len(resultado.comprobaciones) == N_COMPROBACIONES
    assert {
        "tamaño mínimo",
        "proporción de test",
        "columnas",
        "índices disjuntos",
        "filas duplicadas entre conjuntos",
        "cobertura de clases",
        "estratificación",
        "distribución de atributos (KS)",
        "faltantes comparables",
    } == set(severidades(resultado))


def test_alias_en_ingles_apunta_a_la_misma_funcion() -> None:
    """El nombre del enunciado, `validate_train_test_split`, está disponible."""
    assert validate_train_test_split is validar_separacion_train_test


def test_resultado_expone_tabla_y_resumen(conjuntos: Conjuntos) -> None:
    """El resultado se puede inspeccionar como tabla y resumir en una línea."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(x_train, x_test, y_train, y_test)
    tabla = resultado.tabla()
    assert list(tabla.columns) == ["comprobacion", "severidad", "detalle"]
    assert len(tabla) == len(resultado.comprobaciones)
    assert "comprobaciones" in resultado.resumen()


# --- Casos inválidos: errores (fuga de información) ------------------------


def test_detecta_indices_compartidos(conjuntos: Conjuntos) -> None:
    """Una fila presente en ambos conjuntos es fuga directa: error."""
    x_train, x_test, y_train, y_test = conjuntos
    x_test_sucio = pd.concat([x_test, x_train.head(3)])
    y_test_sucio = pd.concat([y_test, y_train.head(3)])

    resultado = validar_separacion_train_test(x_train, x_test_sucio, y_train, y_test_sucio)
    assert not resultado.valida
    assert severidades(resultado)["índices disjuntos"] == "error"


def test_detecta_filas_duplicadas_con_indices_distintos(conjuntos: Conjuntos) -> None:
    """El mismo paciente con otro índice sigue siendo fuga: comparar índices no basta."""
    x_train, x_test, y_train, y_test = conjuntos
    copia = x_train.head(3).copy()
    copia.index = [999_001, 999_002, 999_003]
    etiquetas = y_train.head(3).copy()
    etiquetas.index = copia.index

    resultado = validar_separacion_train_test(
        x_train, pd.concat([x_test, copia]), y_train, pd.concat([y_test, etiquetas])
    )
    assert not resultado.valida
    assert severidades(resultado)["filas duplicadas entre conjuntos"] == "error"
    # Los índices sí son disjuntos: el error lo detecta la huella de la fila.
    assert severidades(resultado)["índices disjuntos"] == "ok"


def test_detecta_clase_ausente_en_test(conjuntos: Conjuntos) -> None:
    """Sin ambas clases en el test, las métricas no significan nada."""
    x_train, x_test, y_train, y_test = conjuntos
    solo_sanos = y_test[y_test == 0]
    resultado = validar_separacion_train_test(
        x_train, x_test.loc[solo_sanos.index], y_train, solo_sanos
    )
    assert not resultado.valida
    assert severidades(resultado)["cobertura de clases"] == "error"


def test_detecta_conjunto_demasiado_pequeno(conjuntos: Conjuntos) -> None:
    """Un test de 5 filas no permite estimar nada."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(x_train, x_test.head(5), y_train, y_test.head(5))
    assert not resultado.valida
    assert severidades(resultado)["tamaño mínimo"] == "error"


def test_detecta_columnas_distintas(conjuntos: Conjuntos) -> None:
    """Train y test tienen que compartir exactamente el mismo esquema."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(
        x_train, x_test.drop(columns=["age"]), y_train, y_test
    )
    assert not resultado.valida
    assert severidades(resultado)["columnas"] == "error"


# --- Casos inválidos: advertencias (representatividad) --------------------


def test_advierte_proporcion_desviada(conjuntos: Conjuntos) -> None:
    """Un reparto muy distinto del configurado se avisa, pero no bloquea."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(x_train, x_test.head(25), y_train, y_test.head(25))
    assert severidades(resultado)["proporción de test"] == "advertencia"
    assert resultado.valida  # una advertencia no invalida la separación


def test_advierte_estratificacion_rota(conjuntos: Conjuntos) -> None:
    """Un test con prevalencia muy distinta a la de train se avisa."""
    x_train, x_test, y_train, y_test = conjuntos
    sesgado = pd.concat([y_test[y_test == 1], y_test[y_test == 0].head(3)])
    resultado = validar_separacion_train_test(x_train, x_test.loc[sesgado.index], y_train, sesgado)
    assert severidades(resultado)["estratificación"] == "advertencia"


def test_advierte_distribucion_distinta(conjuntos: Conjuntos) -> None:
    """Si un atributo se distribuye distinto, el test dejó de representar el problema."""
    x_train, x_test, y_train, y_test = conjuntos
    desplazado = x_test.copy()
    desplazado["age"] = desplazado["age"] + 40  # un test de sólo mayores

    resultado = validar_separacion_train_test(x_train, desplazado, y_train, y_test)
    assert severidades(resultado)["distribución de atributos (KS)"] == "advertencia"
    assert (
        "age"
        in dict((c.nombre, c.detalle) for c in resultado.comprobaciones)[
            "distribución de atributos (KS)"
        ]
    )


def test_advierte_faltantes_desbalanceados(conjuntos: Conjuntos) -> None:
    """Un test con muchos más nulos que el train se avisa."""
    x_train, x_test, y_train, y_test = conjuntos
    con_huecos = x_test.copy()
    con_huecos.loc[con_huecos.index[: len(con_huecos) // 2], "chol"] = np.nan

    resultado = validar_separacion_train_test(x_train, con_huecos, y_train, y_test)
    assert severidades(resultado)["faltantes comparables"] == "advertencia"


# --- Política: qué detiene el pipeline y qué no ---------------------------


def test_aplicar_no_lanza_si_todo_esta_bien(conjuntos: Conjuntos) -> None:
    """Una separación correcta deja continuar sin ruido."""
    x_train, x_test, y_train, y_test = conjuntos
    aplicar_resultado_separacion(validar_separacion_train_test(x_train, x_test, y_train, y_test))


def test_aplicar_lanza_error_controlado_ante_fuga(conjuntos: Conjuntos) -> None:
    """Un error detiene el pipeline con un mensaje que nombra la comprobación."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(
        x_train, pd.concat([x_test, x_train.head(3)]), y_train, pd.concat([y_test, y_train.head(3)])
    )
    with pytest.raises(ErrorDeSeparacion, match="índices disjuntos"):
        aplicar_resultado_separacion(resultado)


def test_aplicar_tolera_advertencias_por_defecto(conjuntos: Conjuntos) -> None:
    """Una advertencia se registra pero no detiene la ejecución."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(x_train, x_test.head(25), y_train, y_test.head(25))
    assert resultado.advertencias
    aplicar_resultado_separacion(resultado, estricto=False)


def test_modo_estricto_convierte_advertencias_en_errores(conjuntos: Conjuntos) -> None:
    """Con `--estricto` cualquier advertencia detiene el pipeline."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(x_train, x_test.head(25), y_train, y_test.head(25))
    with pytest.raises(ErrorDeSeparacion, match="estricto"):
        aplicar_resultado_separacion(resultado, estricto=True)


def test_error_de_separacion_hereda_de_error_de_validacion() -> None:
    """Así el manejador de `main` la trata igual: sin traza y sin persistir nada."""
    assert issubclass(ErrorDeSeparacion, ErrorDeValidacion)


# --- Persistencia de los resultados ---------------------------------------


def test_guardar_checks_escribe_csv(tmp_path: Path, conjuntos: Conjuntos) -> None:
    """Los resultados de la separación quedan en un CSV auditable."""
    x_train, x_test, y_train, y_test = conjuntos
    resultado = validar_separacion_train_test(x_train, x_test, y_train, y_test)
    ruta = tmp_path / "reportes" / "checks_separacion.csv"
    guardar_checks_separacion(resultado, ruta)

    releido = pd.read_csv(ruta)
    assert len(releido) == N_COMPROBACIONES
    assert set(releido["severidad"]) <= {"ok", "advertencia", "error"}


def test_pipeline_completo_guarda_los_checks(
    ruta_features: Path, rutas_salida: RutasSalida
) -> None:
    """La ejecución normal deja también el CSV de comprobaciones y el resumen."""
    resumen = ejecutar_pipeline(ruta_features, rutas_salida)
    assert rutas_salida.checks_split.is_file()
    assert "comprobaciones" in resumen["checks_separacion"]["resumen"]


def test_pipeline_no_entrena_si_la_separacion_tiene_fuga(
    monkeypatch: pytest.MonkeyPatch, ruta_features: Path, rutas_salida: RutasSalida
) -> None:
    """Ante una partición con fuga, el pipeline aborta sin guardar el modelo."""

    def separacion_con_fuga(
        atributos: pd.DataFrame, objetivo: pd.Series, *args: Any, **kwargs: Any
    ) -> Conjuntos:
        """Devuelve el mismo conjunto como train y como test."""
        return atributos, atributos, objetivo, objetivo

    monkeypatch.setattr(
        "pipelines.training_pipeline.train_pipeline.separar_train_test", separacion_con_fuga
    )

    with pytest.raises(ErrorDeSeparacion):
        ejecutar_pipeline(ruta_features, rutas_salida)

    assert not rutas_salida.modelo.exists()
    assert not rutas_salida.metricas.exists()


def test_main_devuelve_uno_ante_fuga(
    monkeypatch: pytest.MonkeyPatch, ruta_features: Path, rutas_salida: RutasSalida
) -> None:
    """El script termina con código 1 y sin modelo cuando detecta fuga."""

    def separacion_con_fuga(
        atributos: pd.DataFrame, objetivo: pd.Series, *args: Any, **kwargs: Any
    ) -> Conjuntos:
        return atributos, atributos, objetivo, objetivo

    monkeypatch.setattr(
        "pipelines.training_pipeline.train_pipeline.separar_train_test", separacion_con_fuga
    )

    codigo = main(
        [
            "--features",
            str(ruta_features),
            "--modelo-salida",
            str(rutas_salida.modelo),
            "--artefacto",
            str(rutas_salida.artefacto),
            "--predicciones",
            str(rutas_salida.predicciones),
            "--metricas",
            str(rutas_salida.metricas),
            "--manifiesto",
            str(rutas_salida.manifiesto),
            "--checks-split",
            str(rutas_salida.checks_split),
        ]
    )
    assert codigo == 1
    assert not rutas_salida.modelo.exists()
