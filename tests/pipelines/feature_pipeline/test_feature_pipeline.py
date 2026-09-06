"""Pruebas unitarias del feature pipeline del proyecto Heart_project."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import (
    CATEGORIAS_VALIDAS,
    OBJETIVO,
    codificar_categoricas,
    construir_features,
    depurar_filas,
    ejecutar_pipeline,
    generar_atributos_clinicos,
    guardar_metadatos,
    guardar_parquet,
    leer_datos_crudos,
    localizar_raiz,
    main,
    normalizar_nombre,
    sanear_dataset,
)

#: Valores esperados usados en las aserciones (evita "magic values" en el linter).
EDAD_PRIMERA_FILA = 63.0
FILAS_TRAS_DEPURAR = 3
CODIGO_SLOPE_3 = 2.0

COLUMNAS = [
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
    "disease",
]


def fila(**cambios: object) -> dict[str, object]:
    """Devuelve una fila válida de ejemplo, con los campos indicados sobrescritos."""
    base: dict[str, object] = {
        "age": "63",
        "sex": "Male",
        "chest_pain": "typical",
        "rest_bp": "145",
        "chol": "233",
        "fbs": "1",
        "rest_ecg": "left ventricular hypertrophy ",
        "max_hr": "150",
        "exang": "0",
        "old_peak": "2.3",
        "slope": "3",
        "ca": "0.0",
        "thal": "fixed",
        "disease": "0",
    }
    base.update(cambios)
    return base


@pytest.fixture
def crudo() -> pd.DataFrame:
    """DataFrame crudo de ejemplo con basura, duplicados y faltantes."""
    filas = [
        fila(),
        fila(),  # duplicado exacto
        fila(age="67", sex="Female", chest_pain="asymptomatic", disease="1", slope="2"),
        fila(age="basura", thal="???", disease="1"),  # valores no interpretables
        fila(age="55", disease="no_se_sabe"),  # sin etiqueta válida
    ]
    return pd.DataFrame(filas, columns=COLUMNAS).astype(str)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def test_normalizar_nombre_convierte_a_sufijo_valido() -> None:
    """Espacios y guiones se convierten en guiones bajos simples."""
    assert normalizar_nombre("st-t wave abnormality") == "st_t_wave_abnormality"
    assert normalizar_nombre("  Normal ") == "normal"


def test_localizar_raiz_encuentra_pyproject(tmp_path: Path) -> None:
    """La raíz es el primer directorio ascendente que contiene pyproject.toml."""
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    anidado = tmp_path / "src" / "pipelines"
    anidado.mkdir(parents=True)
    assert localizar_raiz(anidado / "archivo.py") == tmp_path.resolve()


# --------------------------------------------------------------------------- #
# Lectura
# --------------------------------------------------------------------------- #


def test_leer_datos_crudos_lee_todo_como_texto(tmp_path: Path, crudo: pd.DataFrame) -> None:
    """El CSV se lee sin inferir tipos, para no perder los valores basura."""
    ruta = tmp_path / "corazon.csv"
    crudo.to_csv(ruta, index=False)
    leido = leer_datos_crudos(ruta)
    assert leido.shape == crudo.shape
    assert not any(pd.api.types.is_numeric_dtype(leido[col]) for col in leido.columns)


def test_leer_datos_crudos_falla_si_no_existe(tmp_path: Path) -> None:
    """Un archivo inexistente produce FileNotFoundError con mensaje explícito."""
    with pytest.raises(FileNotFoundError, match="datos crudos"):
        leer_datos_crudos(tmp_path / "no_existe.csv")


def test_leer_datos_crudos_falla_si_faltan_columnas(tmp_path: Path, crudo: pd.DataFrame) -> None:
    """Si el esquema no trae las columnas esperadas, se aborta la ejecución."""
    ruta = tmp_path / "incompleto.csv"
    crudo.drop(columns=["thal", "ca"]).to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="faltan columnas"):
        leer_datos_crudos(ruta)


# --------------------------------------------------------------------------- #
# Saneamiento
# --------------------------------------------------------------------------- #


def test_sanear_convierte_basura_numerica_en_nan(crudo: pd.DataFrame) -> None:
    """Un texto no numérico en una columna numérica se convierte en NaN."""
    saneado = sanear_dataset(crudo)
    assert pd.isna(saneado.loc[3, "age"])
    assert saneado.loc[0, "age"] == EDAD_PRIMERA_FILA


def test_sanear_normaliza_categorias(crudo: pd.DataFrame) -> None:
    """Las categorías se pasan a minúsculas y se recortan los espacios."""
    saneado = sanear_dataset(crudo)
    assert saneado.loc[0, "sex"] == "male"
    assert saneado.loc[0, "rest_ecg"] == "left ventricular hypertrophy"


def test_sanear_descarta_categorias_fuera_del_catalogo(crudo: pd.DataFrame) -> None:
    """Un valor que no está en el catálogo del dominio se marca como faltante."""
    saneado = sanear_dataset(crudo)
    assert pd.isna(saneado.loc[3, "thal"])
    assert set(saneado["thal"].dropna()) <= set(CATEGORIAS_VALIDAS["thal"])


def test_sanear_no_elimina_filas(crudo: pd.DataFrame) -> None:
    """El saneamiento sólo corrige tipos: nunca cambia el número de filas."""
    assert len(sanear_dataset(crudo)) == len(crudo)


# --------------------------------------------------------------------------- #
# Depuración de filas
# --------------------------------------------------------------------------- #


def test_depurar_elimina_duplicados_y_filas_sin_objetivo(crudo: pd.DataFrame) -> None:
    """De 5 filas quedan 3: se va un duplicado exacto y una fila sin etiqueta."""
    depurado = depurar_filas(sanear_dataset(crudo))
    assert len(depurado) == FILAS_TRAS_DEPURAR
    assert depurado[OBJETIVO].notna().all()


def test_depurar_deja_el_objetivo_como_entero(crudo: pd.DataFrame) -> None:
    """La variable objetivo queda como entero 0/1, lista para el modelo."""
    depurado = depurar_filas(sanear_dataset(crudo))
    assert depurado[OBJETIVO].dtype.kind == "i"
    assert set(depurado[OBJETIVO]) <= {0, 1}


# --------------------------------------------------------------------------- #
# Codificación
# --------------------------------------------------------------------------- #


def test_codificar_binarias_y_ordinales(crudo: pd.DataFrame) -> None:
    """`sex` se codifica 0/1 y `slope` conserva el orden 1 < 2 < 3."""
    depurado = depurar_filas(sanear_dataset(crudo))
    codificado = codificar_categoricas(depurado)
    assert codificado.loc[0, "sex"] == 1.0  # male
    assert codificado.loc[1, "sex"] == 0.0  # female
    assert codificado.loc[0, "slope"] == CODIGO_SLOPE_3  # "3" -> tercera posición
    assert codificado.loc[1, "slope"] == 1.0  # "2" -> segunda posición


def test_codificar_nominales_usa_vocabulario_fijo(crudo: pd.DataFrame) -> None:
    """Se genera una columna por categoría del catálogo, exista o no en los datos."""
    depurado = depurar_filas(sanear_dataset(crudo))
    codificado = codificar_categoricas(depurado)
    esperadas = {f"chest_pain_{c}" for c in CATEGORIAS_VALIDAS["chest_pain"]}
    assert esperadas <= set(codificado.columns)
    assert codificado.loc[0, "chest_pain_typical"] == 1.0
    assert codificado.loc[0, "chest_pain_asymptomatic"] == 0.0


def test_codificar_propaga_faltantes_en_one_hot(crudo: pd.DataFrame) -> None:
    """Si la categoría original es NaN, todas sus columnas one-hot quedan en NaN."""
    depurado = depurar_filas(sanear_dataset(crudo))
    codificado = codificar_categoricas(depurado)
    columnas_thal = [f"thal_{c}" for c in CATEGORIAS_VALIDAS["thal"]]
    faltante = depurado["thal"].isna()
    assert faltante.any(), "el fixture debe incluir al menos un `thal` inválido"
    assert codificado.loc[faltante, columnas_thal].isna().all().all()


# --------------------------------------------------------------------------- #
# Atributos derivados
# --------------------------------------------------------------------------- #


def test_atributos_clinicos_calculan_los_valores_esperados() -> None:
    """Los atributos derivados siguen las fórmulas del dominio."""
    datos = pd.DataFrame(
        {
            "age": [60.0],
            "max_hr": [150.0],
            "chol": [200.0],
            "rest_bp": [120.0],
            "old_peak": [2.0],
            "slope": [3.0],
        }
    )
    derivados = generar_atributos_clinicos(datos)
    assert derivados.loc[0, "fc_maxima_teorica"] == pytest.approx(160.0)
    assert derivados.loc[0, "reserva_cardiaca"] == pytest.approx(-10.0)
    assert derivados.loc[0, "pct_fc_alcanzada"] == pytest.approx(150 / 160)
    assert derivados.loc[0, "ratio_chol_edad"] == pytest.approx(200 / 60)
    assert derivados.loc[0, "presion_x_chol"] == pytest.approx(24.0)
    assert derivados.loc[0, "indice_riesgo_st"] == pytest.approx(4.0)


def test_atributos_clinicos_no_dividen_por_cero() -> None:
    """Una edad de 0 no produce infinitos, sino NaN."""
    datos = pd.DataFrame(
        {
            "age": [0.0],
            "max_hr": [150.0],
            "chol": [200.0],
            "rest_bp": [120.0],
            "old_peak": [1.0],
            "slope": [2.0],
        }
    )
    derivados = generar_atributos_clinicos(datos)
    assert np.isnan(derivados.loc[0, "ratio_chol_edad"])
    assert np.isfinite(derivados.to_numpy()[~np.isnan(derivados.to_numpy())]).all()


# --------------------------------------------------------------------------- #
# Construcción completa
# --------------------------------------------------------------------------- #


def test_construir_features_devuelve_tabla_numerica(crudo: pd.DataFrame) -> None:
    """Todas las columnas de la tabla de features son numéricas."""
    _, features = construir_features(crudo)
    assert all(pd.api.types.is_numeric_dtype(features[col]) for col in features.columns)


def test_construir_features_incluye_el_objetivo_al_final(crudo: pd.DataFrame) -> None:
    """La variable objetivo se conserva y queda como última columna."""
    _, features = construir_features(crudo)
    assert features.columns[-1] == OBJETIVO


def test_construir_features_conserva_las_filas_depuradas(crudo: pd.DataFrame) -> None:
    """La tabla de features tiene exactamente las filas del dataset depurado."""
    depurado, features = construir_features(crudo)
    assert len(features) == len(depurado)


def test_construir_features_es_determinista(crudo: pd.DataFrame) -> None:
    """Dos ejecuciones sobre la misma entrada producen el mismo resultado."""
    _, primera = construir_features(crudo)
    _, segunda = construir_features(crudo)
    pd.testing.assert_frame_equal(primera, segunda)


def test_construir_features_no_imputa_faltantes(crudo: pd.DataFrame) -> None:
    """Los faltantes se conservan: la imputación es tarea del pipeline de entrenamiento."""
    _, features = construir_features(crudo)
    assert features["age"].isna().any()


# --------------------------------------------------------------------------- #
# Escritura
# --------------------------------------------------------------------------- #


def test_guardar_parquet_crea_directorios(tmp_path: Path) -> None:
    """El parquet se escribe aunque el directorio destino no exista."""
    destino = tmp_path / "data" / "04_feature" / "features.parquet"
    datos = pd.DataFrame({"a": [1.0, 2.0]})
    guardar_parquet(datos, destino)
    assert destino.is_file()
    pd.testing.assert_frame_equal(pd.read_parquet(destino), datos)


def test_guardar_metadatos_registra_el_resumen(tmp_path: Path, crudo: pd.DataFrame) -> None:
    """El manifiesto JSON documenta filas, columnas y objetivo."""
    _, features = construir_features(crudo)
    destino = tmp_path / "metadata.json"
    guardar_metadatos(features, tmp_path / "corazon.csv", destino)
    manifiesto = json.loads(destino.read_text(encoding="utf-8"))
    assert manifiesto["n_filas"] == len(features)
    assert manifiesto["n_columnas"] == features.shape[1]
    assert manifiesto["objetivo"] == OBJETIVO
    assert manifiesto["columnas"] == list(features.columns)


# --------------------------------------------------------------------------- #
# Ejecución de extremo a extremo
# --------------------------------------------------------------------------- #


def test_ejecutar_pipeline_genera_los_tres_archivos(tmp_path: Path, crudo: pd.DataFrame) -> None:
    """La ejecución completa produce intermedio, features y metadatos."""
    entrada = tmp_path / "corazon.csv"
    crudo.to_csv(entrada, index=False)
    intermedio = tmp_path / "02_intermediate" / "corazon.parquet"
    salida = tmp_path / "04_feature" / "features.parquet"
    metadatos = tmp_path / "04_feature" / "metadata.json"

    features = ejecutar_pipeline(entrada, intermedio, salida, metadatos)

    assert intermedio.is_file()
    assert metadatos.is_file()
    pd.testing.assert_frame_equal(pd.read_parquet(salida), features)


def test_main_devuelve_cero_en_ejecucion_correcta(tmp_path: Path, crudo: pd.DataFrame) -> None:
    """El script termina con código de salida 0 cuando todo va bien."""
    entrada = tmp_path / "corazon.csv"
    crudo.to_csv(entrada, index=False)
    codigo = main(
        [
            "--entrada",
            str(entrada),
            "--intermedio",
            str(tmp_path / "inter.parquet"),
            "--salida",
            str(tmp_path / "features.parquet"),
            "--metadatos",
            str(tmp_path / "meta.json"),
        ]
    )
    assert codigo == 0
    assert (tmp_path / "features.parquet").is_file()


def test_main_devuelve_uno_si_falta_la_entrada(tmp_path: Path) -> None:
    """Un error controlado se registra y devuelve código de salida 1."""
    codigo = main(
        [
            "--entrada",
            str(tmp_path / "no_existe.csv"),
            "--intermedio",
            str(tmp_path / "inter.parquet"),
            "--salida",
            str(tmp_path / "features.parquet"),
            "--metadatos",
            str(tmp_path / "meta.json"),
        ]
    )
    assert codigo == 1
    assert not (tmp_path / "features.parquet").exists()
