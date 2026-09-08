"""Demo web del proyecto Heart_project, construida con Streamlit.

Formulario clínico que recoge las 13 variables del paciente y devuelve la
probabilidad de enfermedad coronaria que estima el modelo entrenado.

Ejecutar en local, desde la raíz del repositorio:

    uv run streamlit run src/demo/app.py

Criterio de diseño
------------------
La app **no transforma nada por su cuenta**. Construye el mismo diccionario de datos
crudos que produciría un CSV y se lo entrega al `inference_pipeline`, que aplica el
saneamiento, la codificación y los atributos derivados exactamente como en el
entrenamiento. Es la misma regla que gobierna los entregables 9 a 14: una única ruta
de transformación, importada, nunca copiada.

Eso tiene una consecuencia práctica que conviene notar: si mañana cambia una regla del
feature pipeline, la demo se entera sola. La alternativa —copiar los transformadores
junto a la app, como hacía la demo anterior— obliga a acordarse de actualizarlos, y el
día que alguien no se acuerde la demo seguirá funcionando y dando otro resultado.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# `src` al path para poder importar los pipelines del proyecto. Streamlit Cloud
# arranca la app desde la raíz del repositorio, así que la ruta se resuelve
# respecto a este archivo y no respecto al directorio de trabajo.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipelines.feature_pipeline.feature_pipeline import (  # noqa: E402
    CATEGORIAS_VALIDAS,
    RANGOS_VALIDOS,
)
from pipelines.inference_pipeline.inference_pipeline import (  # noqa: E402
    BANDAS_RIESGO,
    COLS_REQUERIDAS,
    ModeloCargado,
    alinear_columnas,
    cargar_modelo,
    predecir,
    transformar,
)

RAIZ = _SRC.parent

#: Rutas donde buscar el modelo, en orden de preferencia.
#: `models/` está versionada para que el despliegue en Streamlit Cloud encuentre el
#: artefacto: `data/` está en el `.gitignore` y no viaja al repositorio.
RUTAS_MODELO = [
    RAIZ / "models" / "modelo_corazon_completo.joblib",
    RAIZ / "data" / "06_models" / "modelo_corazon_completo.joblib",
]

#: Etiquetas legibles para las categorías del dominio.
ETIQUETAS: dict[str, dict[str, str]] = {
    "sex": {"female": "Mujer", "male": "Hombre"},
    "chest_pain": {
        "typical": "Angina típica",
        "nontypical": "Angina atípica",
        "nonanginal": "Dolor no anginoso",
        "asymptomatic": "Asintomático",
    },
    "rest_ecg": {
        "normal": "Normal",
        "left ventricular hypertrophy": "Hipertrofia ventricular izquierda",
        "st-t wave abnormality": "Anomalía de la onda ST-T",
    },
    "exang": {"0": "No", "1": "Sí"},
    "slope": {"1": "Ascendente", "2": "Plana", "3": "Descendente"},
    "thal": {"normal": "Normal", "fixed": "Defecto fijo", "reversable": "Defecto reversible"},
}

#: Color de cada banda de riesgo, para el mensaje del resultado.
COLORES_RIESGO = {
    "muy bajo": "#1baf7a",
    "bajo": "#7ac143",
    "medio": "#eda100",
    "alto": "#eb6834",
    "muy alto": "#d63b3b",
}

#: Casos de ejemplo para probar la app sin teclear 13 campos.
EJEMPLOS: dict[str, dict[str, Any]] = {
    "Perfil de bajo riesgo": {
        "age": 42,
        "sex": "female",
        "chest_pain": "nonanginal",
        "rest_bp": 120,
        "chol": 190,
        "fbs": 0,
        "rest_ecg": "normal",
        "max_hr": 175,
        "exang": "0",
        "old_peak": 0.2,
        "slope": "1",
        "ca": 0,
        "thal": "normal",
    },
    "Perfil de alto riesgo": {
        "age": 64,
        "sex": "male",
        "chest_pain": "asymptomatic",
        "rest_bp": 160,
        "chol": 300,
        "fbs": 1,
        "rest_ecg": "left ventricular hypertrophy",
        "max_hr": 108,
        "exang": "1",
        "old_peak": 2.6,
        "slope": "2",
        "ca": 3,
        "thal": "reversable",
    },
    "Caso intermedio": {
        "age": 55,
        "sex": "male",
        "chest_pain": "nontypical",
        "rest_bp": 138,
        "chol": 245,
        "fbs": 0,
        "rest_ecg": "normal",
        "max_hr": 150,
        "exang": "0",
        "old_peak": 1.2,
        "slope": "2",
        "ca": 1,
        "thal": "normal",
    },
}


# --------------------------------------------------------------------------- #
# Carga del modelo
# --------------------------------------------------------------------------- #


def localizar_modelo() -> Path:
    """Devuelve la primera ruta de modelo que exista."""
    for ruta in RUTAS_MODELO:
        if ruta.is_file():
            return ruta
    return RUTAS_MODELO[0]


@st.cache_resource
def obtener_modelo() -> ModeloCargado:
    """Carga el modelo una sola vez y lo reutiliza entre interacciones.

    `cache_resource` es lo que evita releer 170 KB de disco y reconstruir el
    pipeline en cada clic: sin él la app se siente lenta sin motivo.
    """
    return cargar_modelo(localizar_modelo())


# --------------------------------------------------------------------------- #
# Predicción
# --------------------------------------------------------------------------- #


def predecir_paciente(modelo: ModeloCargado, datos: dict[str, Any], umbral: float) -> pd.Series:
    """Convierte el formulario en una fila cruda y la pasa por el inference pipeline.

    Los valores se convierten a texto a propósito: es exactamente lo que llegaría en
    un CSV, y así el saneamiento del feature pipeline hace el mismo trabajo que hace
    en producción, sin un camino especial para la demo.
    """
    crudo = pd.DataFrame([{col: str(datos[col]) for col in COLS_REQUERIDAS}])
    features = alinear_columnas(transformar(crudo), modelo)
    resultado: pd.Series = predecir(modelo, features, umbral=umbral).iloc[0]
    return resultado


def senales_de_riesgo(datos: dict[str, Any]) -> list[tuple[str, bool]]:
    """Cuatro señales clínicas reconocibles, para acompañar la probabilidad.

    No son la explicación del modelo —el gradient boosting no decide así— sino un
    contexto que el usuario puede contrastar con lo que ve en el paciente.
    """
    return [
        ("Talasemia distinta de normal", datos["thal"] != "normal"),
        ("Dolor torácico asintomático", datos["chest_pain"] == "asymptomatic"),
        ("Vasos afectados en fluoroscopia", int(datos["ca"]) >= 1),
        ("Angina inducida por ejercicio", datos["exang"] == "1"),
    ]


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #


def formulario(valores: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Dibuja el formulario clínico y devuelve los datos y si se pulsó calcular."""
    datos: dict[str, Any] = {}

    with st.form("paciente"):
        st.subheader("Datos del paciente")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demografía**")
            datos["age"] = st.number_input(
                "Edad (años)",
                min_value=int(RANGOS_VALIDOS["age"][0]),
                max_value=int(RANGOS_VALIDOS["age"][1]),
                value=int(valores["age"]),
            )
            datos["sex"] = st.selectbox(
                "Sexo",
                CATEGORIAS_VALIDAS["sex"],
                index=CATEGORIAS_VALIDAS["sex"].index(valores["sex"]),
                format_func=lambda v: ETIQUETAS["sex"][v],
            )
            datos["fbs"] = st.selectbox(
                "Glucemia en ayunas > 120 mg/dl",
                ["0", "1"],
                index=int(valores["fbs"]),
                format_func=lambda v: "Sí" if v == "1" else "No",
            )

        with col2:
            st.markdown("**Síntomas**")
            datos["chest_pain"] = st.selectbox(
                "Tipo de dolor torácico",
                CATEGORIAS_VALIDAS["chest_pain"],
                index=CATEGORIAS_VALIDAS["chest_pain"].index(valores["chest_pain"]),
                format_func=lambda v: ETIQUETAS["chest_pain"][v],
            )
            datos["exang"] = st.selectbox(
                "Angina inducida por ejercicio",
                ["0", "1"],
                index=int(valores["exang"]),
                format_func=lambda v: ETIQUETAS["exang"][v],
            )
            datos["max_hr"] = st.number_input(
                "Frecuencia cardíaca máxima (lpm)",
                min_value=int(RANGOS_VALIDOS["max_hr"][0]),
                max_value=int(RANGOS_VALIDOS["max_hr"][1]),
                value=int(valores["max_hr"]),
            )

        with col3:
            st.markdown("**Pruebas**")
            datos["rest_bp"] = st.number_input(
                "Presión arterial en reposo (mm Hg)",
                min_value=int(RANGOS_VALIDOS["rest_bp"][0]),
                max_value=int(RANGOS_VALIDOS["rest_bp"][1]),
                value=int(valores["rest_bp"]),
            )
            datos["chol"] = st.number_input(
                "Colesterol sérico (mg/dl)",
                min_value=int(RANGOS_VALIDOS["chol"][0]),
                max_value=int(RANGOS_VALIDOS["chol"][1]),
                value=int(valores["chol"]),
            )
            datos["old_peak"] = st.number_input(
                "Depresión del ST (mm)",
                min_value=float(RANGOS_VALIDOS["old_peak"][0]),
                max_value=float(RANGOS_VALIDOS["old_peak"][1]),
                value=float(valores["old_peak"]),
                step=0.1,
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            datos["rest_ecg"] = st.selectbox(
                "Electrocardiograma en reposo",
                CATEGORIAS_VALIDAS["rest_ecg"],
                index=CATEGORIAS_VALIDAS["rest_ecg"].index(valores["rest_ecg"]),
                format_func=lambda v: ETIQUETAS["rest_ecg"][v],
            )
        with col5:
            datos["slope"] = st.selectbox(
                "Pendiente del segmento ST",
                CATEGORIAS_VALIDAS["slope"],
                index=CATEGORIAS_VALIDAS["slope"].index(valores["slope"]),
                format_func=lambda v: ETIQUETAS["slope"][v],
            )
        with col6:
            datos["thal"] = st.selectbox(
                "Talasemia",
                CATEGORIAS_VALIDAS["thal"],
                index=CATEGORIAS_VALIDAS["thal"].index(valores["thal"]),
                format_func=lambda v: ETIQUETAS["thal"][v],
            )

        datos["ca"] = st.slider(
            "Vasos principales coloreados por fluoroscopia",
            min_value=0,
            max_value=3,
            value=int(valores["ca"]),
        )

        calcular = st.form_submit_button("Calcular probabilidad", type="primary")

    return datos, calcular


def mostrar_resultado(resultado: pd.Series, umbral: float, datos: dict[str, Any]) -> None:
    """Muestra la probabilidad, el veredicto y las señales clínicas presentes."""
    probabilidad = float(resultado["probabilidad_enfermedad"])
    riesgo = str(resultado["nivel_riesgo"])
    color = COLORES_RIESGO[riesgo]

    st.subheader("Resultado")

    izquierda, derecha = st.columns([2, 1])

    with izquierda:
        st.markdown(
            f"<div style='font-size:2.6rem;font-weight:700;color:{color}'>"
            f"{probabilidad:.1%}</div>"
            f"<div style='color:#52514e'>probabilidad estimada de enfermedad coronaria</div>",
            unsafe_allow_html=True,
        )
        st.progress(probabilidad)

    with derecha:
        st.metric("Nivel de riesgo", riesgo.capitalize())
        st.metric("Umbral aplicado", f"{umbral:.2f}")

    if resultado["prediccion"] == 1:
        st.warning(
            f"Por encima del umbral de decisión ({umbral:.2f}): el modelo lo clasifica "
            "como **caso probable**. Requiere valoración clínica."
        )
    else:
        st.success(
            f"Por debajo del umbral de decisión ({umbral:.2f}): el modelo **no** lo "
            "clasifica como caso probable."
        )

    st.markdown("**Señales clínicas de riesgo presentes**")
    for descripcion, presente in senales_de_riesgo(datos):
        st.markdown(f"- {'🔴' if presente else '⚪'} {descripcion}")

    st.caption(
        "Esta demo es un ejercicio académico. No es un dispositivo médico y no "
        "sustituye el criterio de un profesional sanitario."
    )


def barra_lateral(modelo: ModeloCargado) -> float:
    """Panel lateral con la información del modelo y el control del umbral."""
    with st.sidebar:
        st.header("Modelo")
        metadatos = modelo.metadatos
        st.markdown(
            f"- **Tipo:** {metadatos.get('modelo', 'desconocido')}\n"
            f"- **Atributos:** {len(modelo.atributos_esperados)}\n"
            f"- **Entrenado:** {str(metadatos.get('generado_en', 'n/d'))[:10]}\n"
            f"- **scikit-learn:** {metadatos.get('version_sklearn', 'n/d')}"
        )

        metricas = metadatos.get("metricas", {}).get("test", {})
        if metricas:
            st.markdown(
                f"- **F1 (test):** {metricas.get('f1', 0):.3f}\n"
                f"- **Sensibilidad:** {metricas.get('sensibilidad', 0):.3f}\n"
                f"- **ROC AUC:** {metricas.get('roc_auc', 0):.3f}"
            )

        st.divider()
        st.header("Umbral de decisión")
        umbral = st.slider(
            "Probabilidad a partir de la cual se clasifica como caso probable",
            min_value=0.05,
            max_value=0.95,
            value=float(modelo.umbral),
            step=0.01,
        )
        st.caption(
            f"El valor por defecto ({modelo.umbral:.2f}) es el que maximiza el F1 en "
            "validación cruzada. Bajarlo detecta más enfermos a costa de más falsas "
            "alarmas; subirlo hace lo contrario."
        )

        st.divider()
        st.caption(
            "Las transformaciones las aplica el `inference_pipeline` del proyecto, "
            "el mismo que se usa para predecir sobre archivos."
        )

    return umbral


def main() -> None:
    """Punto de entrada de la aplicación."""
    st.set_page_config(
        page_title="Predicción de enfermedad cardíaca",
        page_icon="🫀",
        layout="wide",
    )

    st.title("🫀 Predicción de enfermedad cardíaca")
    st.markdown(
        "Introduce los datos clínicos del paciente y el modelo estimará la "
        "probabilidad de enfermedad coronaria."
    )

    ruta = localizar_modelo()
    if not ruta.is_file():
        st.error(
            f"No se encuentra el modelo en `{ruta}`.\n\n"
            "Ejecuta antes el training pipeline:\n\n"
            "```bash\nuv run python src/pipelines/training_pipeline/train_pipeline.py\n```"
        )
        st.stop()

    modelo = obtener_modelo()
    umbral = barra_lateral(modelo)

    if "valores" not in st.session_state:
        st.session_state.valores = dict(EJEMPLOS["Caso intermedio"])

    st.markdown("**Cargar un caso de ejemplo**")
    columnas = st.columns(len(EJEMPLOS))
    for columna, (nombre, ejemplo) in zip(columnas, EJEMPLOS.items(), strict=True):
        if columna.button(nombre, width="stretch"):
            st.session_state.valores = dict(ejemplo)

    datos, calcular = formulario(st.session_state.valores)

    if calcular:
        st.session_state.valores = dict(datos)
        resultado = predecir_paciente(modelo, datos, umbral)
        mostrar_resultado(resultado, umbral, datos)
    else:
        st.info("Completa el formulario y pulsa **Calcular probabilidad**.")

    with st.expander("Cómo interpretar el resultado"):
        st.markdown(
            f"""
El modelo devuelve una **probabilidad**, no un diagnóstico. El umbral convierte esa
probabilidad en una decisión, y es un parámetro ajustable: no forma parte del modelo.

Las bandas de riesgo son una ayuda de lectura:

| Banda | Probabilidad |
|---|---|
{chr(10).join(f"| {etiqueta} | < {limite:.0%} |" for limite, etiqueta in BANDAS_RIESGO)}

El umbral por defecto ({modelo.umbral:.2f}) es más permisivo que el 0.5 habitual
porque en un tamizaje cardíaco el falso negativo —un enfermo que se va a casa— cuesta
más que el falso positivo —una prueba adicional—.
"""
        )


if __name__ == "__main__":
    main()
