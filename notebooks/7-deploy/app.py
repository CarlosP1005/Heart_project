"""Demo funcional del modelo de predicción de enfermedad cardíaca.

Aplicación de Streamlit que expone el pipeline entrenado en los notebooks del
proyecto a través de un formulario web: el usuario introduce los datos clínicos
de un paciente y el modelo devuelve la probabilidad de enfermedad coronaria.

Ejecutar en local:

    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import preprocesamiento
import sklearn
import streamlit as st

matplotlib.use("Agg")

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

# Las rutas se resuelven respecto a ESTE archivo, no al directorio desde el que se
# lanza el comando. Así la app funciona tanto con `streamlit run app.py` dentro de su
# carpeta como con `streamlit run notebooks/7-deploy/app.py` desde la raíz del repo,
# que es además como la ejecuta Streamlit Community Cloud.
CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[1]

# El modelo se busca en varios sitios, en este orden: junto a la app, en la carpeta
# `models/` del repositorio y en la del notebook que lo generó. Así la demo funciona
# tanto con una copia local del `.joblib` como leyéndolo de su ubicación original.
CARPETAS_MODELO = [
    CARPETA,
    RAIZ / "models",
    CARPETA.parent / "6-interpretation",
]


def localizar(nombre: str) -> Path:
    """Devuelve la primera ruta existente para un archivo del modelo."""
    for carpeta in CARPETAS_MODELO:
        candidato = carpeta / nombre
        if candidato.is_file():
            return candidato
    return CARPETA / nombre


RUTA_MODELO = localizar("modelo_corazon_interpretado.joblib")
RUTA_METADATOS = localizar("modelo_corazon_interpretado_completo.joblib")

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
]

SIN_DATO = "(sin dato)"

# La app exige al menos esta versión del módulo de preprocesamiento.
VERSION_MODULO_REQUERIDA = 3

# Paleta de estado: verde = riesgo bajo, ámbar = zona dudosa, rojo = riesgo alto.
VERDE, AMBAR, ROJO = "#1baf7a", "#eda100", "#e34948"
TINTA, TINTA2, GRIS = "#0b0b0b", "#52514e", "#e8e7e2"

# Valores por defecto: medianas y categorías más frecuentes del dataset de entrenamiento.
PACIENTES_EJEMPLO: dict[str, dict[str, Any]] = {
    "Riesgo bajo": {
        "age": 42,
        "sex": "female",
        "chest_pain": "nonanginal",
        "rest_bp": 120,
        "chol": 200,
        "fbs": "0",
        "rest_ecg": "normal",
        "max_hr": 175,
        "exang": "0",
        "old_peak": 0.0,
        "slope": "1",
        "ca": 0,
        "thal": "normal",
    },
    "Riesgo alto": {
        "age": 65,
        "sex": "male",
        "chest_pain": "asymptomatic",
        "rest_bp": 160,
        "chol": 290,
        "fbs": "1",
        "rest_ecg": "left ventricular hypertrophy",
        "max_hr": 110,
        "exang": "1",
        "old_peak": 3.0,
        "slope": "2",
        "ca": 2,
        "thal": "reversable",
    },
    # Paciente real de la cohorte cuya probabilidad cae dentro de la zona dudosa.
    "Caso ambiguo": {
        "age": 63,
        "sex": "female",
        "chest_pain": "asymptomatic",
        "rest_bp": 124,
        "chol": 197,
        "fbs": "0",
        "rest_ecg": "normal",
        "max_hr": 136,
        "exang": "1",
        "old_peak": 0.0,
        "slope": "2",
        "ca": 0,
        "thal": "normal",
    },
    "Datos incompletos": {
        "age": 58,
        "sex": "male",
        "chest_pain": "asymptomatic",
        "rest_bp": 140,
        "chol": 260,
        "fbs": "0",
        "rest_ecg": SIN_DATO,
        "max_hr": 145,
        "exang": "1",
        "old_peak": 1.5,
        "slope": "2",
        "ca": 1,
        "thal": SIN_DATO,
    },
}

# Opciones de cada desplegable. `fbs` no aparece en CATEGORIAS_VALIDAS porque el
# pipeline la trata como numérica: el saneador convierte "0"/"1" a 0.0/1.0 antes de
# codificarla, así que enviar el texto es correcto.
OPCIONES_CAMPO: dict[str, list[str]] = {
    "sex": ["male", "female"],
    "chest_pain": ["typical", "nontypical", "nonanginal", "asymptomatic"],
    "rest_ecg": ["normal", "left ventricular hypertrophy", "st-t wave abnormality"],
    "exang": ["0", "1"],
    "slope": ["1", "2", "3"],
    "thal": ["normal", "fixed", "reversable"],
    "fbs": ["0", "1"],
}

ETIQUETAS_CATEGORICAS: dict[str, dict[str, str]] = {
    "sex": {"male": "Hombre", "female": "Mujer"},
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
    "thal": {
        "normal": "Perfusión normal",
        "fixed": "Defecto fijo (tejido infartado)",
        "reversable": "Defecto reversible (isquemia)",
    },
    "slope": {"1": "Ascendente", "2": "Plana", "3": "Descendente"},
    "exang": {"0": "No", "1": "Sí"},
    "fbs": {"0": "No (≤ 120 mg/dl)", "1": "Sí (> 120 mg/dl)"},
}


# --------------------------------------------------------------------------- #
# Carga del modelo
# --------------------------------------------------------------------------- #


@st.cache_resource
def cargar_modelo() -> tuple[Any, dict[str, Any]]:
    """Carga el pipeline entrenado y sus metadatos.

    El modelo se serializó desde un notebook, por lo que dentro del `.joblib` las
    clases del pipeline apuntan a `__main__`. `registrar_en_main()` las publica
    ahí antes de deserializar.
    """
    modelo = preprocesamiento.cargar_pipeline(RUTA_MODELO)
    try:
        metadatos = preprocesamiento.cargar_pipeline(RUTA_METADATOS)
    except FileNotFoundError:
        metadatos = {}
    return modelo, metadatos


def etiqueta(columna: str, valor: str) -> str:
    """Traduce el código interno de una categoría a texto legible."""
    return ETIQUETAS_CATEGORICAS.get(columna, {}).get(valor, valor)


def opciones(columna: str) -> list[str]:
    """Devuelve las categorías seleccionables de una columna, más la opción 'sin dato'."""
    return [*OPCIONES_CAMPO[columna], SIN_DATO]


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #


def figura_riesgo(probabilidad: float, banda: tuple[float, float]) -> Any:
    """Dibuja la barra de riesgo con las tres zonas de decisión."""
    inferior, superior = banda
    figura, ejes = plt.subplots(figsize=(8, 1.15))

    ejes.axhspan(0, 1, xmin=0, xmax=inferior, color=VERDE, alpha=0.22)
    ejes.axhspan(0, 1, xmin=inferior, xmax=superior, color=AMBAR, alpha=0.22)
    ejes.axhspan(0, 1, xmin=superior, xmax=1, color=ROJO, alpha=0.22)

    color = ROJO if probabilidad >= superior else AMBAR if probabilidad > inferior else VERDE
    ejes.axvline(probabilidad, color=color, linewidth=4)
    ejes.plot([probabilidad], [1.12], marker="v", markersize=11, color=color, clip_on=False)
    ejes.text(
        probabilidad,
        1.28,
        f"{probabilidad:.0%}",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=color,
    )

    for posicion, texto in [
        (inferior / 2, "riesgo bajo"),
        ((inferior + superior) / 2, "zona dudosa"),
        ((superior + 1) / 2, "riesgo alto"),
    ]:
        ejes.text(posicion, 0.5, texto, ha="center", va="center", fontsize=9, color=TINTA2)

    ejes.set_xlim(0, 1)
    ejes.set_ylim(0, 1)
    ejes.set_yticks([])
    ejes.set_xticks([0, inferior, superior, 1])
    ejes.set_xticklabels(["0 %", f"{inferior:.0%}", f"{superior:.0%}", "100 %"], fontsize=9)
    ejes.tick_params(colors=TINTA2)
    for lado in ("top", "right", "left"):
        ejes.spines[lado].set_visible(False)
    ejes.spines["bottom"].set_color(GRIS)
    figura.subplots_adjust(top=0.66, bottom=0.32, left=0.04, right=0.98)
    return figura


# --------------------------------------------------------------------------- #
# Formulario
# --------------------------------------------------------------------------- #


def campos_demograficos(valores: dict[str, Any]) -> dict[str, Any]:
    """Campos de datos personales y signos vitales en reposo."""
    datos: dict[str, Any] = {}
    st.markdown("**Datos del paciente**")
    datos["age"] = st.number_input(
        "Edad (años)", min_value=18, max_value=100, value=int(valores["age"]), step=1
    )
    datos["sex"] = st.selectbox(
        "Sexo",
        opciones("sex"),
        index=opciones("sex").index(valores["sex"]),
        format_func=lambda v: etiqueta("sex", v),
    )
    datos["rest_bp"] = st.number_input(
        "Presión arterial en reposo (mm Hg)",
        min_value=80,
        max_value=220,
        value=int(valores["rest_bp"]),
        step=1,
    )
    datos["chol"] = st.number_input(
        "Colesterol sérico (mg/dl)",
        min_value=100,
        max_value=600,
        value=int(valores["chol"]),
        step=1,
    )
    datos["fbs"] = st.selectbox(
        "Glucemia en ayunas > 120 mg/dl",
        opciones("fbs"),
        index=opciones("fbs").index(str(valores["fbs"])),
        format_func=lambda v: etiqueta("fbs", v),
    )
    return datos


def campos_sintomas(valores: dict[str, Any]) -> dict[str, Any]:
    """Campos de sintomatología y electrocardiograma en reposo."""
    datos: dict[str, Any] = {}
    st.markdown("**Síntomas y ECG**")
    datos["chest_pain"] = st.selectbox(
        "Tipo de dolor torácico",
        opciones("chest_pain"),
        index=opciones("chest_pain").index(valores["chest_pain"]),
        format_func=lambda v: etiqueta("chest_pain", v),
        help="'Asintomático' es, contra la intuición, la categoría de mayor riesgo: angina silente.",
    )
    datos["rest_ecg"] = st.selectbox(
        "Electrocardiograma en reposo",
        opciones("rest_ecg"),
        index=opciones("rest_ecg").index(valores["rest_ecg"]),
        format_func=lambda v: etiqueta("rest_ecg", v),
    )
    datos["exang"] = st.selectbox(
        "Angina inducida por el ejercicio",
        opciones("exang"),
        index=opciones("exang").index(str(valores["exang"])),
        format_func=lambda v: etiqueta("exang", v),
    )
    datos["max_hr"] = st.number_input(
        "Frecuencia cardíaca máxima alcanzada",
        min_value=60,
        max_value=220,
        value=int(valores["max_hr"]),
        step=1,
        help="Alcanzar una frecuencia alta en la prueba de esfuerzo indica buena reserva cardíaca.",
    )
    return datos


def campos_pruebas(valores: dict[str, Any]) -> dict[str, Any]:
    """Campos de prueba de esfuerzo, fluoroscopia y gammagrafía."""
    datos: dict[str, Any] = {}
    st.markdown("**Prueba de esfuerzo e imagen**")
    datos["old_peak"] = st.number_input(
        "Depresión del segmento ST (old_peak)",
        min_value=0.0,
        max_value=7.0,
        value=float(valores["old_peak"]),
        step=0.1,
        format="%.1f",
    )
    datos["slope"] = st.selectbox(
        "Pendiente del segmento ST en esfuerzo máximo",
        opciones("slope"),
        index=opciones("slope").index(str(valores["slope"])),
        format_func=lambda v: etiqueta("slope", v),
    )
    datos["ca"] = st.select_slider(
        "Vasos principales afectados (fluoroscopia)",
        options=[0, 1, 2, 3],
        value=int(valores["ca"]),
    )
    datos["thal"] = st.selectbox(
        "Gammagrafía de perfusión (thal)",
        opciones("thal"),
        index=opciones("thal").index(valores["thal"]),
        format_func=lambda v: etiqueta("thal", v),
        help="Es la variable más influyente del modelo. Su ausencia triplica el riesgo de error.",
    )
    return datos


def formulario_paciente(valores: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Renderiza el formulario completo y devuelve los datos y si se ha enviado."""
    with st.form("formulario_paciente"):
        columna_a, columna_b, columna_c = st.columns(3)
        datos: dict[str, Any] = {}
        with columna_a:
            datos.update(campos_demograficos(valores))
        with columna_b:
            datos.update(campos_sintomas(valores))
        with columna_c:
            datos.update(campos_pruebas(valores))

        sin_medir = st.multiselect(
            "Marcar mediciones numéricas como no disponibles",
            ["age", "rest_bp", "chol", "max_hr", "old_peak", "ca"],
            help=(
                "El pipeline imputa los valores que faltan, así que la predicción se emite "
                "igualmente. Sirve para comprobar cómo se degrada."
            ),
        )
        enviado = st.form_submit_button("Calcular predicción", type="primary")

    for columna in sin_medir:
        datos[columna] = None
    for columna, valor in datos.items():
        if valor == SIN_DATO:
            datos[columna] = None
    return datos, enviado


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #


def senales_de_riesgo(datos: dict[str, Any]) -> list[tuple[str, bool]]:
    """Evalúa las cuatro señales clínicas del modelo base heurístico."""
    return [
        ("Defecto de perfusión miocárdica", datos["thal"] not in (None, "normal")),
        ("Angina silente (asintomático)", datos["chest_pain"] == "asymptomatic"),
        ("Al menos un vaso principal afectado", (datos["ca"] or 0) >= 1),
        ("Angina inducida por el ejercicio", datos["exang"] == "1"),
    ]


def mostrar_veredicto(probabilidad: float, umbral: float, banda: tuple[float, float]) -> None:
    """Muestra el mensaje de resultado según la zona en la que cae la probabilidad."""
    inferior, superior = banda
    if inferior < probabilidad < superior:
        st.warning(
            f"**Zona dudosa — requiere valoración clínica.** El modelo asigna un "
            f"{probabilidad:.1%} de probabilidad, dentro de la banda en la que sus predicciones "
            f"son poco fiables: ahí acierta bastante menos que fuera de ella. "
            f"La recomendación es no automatizar esta decisión.",
            icon="⚠️",
        )
    elif probabilidad >= umbral:
        st.error(
            f"**Indicios de enfermedad coronaria.** Probabilidad estimada: {probabilidad:.1%} "
            f"(umbral de decisión: {umbral:.0%}). Se recomienda estudio cardiológico.",
            icon="🔴",
        )
    else:
        st.success(
            f"**Sin indicios de enfermedad coronaria.** Probabilidad estimada: "
            f"{probabilidad:.1%} (umbral de decisión: {umbral:.0%}).",
            icon="🟢",
        )


def mostrar_resultado(
    modelo: Any,
    datos: dict[str, Any],
    umbral: float,
    banda: tuple[float, float],
) -> None:
    """Ejecuta la predicción y presenta el resultado completo."""
    fila = pd.DataFrame([{columna: datos.get(columna) for columna in COLUMNAS}])
    probabilidad = float(modelo.predict_proba(fila)[0, 1])

    st.subheader("Resultado")
    st.pyplot(figura_riesgo(probabilidad, banda), use_container_width=True)
    mostrar_veredicto(probabilidad, umbral, banda)

    columna_a, columna_b = st.columns([1, 1])

    with columna_a:
        st.markdown("**Señales clínicas de riesgo presentes**")
        for nombre, presente in senales_de_riesgo(datos):
            st.markdown(f"{'🔺' if presente else '▫️'} {nombre}")

    with columna_b:
        st.markdown("**Calidad de los datos introducidos**")
        faltantes = [columna for columna in COLUMNAS if datos.get(columna) is None]
        if not faltantes:
            st.markdown("✅ Registro completo: ninguna variable imputada.")
        else:
            st.markdown(f"⚠️ {len(faltantes)} variable(s) imputada(s): `{'`, `'.join(faltantes)}`")
        if datos.get("thal") is None:
            st.markdown(
                "🔻 **Falta `thal`**, la variable más influyente. En el análisis de errores, "
                "los registros sin este dato se clasifican mal casi tres veces más a menudo."
            )

    with st.expander("Ver los datos enviados al modelo"):
        st.dataframe(fila.T.rename(columns={0: "valor"}), use_container_width=True)


# --------------------------------------------------------------------------- #
# Pestañas
# --------------------------------------------------------------------------- #


def pestana_individual(modelo: Any, umbral: float, banda: tuple[float, float]) -> None:
    """Pestaña del formulario de predicción individual."""
    st.caption("Rellena los datos del paciente y pulsa **Calcular predicción**.")

    columnas = st.columns(len(PACIENTES_EJEMPLO) + 1)
    for columna, nombre in zip(columnas, PACIENTES_EJEMPLO, strict=False):
        with columna:
            if st.button(nombre, use_container_width=True):
                st.session_state["valores"] = PACIENTES_EJEMPLO[nombre]

    valores = st.session_state.get("valores", PACIENTES_EJEMPLO["Caso ambiguo"])
    datos, enviado = formulario_paciente(valores)

    if enviado:
        mostrar_resultado(modelo, datos, umbral, banda)
    else:
        st.info(
            "Los botones de arriba cargan casos de ejemplo en el formulario. "
            "«Datos incompletos» muestra cómo responde el modelo cuando faltan mediciones.",
            icon="💡",
        )


def pestana_lote(modelo: Any, umbral: float) -> None:
    """Pestaña de predicción sobre un archivo CSV."""
    st.caption(
        "Sube un CSV con una fila por paciente. Debe tener las 13 columnas del dataset "
        "original; los valores no interpretables y los huecos se sanean automáticamente."
    )
    st.code(", ".join(COLUMNAS), language=None)

    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is None:
        return

    datos = pd.read_csv(archivo, dtype=str)
    faltan = [columna for columna in COLUMNAS if columna not in datos.columns]
    if faltan:
        st.error(f"Al archivo le faltan estas columnas: `{'`, `'.join(faltan)}`")
        return

    probabilidades = modelo.predict_proba(datos[COLUMNAS])[:, 1]
    resultado = datos.copy()
    resultado.insert(0, "prediccion", (probabilidades >= umbral).astype(int))
    resultado.insert(0, "probabilidad", probabilidades.round(4))

    st.success(f"{len(resultado)} pacientes procesados.")
    columna_a, columna_b, columna_c = st.columns(3)
    columna_a.metric("Con indicios", int(resultado["prediccion"].sum()))
    columna_b.metric("Sin indicios", int((resultado["prediccion"] == 0).sum()))
    columna_c.metric("Probabilidad media", f"{probabilidades.mean():.1%}")

    st.dataframe(resultado, use_container_width=True, height=320)

    memoria = io.StringIO()
    resultado.to_csv(memoria, index=False)
    st.download_button(
        "Descargar resultados en CSV",
        memoria.getvalue(),
        file_name="predicciones.csv",
        mime="text/csv",
    )


def pestana_modelo(metadatos: dict[str, Any]) -> None:
    """Pestaña informativa sobre el modelo y sus limitaciones."""
    st.markdown(
        """
### Cómo se construyó

El modelo es el resultado de cuatro etapas documentadas en los notebooks del proyecto:

1. **Feature engineering** — saneamiento de un dataset con un 81 % de duplicados y valores
   corruptos, y un pipeline de 9 ramas que convierte 13 variables clínicas en 44 atributos.
2. **Modelo base** — una heurística de cuatro reglas clínicas que fija el listón a superar.
3. **Selección de modelo** — 12 candidatos de 7 familias, criba por rendimiento, optimización
   de hiperparámetros en dos rondas y comparación con la prueba t pareada corregida de
   Nadeau-Bengio.
4. **Interpretación y análisis de errores** — de dónde vienen los fallos y cuándo no fiarse.

### Qué mira el modelo

Tres variables concentran cerca de la mitad de su decisión: la **gammagrafía de perfusión**
(`thal`), el **número de vasos afectados** (`ca`) y el **tipo de dolor torácico**
(`chest_pain`). La **depresión del segmento ST** (`old_peak`) es la que más se echa en falta
cuando se elimina, porque aporta información que ninguna otra duplica.

### Limitaciones que conviene conocer

- **La zona dudosa existe y es real.** Entre el 35 % y el 65 % de probabilidad el modelo se
  equivoca con mucha más frecuencia. Ahí la respuesta correcta es derivar, no automatizar.
- **Necesita `thal` y `ca`**, que provienen de pruebas de imagen caras. Sirve como apoyo a la
  interpretación de esas pruebas, **no como cribado para evitarlas**.
- **Entrenado con 480 pacientes de una sola cohorte** (Cleveland, base UCI). No está validado
  en otra población.
- **El umbral de decisión es una decisión clínica**, no estadística. El de la barra lateral se
  puede mover: bajarlo detecta más enfermos a costa de derivar más sanos.
"""
    )

    if metadatos:
        st.markdown("### Metadatos guardados junto al modelo")
        tabla = pd.DataFrame(
            [
                {"clave": clave, "valor": str(valor)}
                for clave, valor in metadatos.items()
                if clave != "pipeline"
            ]
        )
        st.dataframe(tabla, use_container_width=True, hide_index=True)


def barra_lateral(metadatos: dict[str, Any]) -> tuple[float, tuple[float, float]]:
    """Renderiza la barra lateral y devuelve el umbral y la banda de abstención."""
    st.sidebar.header("Configuración")

    banda_guardada = metadatos.get("banda_abstencion", [0.35, 0.65])
    umbral_defecto = float(metadatos.get("umbral_defecto", 0.5))

    umbral = st.sidebar.slider(
        "Umbral de decisión",
        min_value=0.05,
        max_value=0.95,
        value=umbral_defecto,
        step=0.01,
        help=(
            "Probabilidad a partir de la cual se clasifica como enfermo. Bajarlo aumenta la "
            "sensibilidad (menos enfermos sin detectar) a costa de más falsos positivos."
        ),
    )

    banda = (float(banda_guardada[0]), float(banda_guardada[1]))
    mostrar_banda = st.sidebar.checkbox("Señalar la zona dudosa", value=True)
    if not mostrar_banda:
        banda = (0.0, 0.0)

    st.sidebar.divider()
    st.sidebar.subheader("Rendimiento del modelo")
    st.sidebar.metric("F1 en el conjunto de prueba", f"{float(metadatos.get('f1_test', 0)):.3f}")
    st.sidebar.caption(
        f"Modelo: **{metadatos.get('nombre_modelo', 'no disponible')}**  \n"
        f"scikit-learn del entrenamiento: `{metadatos.get('version_sklearn', '?')}`  \n"
        f"scikit-learn en ejecución: `{sklearn.__version__}`  \n"
        f"módulo de preprocesamiento: `v{getattr(preprocesamiento, 'VERSION_MODULO', 1)}`  \n"
        f"modelo: `{RUTA_MODELO.parent.name}/{RUTA_MODELO.name}`"
    )
    if metadatos.get("version_sklearn") not in (None, sklearn.__version__):
        st.sidebar.warning(
            "La versión de scikit-learn no coincide con la del entrenamiento. "
            "El modelo puede comportarse de forma distinta.",
            icon="⚠️",
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "**Demo académica.** No es un dispositivo médico y no debe usarse para tomar "
        "decisiones clínicas sobre personas reales."
    )
    return umbral, banda


# --------------------------------------------------------------------------- #
# Aplicación
# --------------------------------------------------------------------------- #


def main() -> None:
    """Punto de entrada de la aplicación."""
    st.set_page_config(
        page_title="Predicción de enfermedad cardíaca",
        page_icon="🫀",
        layout="wide",
    )

    st.title("🫀 Predicción de enfermedad cardíaca")
    st.caption(
        "Demo del modelo entrenado sobre la cohorte Cleveland del dataset UCI Heart Disease."
    )

    if getattr(preprocesamiento, "VERSION_MODULO", 1) < VERSION_MODULO_REQUERIDA:
        st.error(
            "Estás usando una copia antigua de `preprocesamiento.py`. Reemplázala por la "
            "versión actual **y reinicia el servidor** (`Ctrl+C` y volver a lanzar): "
            "Streamlit no recarga los módulos ya importados al guardar un archivo.\n\n"
            f"Módulo cargado desde: `{preprocesamiento.__file__}`",
            icon="🚫",
        )
        return

    try:
        modelo, metadatos = cargar_modelo()
    except FileNotFoundError:
        buscado = "\n".join(f"- `{carpeta}`" for carpeta in CARPETAS_MODELO)
        st.error(
            f"No se encuentra `{RUTA_MODELO.name}`. Se ha buscado en:\n\n{buscado}\n\n"
            "Copia ahí el `.joblib` generado en el notebook de interpretación.",
            icon="🚫",
        )
        return
    except (ModuleNotFoundError, ImportError) as error:
        st.error(
            f"El modelo no es compatible con la versión de scikit-learn instalada "
            f"(`{sklearn.__version__}`): falta `{error.name}`.\n\n"
            "Se guardó con otra versión, cuya estructura interna es distinta. "
            "La solución más directa es usar **tu propio** `.joblib`, el que generó tu "
            "notebook en este mismo entorno:\n\n"
            "```\ncp notebooks/6-interpretation/*.joblib notebooks/7-deploy/\n```\n\n"
            "Para ver el detalle completo: `uv run python notebooks/7-deploy/diagnostico.py`",
            icon="🚫",
        )
        return

    umbral, banda = barra_lateral(metadatos)

    individual, lote, informacion = st.tabs(
        ["Predicción individual", "Predicción por lotes", "Sobre el modelo"]
    )
    with individual:
        pestana_individual(modelo, umbral, banda)
    with lote:
        pestana_lote(modelo, umbral)
    with informacion:
        pestana_modelo(metadatos)


if __name__ == "__main__":
    main()
