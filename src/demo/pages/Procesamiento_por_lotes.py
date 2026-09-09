"""Página de procesamiento por lotes de la demo.

Recibe un archivo con varios pacientes, predice sobre todos y deja el resultado
visible en pantalla y descargable como CSV.

Se abre desde la barra lateral de la demo principal:

    uv run streamlit run src/demo/app.py

o directamente, si sólo interesa el lote:

    uv run streamlit run src/demo/pages/Procesamiento_por_lotes.py

Criterio de diseño
------------------
Este archivo son **sólo widgets**. La lectura del archivo, la validación, la
predicción y el resumen viven en `demo/lote.py`, que no importa Streamlit y por eso
se puede probar con pytest. Lo que se ve aquí no debería contener ninguna decisión
que merezca una prueba.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from demo import lote  # noqa: E402
from demo.app import localizar_modelo, obtener_modelo  # noqa: E402
from pipelines.inference_pipeline.inference_pipeline import (  # noqa: E402
    COLS_REQUERIDAS,
    ErrorDeValidacion,
    ModeloCargado,
)

#: Alto de la tabla de resultados, en píxeles.
ALTO_TABLA = 420


def cabecera() -> None:
    """Título y explicación de qué hace la página."""
    st.title("📄 Procesamiento por lotes")
    st.markdown(
        "Sube un archivo con **varios pacientes** y obtén la predicción de todos de "
        "una vez. El resultado se muestra en pantalla y se puede descargar en CSV."
    )


def panel_de_ayuda() -> None:
    """Formato esperado del archivo y plantilla descargable."""
    with st.expander("Qué formato debe tener el archivo", expanded=False):
        st.markdown(
            f"""
Una fila por paciente y, como mínimo, estas **13 columnas** con estos nombres exactos:

`{"`, `".join(COLS_REQUERIDAS)}`

Detalles que conviene saber:

- **Formatos aceptados:** `.csv`, `.xlsx` y `.parquet`. Máximo {lote.MAX_FILAS:,} filas.
- **Columnas de más:** se conservan en la salida pero no entran al modelo. Si el
  archivo trae la etiqueta real (`disease`), sirve para comparar lo predicho con lo
  observado.
- **Identificador:** si hay una columna `id_paciente`, `id` o `paciente`, se usa para
  etiquetar cada fila del resultado. Si no, se numeran de 1 en adelante.
- **Valores ausentes:** se pueden dejar vacíos. El modelo los imputa con la mediana
  aprendida en el entrenamiento, igual que hace el `inference_pipeline`.
"""
        )
        st.download_button(
            "⬇️ Descargar plantilla de entrada",
            data=lote.a_csv(lote.plantilla()),
            file_name="plantilla_entrada.csv",
            mime="text/csv",
        )


def barra_lateral(modelo: ModeloCargado) -> float:
    """Información del modelo y control del umbral, como en la demo individual."""
    with st.sidebar:
        st.header("Modelo")
        metadatos = modelo.metadatos
        st.markdown(
            f"- **Tipo:** {metadatos.get('modelo', 'desconocido')}\n"
            f"- **Atributos:** {len(modelo.atributos_esperados)}\n"
            f"- **scikit-learn:** {metadatos.get('version_sklearn', 'n/d')}"
        )

        st.divider()
        st.header("Umbral de decisión")
        # Anotación explícita: el hook de mypy corre sin streamlit instalado y ve el
        # retorno del widget como `Any`, lo que dispararía `no-any-return`.
        umbral: float = st.slider(
            "Probabilidad a partir de la cual se clasifica como caso probable",
            min_value=0.05,
            max_value=0.95,
            value=float(modelo.umbral),
            step=0.01,
        )
        st.caption(
            "Cambiar el umbral vuelve a clasificar el lote entero sin volver a "
            "subirlo: las probabilidades no cambian, sólo el corte."
        )

    return umbral


def mostrar_resumen(resumen: dict[str, Any], predicciones: pd.DataFrame, umbral: float) -> None:
    """Cifras agregadas del lote: tarjetas, bandas de riesgo e histograma."""
    st.subheader("Resumen del lote")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pacientes", f"{resumen['filas']:,}")
    col2.metric("Casos probables", f"{resumen['casos_probables']:,}")
    col3.metric("Proporción", f"{float(resumen['proporcion_positivos']):.1%}")
    col4.metric("Probabilidad media", f"{float(resumen['probabilidad_media']):.1%}")

    st.markdown("**Pacientes por banda de riesgo**")
    # Las bandas se recorren en el orden de `BANDAS_RIESGO`, de menor a mayor. Un
    # gráfico de barras las ordenaría alfabéticamente —'alto' antes que 'bajo'— y la
    # lectura de un vistazo dejaría de funcionar.
    columnas = st.columns(len(resumen["por_banda_de_riesgo"]))
    for columna, (banda, cuantos) in zip(
        columnas, resumen["por_banda_de_riesgo"].items(), strict=True
    ):
        columna.metric(banda.capitalize(), f"{cuantos:,}")

    st.markdown("**Distribución de la probabilidad estimada**")
    st.caption(
        f"Cada barra es un tramo de 10 puntos de probabilidad. El umbral aplicado es "
        f"{umbral:.2f}: todo lo que cae a su derecha se clasifica como caso probable."
    )
    tramos = pd.cut(
        predicciones["probabilidad_enfermedad"].astype(float),
        bins=[i / 10 for i in range(11)],
        labels=[f"{i * 10}-{i * 10 + 10}%" for i in range(10)],
        include_lowest=True,
    )
    st.bar_chart(tramos.value_counts().sort_index().rename("pacientes"))


def mostrar_tabla(predicciones: pd.DataFrame) -> None:
    """Tabla de resultados, con filtro por banda de riesgo."""
    st.subheader("Predicciones")

    bandas = list(dict.fromkeys(predicciones["nivel_riesgo"]))
    seleccion = st.multiselect(
        "Filtrar por banda de riesgo",
        options=bandas,
        default=bandas,
        help="Sólo afecta a lo que se ve; la descarga incluye siempre el lote completo.",
    )
    visible = predicciones[predicciones["nivel_riesgo"].isin(seleccion)]

    columnas_primero = [lote.COL_ID, *lote.COLS_PREDICCION]
    resto = [col for col in visible.columns if col not in columnas_primero]

    st.dataframe(
        visible[[*columnas_primero, *resto]],
        height=ALTO_TABLA,
        width="stretch",
        hide_index=True,
        column_config={
            "probabilidad_enfermedad": st.column_config.ProgressColumn(
                "Probabilidad",
                min_value=0.0,
                max_value=1.0,
                # `percent` multiplica por 100 al mostrar. Un `"%.1f%%"` escribiría
                # "0.2%" para una probabilidad de 0.16: el símbolo sin la escala.
                format="percent",
            )
        },
    )
    st.caption(f"Mostrando {len(visible):,} de {len(predicciones):,} pacientes.")


def descargas(predicciones: pd.DataFrame, nombre_entrada: str) -> None:
    """Botón de descarga del lote completo."""
    nombre = f"predicciones_{Path(nombre_entrada).stem}.csv"
    st.download_button(
        "⬇️ Descargar predicciones (CSV)",
        data=lote.a_csv(predicciones),
        file_name=nombre,
        mime="text/csv",
        type="primary",
    )
    st.caption(
        "El CSV incluye el archivo original completo más las cuatro columnas de "
        "predicción, para que sea auditable sin cruzarlo con la entrada."
    )


def main() -> None:
    """Punto de entrada de la página."""
    st.set_page_config(page_title="Procesamiento por lotes", page_icon="📄", layout="wide")

    cabecera()

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

    panel_de_ayuda()

    subido = st.file_uploader(
        "Archivo de pacientes",
        type=lote.EXTENSIONES_ACEPTADAS,
        help="Una fila por paciente, con las 13 columnas obligatorias.",
    )

    if subido is None:
        st.info("Sube un archivo para empezar. Puedes usar la plantilla de arriba.")
        return

    try:
        datos = lote.leer_archivo(subido.name, subido.getvalue())
        ignoradas = lote.validar_columnas(datos)
        predicciones = lote.predecir_lote(modelo, datos, umbral)
    except ErrorDeValidacion as error:
        # Un lote a medias es peor que ninguno: se muestra el motivo y no se
        # produce ninguna descarga.
        st.error(f"**No se pudo procesar el archivo**\n\n{error}")
        return

    st.success(f"Procesados {len(predicciones):,} pacientes de **{subido.name}**.")
    if ignoradas:
        st.caption(
            "Columnas que el modelo no usa y que se conservan en la salida: "
            f"`{'`, `'.join(ignoradas)}`."
        )

    mostrar_resumen(lote.resumen_lote(predicciones, umbral), predicciones, umbral)
    descargas(predicciones, subido.name)
    mostrar_tabla(predicciones)

    st.caption(
        "Esta demo es un ejercicio académico. No es un dispositivo médico y no "
        "sustituye el criterio de un profesional sanitario."
    )


main()
