"""Diagnóstico de compatibilidad entre el modelo serializado y el entorno.

Responde a la pregunta que casi siempre está detrás de un fallo al cargar un
`.joblib`: ¿qué necesita este archivo y qué falta en este entorno?

Uso:

    uv run python notebooks/7-deploy/diagnostico.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy
import preprocesamiento
import sklearn
from joblib.numpy_pickle import NumpyUnpickler

CARPETA = Path(__file__).resolve().parent
ARCHIVOS = [
    "modelo_corazon_interpretado.joblib",
    "modelo_corazon_interpretado_completo.joblib",
]


class _Espia(NumpyUnpickler):  # type: ignore[misc]
    """Unpickler que anota cada clase referenciada antes de resolverla."""

    referencias: ClassVar[set[tuple[str, str]]] = set()

    def find_class(self, module: str, name: str) -> Any:
        """Registra la referencia y delega en el comportamiento normal."""
        _Espia.referencias.add((module, name))
        return super().find_class(module, name)


def referencias_de(ruta: Path) -> set[tuple[str, str]]:
    """Devuelve las clases que el archivo necesita, aunque la carga falle."""
    _Espia.referencias = set()
    try:
        with ruta.open("rb") as archivo:
            _Espia(str(ruta), archivo, ensure_native_byte_order=True).load()
    except Exception as error:
        print(f"    (la carga se detuvo en: {type(error).__name__}: {error})")
    return set(_Espia.referencias)


def revisar(ruta: Path) -> bool:
    """Informa del estado de un archivo y devuelve si se pudo cargar."""
    print(f"\n{'=' * 72}\n{ruta.name}\n{'=' * 72}")
    if not ruta.exists():
        print("  NO EXISTE en esta carpeta.")
        return False

    print(f"  Tamaño: {ruta.stat().st_size / 1024:.0f} KB")

    preprocesamiento.registrar_en_main()
    preprocesamiento.alias_modulos_compilados()
    referencias = referencias_de(ruta)

    faltantes = []
    for modulo, nombre in sorted(referencias):
        if modulo == "__main__":
            continue
        try:
            importlib.import_module(modulo)
        except ImportError:
            faltantes.append((modulo, nombre))

    print(f"  Clases referenciadas: {len(referencias)}")
    if faltantes:
        print("\n  NO SE ENCUENTRAN estos módulos en el entorno:")
        for modulo, nombre in faltantes:
            print(f"     - {modulo}.{nombre}")
        print(
            "\n  Es una incompatibilidad de versión: el modelo se guardó con una\n"
            "  versión de scikit-learn cuya estructura interna no coincide con la\n"
            "  instalada aquí."
        )
    else:
        print("  Todos los módulos referenciados existen en este entorno.")

    try:
        preprocesamiento.cargar_pipeline(ruta)
    except Exception as error:
        print(f"\n  RESULTADO: NO CARGA -> {type(error).__name__}: {error}")
        return False
    print("\n  RESULTADO: CARGA CORRECTAMENTE")
    return True


def main() -> None:
    """Ejecuta el diagnóstico completo."""
    print("Entorno de ejecución")
    print(f"  python       : {sys.version.split()[0]}")
    print(f"  scikit-learn : {sklearn.__version__}")
    print(f"  numpy        : {numpy.__version__}")
    print(f"  joblib       : {joblib.__version__}")
    print(f"  preprocesamiento : v{preprocesamiento.VERSION_MODULO}")
    print(f"                     {preprocesamiento.__file__}")

    resultados = [revisar(CARPETA / nombre) for nombre in ARCHIVOS]

    print(f"\n{'=' * 72}")
    if all(resultados):
        print("Todo correcto: la aplicación debería arrancar sin problemas.")
    else:
        print(
            "Qué hacer si algún archivo no carga:\n"
            "  1. Usa TU propio modelo, el que generó tu notebook en este mismo\n"
            "     entorno:  cp notebooks/6-interpretation/*.joblib notebooks/7-deploy/\n"
            "  2. O instala la versión de scikit-learn con la que se guardó el\n"
            "     modelo y fíjala en requirements.txt."
        )


if __name__ == "__main__":
    main()
