# Demo — Predicción de enfermedad cardíaca

Aplicación web construida con [Streamlit](https://streamlit.io/) que expone el modelo
entrenado en los notebooks del proyecto. El usuario introduce los datos clínicos de un
paciente en un formulario y el modelo devuelve la probabilidad de enfermedad coronaria.

## Contenido de la carpeta

| Archivo | Qué es |
|---|---|
| `app.py` | La aplicación de Streamlit |
| `preprocesamiento.py` | Los transformadores del pipeline, necesarios para deserializar el modelo |
| `modelo_corazon_interpretado.joblib` | Pipeline completo entrenado (preprocesamiento + modelo) |
| `modelo_corazon_interpretado_completo.joblib` | Metadatos: umbrales, métricas, advertencias |
| `ejemplo_lote.csv` | CSV de muestra para probar la predicción por lotes |
| `diagnostico.py` | Comprueba si el modelo es compatible con el entorno |
| `requirements.txt` | Dependencias |
| `.streamlit/config.toml` | Tema visual |

## Ejecutar en local

Desde la raíz del repositorio, con `uv`, que es lo que usa el proyecto:

```bash
uv pip install -r notebooks/7-deploy/requirements.txt
uv run streamlit run notebooks/7-deploy/app.py
```

Se abre en <http://localhost:8501>.

Las rutas de los `.joblib` se resuelven respecto a `app.py` (`Path(__file__).parent`), no
respecto al directorio desde el que lanzas el comando. Da igual desde dónde ejecutes, y es
lo que hace que también funcione en Streamlit Cloud, que arranca la app desde la raíz del
repositorio.

## Qué hace la demo

**Predicción individual.** Formulario con las 13 variables clínicas, agrupadas en tres
bloques. Cuatro botones cargan casos de ejemplo, incluido uno con datos incompletos para
ver cómo responde el modelo cuando faltan mediciones. El resultado muestra:

- La probabilidad sobre una barra con las tres zonas de decisión.
- Un veredicto que **distingue la zona dudosa**: entre el 35 % y el 65 % el modelo se
  equivoca mucho más a menudo, así que ahí la app recomienda valoración clínica en lugar
  de dar una respuesta binaria. Sale del análisis de errores del notebook 7.
- Las cuatro señales clínicas de riesgo presentes.
- Un aviso de calidad de datos: qué variables se han imputado y, en particular, si falta
  `thal` — los registros sin ese dato se clasifican mal casi tres veces más a menudo.

**Predicción por lotes.** Se sube un CSV con las 13 columnas y se descargan las
predicciones. El pipeline sanea los valores corruptos e imputa los que faltan, así que
acepta archivos tan sucios como el original: `ejemplo_lote.csv` incluye a propósito una
fila con basura para comprobarlo.

**Sobre el modelo.** Cómo se construyó, en qué se apoya para decidir y sus limitaciones.

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta al repositorio de GitHub (el `.joblib` ocupa ~160 KB, entra sin
   problemas; si superara los 100 MB haría falta Git LFS). El `requirements.txt` debe
   quedar junto a `app.py`; si Streamlit Cloud no lo detecta, copia también uno a la raíz
   del repositorio.
2. Entra en <https://share.streamlit.io> y conecta la cuenta de GitHub.
3. **New app** → elige el repositorio, la rama y como *Main file path* la ruta al archivo
   principal, por ejemplo `notebooks/7-deploy/app.py`.
4. Deploy. Streamlit Cloud instala `requirements.txt` automáticamente.

La URL pública queda con el formato `https://<usuario>-<repo>-<hash>.streamlit.app`.

## Dos detalles técnicos que conviene conocer

### Por qué existe `preprocesamiento.py`

`joblib` **no guarda el código** de las clases personalizadas dentro del `.joblib`, solo
una referencia. Como el modelo se serializó desde un notebook, dentro del archivo las
referencias apuntan a `__main__.SaneadorTipos`, `__main__.a_texto`, etc.

Al cargarlo desde la aplicación esos nombres no existen y `joblib.load` falla con
`AttributeError`. `preprocesamiento.registrar_en_main()` los publica en `__main__` justo
antes de deserializar, así que el modelo se reutiliza **tal cual salió del notebook**, sin
reentrenarlo.

Si en el futuro se reentrena importando las clases desde este módulo en vez de
definiéndolas en el notebook, el `.joblib` guardará las referencias ya apuntando a
`preprocesamiento` y ese paso dejará de ser necesario.

### La versión de scikit-learn tiene que coincidir

Un modelo serializado con una versión de scikit-learn puede comportarse de forma distinta
—o no cargarse— con otra. Por eso `requirements.txt` fija la versión exacta.

Comprueba con qué versión se entrenó el `.joblib` que vas a desplegar:

```python
import joblib
print(joblib.load("modelo_corazon_interpretado_completo.joblib")["version_sklearn"])
```

Si no coincide con la del entorno, tienes dos salidas:

1. **Usar tu propio `.joblib`**, el que generó tu notebook en este mismo entorno. Es la
   opción segura:
   `cp notebooks/6-interpretation/*.joblib notebooks/7-deploy/`
2. Instalar la versión con la que se guardó el modelo y fijarla en `requirements.txt`.

Para ver exactamente qué falta:

```bash
uv run python notebooks/7-deploy/diagnostico.py
```

Ese script imprime las versiones del entorno, las clases que el `.joblib` necesita y
cuáles no se encuentran. La barra lateral de la app muestra además ambas versiones de
scikit-learn y la del módulo de preprocesamiento.

## Nota sobre `ruff` y el orden de los imports

`import preprocesamiento` aparece ordenado alfabéticamente **entre las dependencias de
terceros**, no en un bloque aparte. Es intencionado: `ruff` decide si un módulo es propio
del proyecto según la carpeta desde la que se ejecuta, y el CI lo ejecuta desde la raíz
del repositorio, donde `preprocesamiento` no está a la vista.

Si ejecutas `ruff` desde dentro de esta carpeta te pedirá lo contrario. Hazle caso al CI,
que es el que tiene que pasar.

## Aviso

Demo académica construida sobre la cohorte Cleveland del dataset UCI Heart Disease
(480 pacientes tras el saneamiento). **No es un dispositivo médico** y no debe usarse para
tomar decisiones clínicas sobre personas reales.
