# Demo online — Predicción de enfermedad cardíaca

Aplicación web en [Streamlit](https://streamlit.io/) que expone el modelo entrenado por
el *training pipeline*. El usuario introduce los 13 datos clínicos de un paciente y
obtiene la probabilidad estimada de enfermedad coronaria.

## Contenido

| Archivo | Qué es |
|---|---|
| `app.py` | La aplicación |
| `models/modelo_corazon_completo.joblib` (raíz) | Modelo + umbral calibrado + metadatos |
| `requirements.txt` (raíz) | Dependencias para Streamlit Community Cloud |

La app **no incluye código de transformación**: importa el `inference_pipeline`, que a
su vez importa el `feature_pipeline`. Una sola ruta de transformación para el
entrenamiento, la inferencia por archivo y la demo.

## Ejecutar en local

Desde la raíz del repositorio:

```bash
uv pip install -r requirements.txt
uv run streamlit run src/demo/app.py
```

Se abre en <http://localhost:8501>.

Si el modelo no existe todavía, la app lo dice y te indica el comando; hay que ejecutar
antes los dos pipelines:

```bash
uv run python src/pipelines/feature_pipeline/feature_pipeline.py
uv run python src/pipelines/training_pipeline/train_pipeline.py
cp data/06_models/modelo_corazon_completo.joblib models/
```

## Publicar en Streamlit Community Cloud

1. **Asegúrate de que el modelo está versionado.** `data/` está en el `.gitignore`, así
   que el artefacto no viaja al repositorio desde ahí. La copia de `models/` sí:

   ```bash
   git add models/modelo_corazon_completo.joblib
   ```

   Ocupa unos 170 KB, muy por debajo del límite de GitHub. Si algún día superara los
   100 MB haría falta Git LFS.

2. **Sube la rama a GitHub** con `app.py`, `models/` y `requirements.txt`.

3. Entra en <https://share.streamlit.io> y conecta tu cuenta de GitHub.

4. **New app** → selecciona:
   - *Repository*: `CarlosP1005/Heart_project`
   - *Branch*: la rama de este entregable (o `main` una vez mergeada)
   - *Main file path*: `src/demo/app.py`

5. **Deploy**. Streamlit Cloud instala `requirements.txt` automáticamente y publica la
   app en una URL con el formato `https://<algo>.streamlit.app`.

### Por qué `scikit-learn` va con versión exacta

En `requirements.txt` la línea es `scikit-learn==1.9.0`, no `>=`. El modelo se serializó
con `joblib`, que guarda objetos de Python: cargarlo con una versión distinta de
scikit-learn puede fallar o, peor, funcionar y comportarse de otra manera. La versión
con la que se entrenó viaja dentro del propio artefacto (`version_sklearn`) y se muestra
en la barra lateral de la app, para poder comprobarlo de un vistazo.

Si reentrenas con otra versión de scikit-learn, actualiza también esa línea.

## Uso de la aplicación

**Barra lateral.** Muestra con qué modelo estás prediciendo —tipo, fecha, versión de
scikit-learn y métricas de test— y permite mover el **umbral de decisión**. El valor por
defecto (0.31) es el que maximiza el F1 en validación cruzada; bajarlo detecta más
enfermos a costa de más falsas alarmas, subirlo hace lo contrario.

**Casos de ejemplo.** Tres botones rellenan el formulario con un perfil de bajo riesgo,
uno de alto riesgo y un caso intermedio, para probar la app sin teclear 13 campos.

**Formulario.** Las 13 variables agrupadas en demografía, síntomas y pruebas. Los rangos
de cada campo numérico son los mismos que valida el feature pipeline, así que no se
puede introducir un valor clínicamente imposible.

**Resultado.** Probabilidad, banda de riesgo, veredicto según el umbral y las cuatro
señales clínicas de riesgo presentes. Estas últimas no son la explicación del modelo
—un *gradient boosting* no decide así— sino un contexto que el usuario puede contrastar
con lo que ve en el paciente.

## Advertencia

Esta demo es un ejercicio académico. No es un dispositivo médico y no sustituye el
criterio de un profesional sanitario.
