# Demo online — Predicción de enfermedad cardíaca

Aplicación web en [Streamlit](https://streamlit.io/) que expone el modelo entrenado por
el *training pipeline*. Tiene dos páginas:

- **Paciente individual** (`app.py`): se introducen los 13 datos clínicos de un paciente
  y se obtiene la probabilidad estimada de enfermedad coronaria.
- **Procesamiento por lotes** (`pages/Procesamiento_por_lotes.py`): se sube un archivo
  con varios pacientes y se obtienen todas las predicciones, visibles en pantalla y
  descargables en CSV.

## Contenido

| Archivo | Qué es |
|---|---|
| `app.py` | Página de predicción individual y punto de entrada |
| `pages/Procesamiento_por_lotes.py` | Página de lotes (sólo widgets) |
| `lote.py` | Lógica del lote: lectura, validación, predicción, resumen |
| `models/modelo_corazon_completo.joblib` (raíz) | Modelo + umbral calibrado + metadatos |
| `requirements.txt` (raíz) | Dependencias para Streamlit Community Cloud |

La app **no incluye código de transformación**: importa el `inference_pipeline`, que a
su vez importa el `feature_pipeline`. Una sola ruta de transformación para el
entrenamiento, la inferencia por archivo, la demo individual y el lote.

La lógica del lote vive en `lote.py`, separada de la página, porque la interfaz de
Streamlit no se puede probar con pytest sin levantar un navegador. Así las pruebas
cubren el comportamiento real —qué archivos se aceptan, qué pasa si falta una columna,
que ninguna fila se pierde— y no sólo el dibujo.

## Ejecutar en local

Desde la raíz del repositorio:

```bash
uv pip install -r requirements.txt
uv run streamlit run src/demo/app.py
```

Se abre en <http://localhost:8501>, con las dos páginas en la barra lateral. Si sólo
interesa el lote, también se puede arrancar esa página sola:

```bash
uv run streamlit run src/demo/pages/Procesamiento_por_lotes.py
```

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

## Uso del procesamiento por lotes

**1. Prepara el archivo.** Una fila por paciente y, como mínimo, las 13 columnas
obligatorias con estos nombres exactos:

```
age, sex, chest_pain, rest_bp, chol, fbs, rest_ecg, max_hr, exang, old_peak, slope, ca, thal
```

Formatos aceptados: `.csv`, `.xlsx` y `.parquet`, hasta 5 000 filas. Hay una plantilla
descargable dentro de la propia página, y ejemplos completos de entrada y salida en
`notebooks/16.Procesamiento batch con Streamlit/ejemplos/`.

Conviene saber:

- **Columnas de más:** se conservan en la salida pero no entran al modelo. Un
  identificador (`id_paciente`, `id` o `paciente`) se usa para etiquetar cada
  resultado; si no hay ninguno, las filas se numeran de 1 en adelante. Si el archivo
  trae la etiqueta real (`disease`), no se usa para predecir, pero queda en la salida y
  permite comparar lo predicho con lo observado.
- **Valores ausentes:** se pueden dejar vacíos. Los imputa el propio `Pipeline` con la
  mediana aprendida en el entrenamiento.
- **Columnas obligatorias que falten:** el lote **no se procesa**. La página dice
  cuáles faltan y no genera ninguna descarga. Un lote a medias es peor que ninguno,
  porque el usuario se lleva un CSV que parece completo.

**2. Súbelo** en *Archivo de pacientes*.

**3. Lee el resultado.** El resumen da el número de casos probables, la proporción, la
probabilidad media, el reparto por banda de riesgo y el histograma de probabilidades.
La tabla se puede filtrar por banda; el filtro sólo afecta a lo que se ve.

**4. Descarga el CSV.** Incluye siempre el lote completo: el archivo original tal cual
llegó más `probabilidad_enfermedad`, `prediccion`, `diagnostico` y `nivel_riesgo`. Va
en UTF-8 con BOM para que Excel en Windows abra bien las tildes.

**El umbral se aplica sobre el lote ya predicho.** Moverlo reclasifica sin volver a
subir el archivo: las probabilidades no cambian, sólo el corte. Es la forma rápida de
ver cuántos casos entrarían en revisión con una política más o menos conservadora.

## Advertencia

Esta demo es un ejercicio académico. No es un dispositivo médico y no sustituye el
criterio de un profesional sanitario.
