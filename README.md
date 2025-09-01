# Cow Detect

Proyecto para la materia Desarrollo Soluciones de MAIA


# Setup inicial

Lo siguiente asume que `astral-uv` está ya instalado a nivel de sistema operativo.
Para instrucciones de instalación referirse a:
https://docs.astral.sh/uv/getting-started/installation/

```bash
# instala la versión de python especificada en .python-version
uv python install

# instala las dependencias del grupo [dev] definidas en pyproject.toml
uv sync --native-tls --dev
source .venv/bin/activate
```

## Activar checks de código automáticos antes de commit

Altamente recomendado!

```bash
pre-commit install
```

### Obtener datos de repo remoto DVC remoto

```bash
dvc pull  # .venv debe estar activado
```


## Entrenamiento

Desde la raiz del repo ejecutar 

```bash
dvc repro -s train-v1
```

## Ejecución del API
Previamente haber instalado las dependencias del grupo [dev] definidas en pyproject.toml

Se debe ejecutar el script run_api.py ubicado en el folder api/.

```bash
python api/run_api.py
```

La configuración del endpoint se encuentra en api/config.py

La consulta de los endpoints se encuentra en http://localhost:8000/docs, de acuerdo a las configuraciones iniciales del endpoint.

## Aplicación Streamlit

Este proyecto incluye una aplicación web Streamlit para cargar imágenes a S3 y visualizar resultados de detección.

### Ejecutar la Aplicación Streamlit

```bash
# Asegurarse de que las dependencias estén instaladas
uv sync --native-tls

# Ejecutar la aplicación streamlit
uv run streamlit run streamlit_app.py
```

La aplicación estará disponible en `http://localhost:8501`

### Características

La aplicación tiene dos páginas:

1. **Cargar Imágenes**: Cargar imágenes al bucket S3 `cow-detect-maia` para procesamiento
   - Configurar finca y sobrevuelo para organización
   - Carga por lotes de múltiples imágenes
   - Llama automáticamente al endpoint de procesamiento después de cargas exitosas

2. **Ver Detecciones**: Mostrar imágenes con cuadros delimitadores desde API de detección
   - Obtener datos de detección desde endpoints API
   - Visualizar cuadros delimitadores con colores personalizables
   - Soporte para múltiples formatos JSON
   - Mostrar puntuaciones de confianza y etiquetas

### Configuración

Antes de usar la función de carga:
- Configurar credenciales AWS (`aws configure` o variables de entorno)
- Actualizar el `ENDPOINT_URL` en `pages/1_Upload_Images.py` con su endpoint de procesamiento

# TODO: 

- [ ] Revisar custom_collate_fn
- [ ] Experiment freezing all parameters except final BoxPRedictor head...