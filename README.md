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
uv sync --all-extras
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

## (Re)Generar conjuntos de datos standarizados

Toma unos tres minutos:
```bash
dvc repro -s std-sky-ds1    
dvc repro -s std-sky-ds2
dvc repro -s std-sky-all
dvc repro -s std-icaerus-derval
dvc repro -s std-icaerus-jalogny
dvc repro -s std-icaerus-mauron
dvc repro -s std-icaerus-other
dvc repro -s std-icaerus-all
```

## Entrenamiento

Desde la raiz del repo ejecutar 

```bash
# Install torch-train dependencies 
uv sync --group dev --group torch-train
```

```bash
dvc repro -s train-v1
```

## Ejecución del API
Previamente haber instalado las dependencias del grupo [dev] definidas en pyproject.toml

Se debe ejecutar el script run_api.py ubicado en el folder api/.

```bash
python api/run_api.py
```

La configuración del endpoint se encuentra en [api/config.py](api/config.py)

La consulta de los endpoints se encuentra en [localhost:8000](http://localhost:8000/docs), de acuerdo a las configuraciones iniciales del endpoint.

## Aplicación Streamlit

Este proyecto incluye una aplicación web Streamlit para cargar imágenes a S3 y visualizar resultados de detección.

### Ejecutar la Aplicación Streamlit

Este proyecto utiliza [Poe the Poet](https://github.com/nat-n/poe-the-poet) para gestionar y ejecutar tareas de forma consistente.

1.  **Instalar dependencias:**
    Asegúrate de tener todas las dependencias instaladas, con el siguiente comando:
    
    ```bash
    uv sync --all-extras
    ```
    

2.  **Configurar credenciales de AWS:**
    La aplicación necesita acceso a S3. Asegúrate de haber configurado tu perfil de AWS (`aws configure`). La tarea de Poe está configurada para usar el perfil llamado `dvc-user`.

3.  **Ejecutar la aplicación:**
    Para iniciar la aplicación Streamlit, ejecuta el siguiente comando:
    
    ```bash
    poe dashboard
    ```
    
    Este comando se encargará de establecer la variable de entorno `AWS_PROFILE=dvc-user` y lanzar la aplicación, que estará disponible en `http://localhost:8501`.

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
idealmente ponerlas en un perfil llamado dvc-user
- ~~Actualizar el `ENDPOINT_URL` en `pages/1_Upload_Images.py` con su endpoint de procesamiento~~ (Ahora se configura automáticamente)


## Note on BBOX formats:


SSD/ RCNN/ Fast RCNN/ Faster RCNN use the same format while training an object detection model. They use the Pascal VOC dataset format.

In this format, the bounding box is represented in as follows `[x_min, y_min, x_max, y_max]`

Source: https://lohithmunakala.medium.com/bounding-box-formats-for-models-like-yolo-ssd-rcnn-fast-rcnn-faster-rcnn-807be7721527


## Ideas para el futuro: 

Usar otras arquitecturas de acá: 
https://docs.pytorch.org/vision/main/models/faster_rcnn.html

# TODO:

- [ ] Averiguar lo de iscrow que se menciona en la funcion de mean_average_precision de torchmetrics 
y también se estaba poniendo en los diccionarios devueltos por datasetv1... 
- [ ] Experiment freezing all parameters except final BoxPRedictor head...
- [ ] Experiment with other optimizers
