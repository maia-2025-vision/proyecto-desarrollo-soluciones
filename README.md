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

El API necesita acceso a S3. Asegúrate de haber configurado tu perfil de AWS (`aws configure`).
A continuación se asume que el perfil configurado se llama 'dvc-user'

Se debe ejecutar el script `api/run_api.py` asegurándose de que al menos la variable de entorno
`MODEL_PATH` esté definida en el ambiente.

Ejemplo:
```bash
export AWS_PROFILE="dvc-user"
export MODEL_PATH="data/training/v1/faster-rcnn/model.pth"
python api/run_api.py
```

La configuración del endpoint se encuentra en [api/config.py](api/config.py)

La consulta de los endpoints se encuentra en [localhost:8000](http://localhost:8000/docs), de acuerdo con las configuraciones iniciales del endpoint.

## Dockerización y Ejecución dentro de docker

El api se dockeriza con el comando:

```bash
poe dockerize-api
```

Este comando ejecuta algo similar a esto: `docker build --progress=plain -t cowd-api -f api/Dockerfile .`


Notar que el modelo está "quemado" dentro de la imagen de docker.

Si se quiere cambiar el modelo hay que cambiar dos líneas de `api/Dockerfile`.
1. La línea justo después del comentario `# Model weights copied into image here:` (cambiar ambas menciones a ruta externa y ruta interna)
2. La línea que define la variable de entorno MODEL_PATH dentro del contenedor: `ENV MODEL_PATH=....`


Para ejecutar el api desde docker y exponer el puerto 8000 compartiendo las credenciales configuradas arriba:

```bash
docker run -p 8000:8000 -e AWS_PROFILE='dvc-user' -v $HOME/.aws:/root/.aws  cowd-api
```

Otra opción es no definir AWS_PROFILE pero en su lugar pasar valores para las variables de entorno de credenciales:

```bash
export AWS_ACCESS_KEY_ID=....
export AWS_SECRET_ACCESS_KEY=....
docker run -p 8000:8000 \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    cowd-api
```



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

### Dockerización de Streamlit

La aplicación Streamlit se puede ejecutar en Docker usando el archivo `streamlit.Dockerfile`:

1. **Construir la imagen Docker:**
   ```bash
   docker build -t cowd-streamlit -f streamlit.Dockerfile .
   ```

2. **Ejecutar el contenedor con credenciales AWS:**
   
   Opción 1 - Usando perfil AWS local:
   ```bash
   docker run -p 8501:8501 \
     -e AWS_PROFILE='dvc-user' \
     -v $HOME/.aws:/root/.aws:ro \
     cowd-streamlit
   ```
   
   Opción 2 - Usando credenciales explícitas:
   ```bash
   export AWS_ACCESS_KEY_ID=....
   export AWS_SECRET_ACCESS_KEY=....
   docker run -p 8501:8501 \
     -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
     -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
     cowd-streamlit
   ```

3. **Acceder a la aplicación:**
   Una vez ejecutado, la aplicación estará disponible en `http://localhost:8501`

**Nota:** Para que la aplicación Streamlit pueda comunicarse con el API, asegúrate de que el API esté ejecutándose y accesible. Si ambos servicios están en Docker, considera usar una red Docker compartida.



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
