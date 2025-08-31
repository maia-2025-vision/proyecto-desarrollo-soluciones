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
uv sync --native-tls --devq
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

## Streamlit Application

This project includes a Streamlit web application for uploading images to S3 and viewing detection results.

### Running the Streamlit App

```bash
# Make sure dependencies are installed
uv sync --native-tls

# Run the streamlit app
uv run streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Features

The application has two pages:

1. **Upload Images**: Upload images to the S3 bucket `cow-detect-maia` for processing
   - Configure S3 prefix/folder path
   - Batch upload multiple images
   - Automatically calls processing endpoint after successful uploads

2. **View Detections**: Display images with bounding boxes from detection API
   - Fetch detection data from API endpoints
   - Visualize bounding boxes with customizable colors
   - Support for multiple JSON formats
   - Display confidence scores and labels

### Configuration

Before using the upload feature:
- Configure AWS credentials (`aws configure` or environment variables)
- Update the `ENDPOINT_URL` in `pages/1_Upload_Images.py` with your processing endpoint

# TODO: 

- [ ] Revisar custom_collate_fn
- [ ] Experiment freezing all parameters except final BoxPRedictor head...