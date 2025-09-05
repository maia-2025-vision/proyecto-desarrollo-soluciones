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

# TODO: 

- [ ] Revisar custom_collate_fn
- [ ] Experiment freezing all parameters except final BoxPRedictor head...



## Note on BBOX formats:


SSD/ RCNN/ Fast RCNN/ Faster RCNN use the same format while training an object detection model. They use the Pascal VOC dataset format.

In this format, the bounding box is represented in as follows `[x_min, y_min, x_max, y_max]`

Source: https://lohithmunakala.medium.com/bounding-box-formats-for-models-like-yolo-ssd-rcnn-fast-rcnn-faster-rcnn-807be7721527


## Ideas para el futuro: 

Usar otras arquitecturas de acá: 
https://docs.pytorch.org/vision/main/models/faster_rcnn.html