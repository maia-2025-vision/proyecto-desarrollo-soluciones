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

# TODO: 

- [ ] Revisar custom_collate_fn
- [ ] Experiment freezing all parameters except final BoxPRedictor head...