# Manual de Instalación 

A continuación se presenta el manual de instalación de la solución en una máquina local, 

Ofrecemos esta opción como una opción ligera para uso privado en casos en que solo se requiera acceso para 
un usuario.
La ventaja de esta opción es que se evita tener que hacer el montaje de la solución en ECS o Railway.


## 0 Prerrequisitos:

- Instalar Docker desktop en la máquina. Instrucciones detalladas aquí https://docs.docker.com/desktop/
- Crear un bucket de S3 en AWS y obtener credenciales (access_key_id y secret_access_key) 
que permitan acceso de lectura y escritura al mismo.
- Crear un archivo de texto plano con esta información así, ejemplo: 

```bash
# nombre de archivo: env.secrets
AWS_ACCESS_KEY_ID=AKIA555555555555555
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
S3_BUCKET=bucket-name-here
```

## 1. Descargar docker-compose.yml del repo

Descargar el archivo [docker-compose.yml](https://raw.githubusercontent.com/maia-2025-vision/proyecto-desarrollo-soluciones/refs/heads/main/docker-compose.yml) desde el repo de github.


## 2. Arrancar el APP con docker compose

```bash
docker compose up
```

Después del arranque, el dashboard estará disponible en `http://0.0.0.0:8501`