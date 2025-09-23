# Manual de instalación: Instrucciones para ejecutar el proyecto con Docker Compose

## 1. Pre-requisitos

- Docker y Docker Compose instalados: https://docs.docker.com/desktop/
- AWS CLI configurado con credenciales (`aws configure`)
- Bucket S3 creado en AWS para almacenar los datos
- Usuario IAM con permisos de acceso al bucket S3

## 2. Descargar docker-compose.yml

Descarga el archivo de configuración:

```bash
wget https://raw.githubusercontent.com/maia-2025-vision/proyecto-desarrollo-soluciones/refs/heads/main/docker-compose.yml
```

O cópialo desde: https://raw.githubusercontent.com/maia-2025-vision/proyecto-desarrollo-soluciones/refs/heads/main/docker-compose.yml

## 3. Configuración AWS

### 3.1 Crear bucket S3
Crea un bucket S3 en AWS Console o con AWS CLI:
```bash
aws s3 mb s3://cow-detect-maia --region us-east-1
```

### 3.2 Configurar credenciales
Crea un archivo `.aws-credentials` en la misma carpeta del `docker-compose.yml`:

```ini
[default]
aws_access_key_id = TU_ACCESS_KEY
aws_secret_access_key = TU_SECRET_KEY
```

**Nota**: El usuario IAM debe tener permisos de lectura/escritura en el bucket S3.

## 4. Configurar bucket S3 (opcional)

Para usar un bucket S3 diferente, modifica la variable `S3_BUCKET` en el `docker-compose.yml`:

```yaml
environment:
  - S3_BUCKET=tu-bucket-personalizado
```

Si no se especifica, usa por defecto: `cow-detect-maia`

## 5. Ejecutar

```bash
docker-compose up
```

## 6. Acceso

- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
