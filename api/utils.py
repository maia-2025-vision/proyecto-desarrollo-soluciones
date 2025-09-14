import json
from urllib.parse import unquote, urlparse

import boto3

session = boto3.Session(profile_name="dvc-user")
s3_client = session.client("s3")
bucket = "cow-detect-maia"


def parse_s3_url(url: str) -> tuple[str, str]:
    """Parses an S3 URL into its bucket name and key.

    Args:
        url (str): The S3 URL to parse.

    Returns:
        tuple: A tuple containing the bucket name (str) and key (str).
               Raises if the URL is not a valid S3 URL.
    """
    parsed_url = urlparse(url)
    assert parsed_url.scheme == "s3"
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip("/")

    return bucket, key


def download_file_from_s3(url: str) -> bytes:
    bucket, key = parse_s3_url(url)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    resp_content = response["Body"].read()
    assert isinstance(resp_content, bytes), f"{type(resp_content)=}"

    return resp_content


def upload_json_to_s3(prediction: dict, image_url: str):
    """Sube un diccionario de predicción como JSON a S3.

    Convierte la URL de la imagen en la clave JSON, cambia la extensión y guarda el archivo.

    Args:
        prediction (dict): Datos de la predicción.
        image_url (str): URL original de la imagen.

    Returns:
        str: Ruta S3 del archivo JSON.

    Raises:
        ValueError: Si la URL no tiene una extensión de imagen válida.
    """
    parsed = urlparse(image_url)

    # Decodificar y ajustar key
    original_key = unquote(parsed.path.lstrip("/"))

    # Cambiar extensión a .json
    if original_key.lower().endswith((".jpg", ".jpeg", ".png")):
        json_key = original_key.rsplit(".", 1)[0] + ".json"
    else:
        raise ValueError("URL no termina en extensión de imagen válida.")

    # Subir JSON a S3
    s3_client.put_object(
        Bucket=bucket, Key=json_key, Body=json.dumps(prediction), ContentType="application/json"
    )

    return f"s3://{bucket}/{json_key}"


def list_flyover_folders(farm: str) -> list[str]:
    """Lista las carpetas de sobrevuelos para una granja.

    Args:
        farm (str): Nombre de la granja.

    Returns:
        list[str]: Nombres de las carpetas encontradas en S3.
    """
    prefix = f"{farm}/"
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    folders = [cp["Prefix"].rstrip("/").split("/")[-1] for cp in response.get("CommonPrefixes", [])]
    return folders


def get_predictions_from_s3_folder(farm: str, flyover: str) -> list[dict]:
    """Obtiene predicciones JSON desde una carpeta en S3.

    Args:
        farm (str): Nombre de la granja.
        flyover (str): Carpeta de sobrevuelos.

    Returns:
        list[dict]: Predicciones o errores de lectura.
    """
    prefix = f"{farm}/{flyover}/"
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    objects = response.get("Contents", [])
    json_keys = [obj["Key"] for obj in objects if obj["Key"].endswith(".json")]

    results = []
    for key in json_keys:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read()
        try:
            result = json.loads(content)
            results.append(result)
        except Exception as e:
            results.append({"error": f"Error leyendo {key}: {str(e)}"})

    return results
