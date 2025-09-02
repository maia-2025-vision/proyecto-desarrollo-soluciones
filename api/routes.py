import io
from http import HTTPStatus
from pathlib import Path

import requests
import torch
import torchvision.transforms as transforms
from fastapi import APIRouter, HTTPException
from loguru import logger
from PIL import Image

from api.types import (
    PredictionError,
    PredictionResult,
    PredictManyRequest,
    PredictManyResult,
    PredictOneRequest,
)
from api.utils import (
    download_file_from_s3,
    get_predictions_from_s3_folder,
    list_flyover_folders,
    upload_json_to_s3,
)
from cow_detect.predict.batch import get_prediction_model

router = APIRouter()

# Carga el modelo solo una vez
model_weights_path = Path("data/training/teo/v1/faster-rcnn/model.pth")
model = get_prediction_model(model_weights_path)
model.eval()

transform = transforms.ToTensor()


def download_image_from_url(url: str):
    if url.startswith("s3://"):  # an s3 url that might not be public!
        file_bytes = download_file_from_s3(url)
    else:  # regular "public" url
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # lanza excepción si no es 200 OK
        file_bytes = response.content

    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


@router.post("/predict-many")
async def predict_many_endpoint(req: PredictManyRequest):
    """Realiza la predicción a partir de una lista de URLs de imágenes.

    Descarga cada imagen, la transforma en tensor, ejecuta el modelo de predicción
    y sube los resultados a S3. Devuelve una lista con los resultados o errores por imagen.
    """
    results = []

    for url in req.urls:
        result = predict_one(url)
        results.append(result)

    return PredictManyResult(results=results)


@router.post("/predict")
def predict_one_endpoint(req: PredictOneRequest) -> PredictionResult:
    return predict_one(url=req.s3_path)


def predict_one(url: str):
    try:
        image = download_image_from_url(url)
    except Exception as e:
        return PredictionError(
            url=url,
            status=HTTPStatus.UNAUTHORIZED,
            error=f"No se pudo descargar o abrir la imagen: {str(e)}",
        )

    image_tensor = transform(image).unsqueeze(0)  # batch size 1

    try:
        with torch.no_grad():
            model_outputs = model(image_tensor)

        pred = model_outputs[0]  # just the first one since we only passed one image in the batch
        logger.info(f"pred has keys: {pred.keys()}")
        pred_obj = {k: v.tolist() for k, v in pred.items()}
        pred_result = PredictionResult(url=url, detections=pred_obj)

    except Exception as e:
        return PredictionError(
            url=url,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            error=f"Error durante la predicción: {str(e)}",
        )

    try:
        upload_json_to_s3(pred_result.model_dump(), url)
    except Exception as e:
        return PredictionError(
            url=url, status=HTTPStatus.UNAUTHORIZED, error=f"Error durante la subida a S3: {str(e)}"
        )

    return pred_result


@router.get("/flyovers/{farm}")
def list_flyovers(farm: str):
    """Lista las carpetas de sobrevuelos disponibles para una granja específica."""
    try:
        folders = list_flyover_folders(farm)
        return {"flyovers": folders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar sobrevuelos: {str(e)}") from e


@router.get("/results/{farm}/{flyover}")
def get_predictions_from_folder(farm: str, flyover: str):
    """Obtiene las predicciones almacenadas en S3 para una granja y sobrevuelo dadas."""
    try:
        results = get_predictions_from_s3_folder(farm, flyover)
        return {"results": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al reconstruir predicciones: {str(e)}"
        ) from e
