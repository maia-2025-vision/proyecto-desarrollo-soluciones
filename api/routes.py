import io
from pathlib import Path

import requests
import torch
import torchvision.transforms as transforms
from fastapi import APIRouter, HTTPException
from PIL import Image

from api.utils import get_predictions_from_s3_folder, list_flyover_folders, upload_json_to_s3
from cow_detect.predict.batch import get_prediction_model

router = APIRouter()

# Carga el modelo solo una vez
model_weights_path = Path("data/training/teo/v1/faster-rcnn-toy/model.pth")
model = get_prediction_model(model_weights_path)
model.eval()

transform = transforms.ToTensor()


@router.post("/predict")
async def predict(image_urls: list[str]):
    """Realiza la predicción a partir de una lista de URLs de imágenes.

    Descarga cada imagen, la transforma en tensor, ejecuta el modelo de predicción
    y sube los resultados a S3. Devuelve una lista con los resultados o errores por imagen.
    """
    results = []

    for url in image_urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # lanza excepción si no es 200 OK
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            results.append(
                {"url": url, "error": f"No se pudo descargar o abrir la imagen: {str(e)}"}
            )
            continue

        image_tensor = transform(image).unsqueeze(0)  # batch size 1

        try:
            with torch.no_grad():
                predictions = model(image_tensor)
            pred = predictions[0]
            pred_json = {k: v.tolist() for k, v in pred.items()}

            pred_result = {"url": url, "predictions": pred_json}

            _ = upload_json_to_s3(pred_result, url)

            results.append(pred_result)
        except Exception as e:
            results.append(
                {"url": url, "error": f"Error durante la predicción o subida a S3: {str(e)}"}
            )

    return {"results": results}


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
