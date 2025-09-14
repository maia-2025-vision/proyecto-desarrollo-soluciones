from http import HTTPStatus

from pydantic import BaseModel


class PredictionError(Exception):
    """Signals any error in the whole prediction process."""

    def __init__(self, url: str, status: HTTPStatus, error: str):
        super().__init__(error)
        self.url = url
        self.status = status


class PredictOneRequest(BaseModel):
    """Request for detection on one image."""

    name: str  # Nombre de la imagen
    s3_path: str


class PredictManyRequest(BaseModel):
    """Request for processing many images."""

    urls: list[str]


class Detections(BaseModel):
    """Boxes, and their scores in one image."""

    # parallel arrays:
    boxes: list[list[float]]
    scores: list[float]
    labels: list[int]


class PredictionResult(BaseModel):
    """Result of predicting on one image."""

    url: str
    detections: Detections


class PredictManyResult(BaseModel):
    """Detection results for many images."""

    results: list[PredictionResult]
