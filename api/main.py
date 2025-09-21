import os
from contextlib import asynccontextmanager
from pathlib import Path

import torchvision.transforms as transforms
from fastapi import FastAPI, requests
from fastapi.responses import JSONResponse
from loguru import logger

from api.req_resp_types import PredictionError
from api.routes import model_pack, router
from api.torch_utils import get_prediction_model


# Proper way to load a model on startup
# https://fastapi.tiangolo.com/advanced/events/#use-case
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model

    model_path = os.environ["MODEL_PATH"]
    logger.info(f"from env var got: MODEL_PATH={model_path}")

    aws_profile = os.getenv("AWS_PROFILE")
    logger.info(f"env var AWS_PROFILE={aws_profile!r}")
    aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_is_defined = os.getenv("AWS_SECRET_ACCESS_KEY") is not None

    if aws_profile is None and (aws_key_id is None or not aws_secret_is_defined):
        logger.error(
            "Need to provide at least AWS_PROFILE env var,"
            " or both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
        )
        raise RuntimeError("No AWS credentials!")

    model_weights_path = Path(model_path)
    pt_model = get_prediction_model(model_weights_path)
    pt_model.eval()

    model_pack["model"] = pt_model
    model_pack["transform"] = transforms.ToTensor()

    yield
    # Clean up the ML models and release the resources
    model_pack.clear()


app = FastAPI(title="Cow Detection API", lifespan=lifespan)
app.include_router(router)


@app.exception_handler(PredictionError)
async def custom_exception_handler(request: requests.Request, exc: PredictionError):
    return JSONResponse(
        status_code=exc.status,
        content={"url": exc.url, "error": exc.args[0]},
    )
