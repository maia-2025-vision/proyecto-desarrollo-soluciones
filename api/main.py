import os
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import torchvision.transforms as transforms
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI, requests
from fastapi.responses import JSONResponse
from loguru import logger

from api.req_resp_types import PredictionError
from api.routes import model_pack, router
from api.torch_utils import get_prediction_model


def check_aws_credentials():
    # verificar credenciales
    try:
        s3 = boto3.client("s3")
        s3.list_objects_v2(Bucket="cow-detect-maia")
        logger.info("AWS credentials are valid and S3 is accessible.")
    except NoCredentialsError as e:
        logger.error("AWS credentials not found.")
        raise RuntimeError("Missing AWS credentials.") from e
    except ClientError as e:
        logger.error(f"AWS credential issue or permission error: {e}")
        raise RuntimeError("AWS credentials invalid or insufficient permissions.") from e


# Proper way to load a model on startup
# https://fastapi.tiangolo.com/advanced/events/#use-case
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model

    model_path = os.environ["MODEL_PATH"]
    logger.info(f"from env var got: MODEL_PATH={model_path}")

    aws_profile = os.getenv("AWS_PROFILE")
    logger.info(f"env var AWS_PROFILE={aws_profile!r}")

    check_aws_credentials()

    model_weights_path = Path(model_path)
    pt_model = get_prediction_model(model_weights_path)
    pt_model.eval()

    model_pack["model"] = pt_model
    model_pack["transform"] = transforms.ToTensor()

    yield
    # Clean up the ML models and release the resources
    model_pack.clear()


app = FastAPI(title="Cow Detection API", lifespan=lifespan)
app.include_router(router, prefix="/api")


@app.exception_handler(PredictionError)
async def custom_exception_handler(request: requests.Request, exc: PredictionError):
    return JSONResponse(
        status_code=exc.status,
        content={"url": exc.url, "error": exc.args[0]},
    )
