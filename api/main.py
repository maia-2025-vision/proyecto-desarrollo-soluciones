from fastapi import FastAPI, requests
from fastapi.responses import JSONResponse

from api.types import PredictionException

from .routes import router

app = FastAPI(title="Cow Detection API")

app.include_router(router)


@app.exception_handler(PredictionException)
async def custom_exception_handler(request: requests.Request, exc: PredictionException):
    return JSONResponse(
        status_code=exc.status,
        content={"url": exc.url, "error": exc.args[0]},
    )
