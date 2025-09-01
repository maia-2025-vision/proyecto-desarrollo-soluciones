from fastapi import FastAPI

from .routes import router

app = FastAPI(title="Cow Detection API")

app.include_router(router)
