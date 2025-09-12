import os
import sys

import uvicorn
from loguru import logger

from api.config import settings

# If UVICORN_PORT, UVICORN_RELOAD env var  set then take values from them,
# otherwise take them from default settings
HOST = os.getenv("UVICORN_HOST", settings.HOST)
PORT = int(os.getenv("UVICORN_PORT", settings.PORT))
RELOAD = bool(os.getenv("UVICORN_RELOAD", settings.RELOAD))


if __name__ == "__main__":
    logger.info(f"Starting uvicorn: HOST={HOST} PORT={PORT} RELOAD={RELOAD}")
    uvicorn.run("api.main:app", host=settings.HOST, port=PORT, reload=RELOAD)
