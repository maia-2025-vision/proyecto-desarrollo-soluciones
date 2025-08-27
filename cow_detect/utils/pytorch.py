import torch
from loguru import logger


def auto_detect_device() -> torch.device:
    """Get an available accelerator device, default to CPU."""
    if torch.cuda.is_available():
        logger.info("Detected cuda device")
        return torch.device("cuda")
    elif torch.mps.is_available():
        logger.info("Detected mps device")
        return torch.device("mps")
    else:
        logger.info("No cuda/mps accelerator, using cpu.")
        return torch.device("cpu")
