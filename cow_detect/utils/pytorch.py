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


def try_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device == "cpu":
        return torch.device("cpu")
    elif device == "mps" and torch.mps.is_available():
        return torch.device("mps")
    else:
        logger.warning(f"Unknown device: `{device}`, defaulting to 'cpu'")
        return torch.device("cpu")


def detach_dict(a_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Make sure all tensors in a tensor valued dict are detached and moved to cpu."""
    return {k: v.detach().cpu() for k, v in a_dict.items()}


def detach_dicts(dicts: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Like before but for a list of dicts."""
    return [detach_dict(a_dict) for a_dict in dicts]


def dict_to_device(a_dict: dict[str, object], device: torch.device) -> dict[str, object]:
    """Produce a new dict but with values that are tensors moved to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in a_dict.items()}
