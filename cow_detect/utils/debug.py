import os

import psutil
import torch

DepthLine = tuple[int, str]


def get_process_memory_info_mb() -> float:
    """Process resident memory size in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (2**20)  # Resident Set Size (physical memory used)


def get_cuda_mem() -> tuple[float, float]:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def mem_info() -> dict[str, int]:
    ret = {}
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    ret["rss"] = mem_info.rss
    if torch.cuda.is_available():
        ret["cuda-alloc"] = torch.cuda.memory_allocated()
        ret["cuda-reserved"] = torch.cuda.memory_reserved()

    return ret


def mem_info_str(
    unit: str = "MB",
    prefix: str = "Mem-Info(",
    decimals: int = 1,
    sep=", ",
    postfix: str = ")[{unit}]",
) -> dict[str, int]:
    """Return RAM and GPU memory information as a formatted string."""
    denom = {"MB": 2**20, "GB": 2**30}[unit]
    pieces = []

    mem_info_dict = mem_info()
    for key in ["rss", "cuda-alloc", "cuda-reserved"]:
        if key in mem_info_dict:
            pieces.append(f"{key}={mem_info_dict[key] / denom:.{decimals}f}")

    return prefix.format(unit=unit) + sep.join(pieces) + postfix.format(unit=unit)


def summarize_lines(obj: object, depth: int = 0, pretty: bool = True) -> list[DepthLine]:
    """Produce lines summarizing and object."""
    if isinstance(obj, torch.Tensor):
        shape_str = " x ".join(str(dim) for dim in tuple(obj.shape))
        return [(depth, f"Tensor[{shape_str}] (device={obj.device})")]

    elif isinstance(obj, dict):
        ret = [(depth, f"Dict[{len(obj)}](")]
        for key, val in obj.items():
            v_lines = summarize_lines(val, depth + 1, pretty=pretty)
            for i, (dep, line) in enumerate(v_lines):
                if i == 0:  # put key only on first line
                    ret.append((dep + 1, f"{key}: {line}"))
                else:
                    ret.append((dep + 1, line))

        ret.append((depth, ") # /Dict"))

    elif isinstance(obj, tuple | list):
        type_name = type(obj).__name__.upper()
        ret = [(depth, f"{type_name}[{len(obj)}](")]
        for val in obj:
            v_lines = summarize_lines(val, depth + 1, pretty=pretty)
            for dep, line in v_lines:
                ret.append((dep + 1, line))
        ret.append((depth, f") # /{type_name}"))
    else:  # default case
        ret = [(depth, repr(obj))]

    return ret


def summarize(obj: object, depth: int = 0, pretty: bool = True) -> str:
    """Produce a compact summary of an object."""
    depth_lines = summarize_lines(obj, depth=depth, pretty=pretty)
    if pretty:
        parts = ["  " * d + line for (d, line) in depth_lines]
        return "\n".join(parts)
    else:
        parts = [line for _, line in depth_lines]
        return "".join(parts)
