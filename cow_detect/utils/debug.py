import torch

DepthLine = tuple[int, str]


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
