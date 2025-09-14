import torch

NaN = float("nan")


def make_jsonifiable(record: dict[str, torch.Tensor]) -> dict[str, list]:
    """Convert values of input dict to list.

    Resulting dict can be passed through json.dumps()
    """
    return {k: v.tolist() for k, v in record.items()}


def _process_singleton(x: torch.Tensor, *, negative_to_nan: bool) -> float:
    if not x.numel() == 1:
        raise ValueError(f"x={x}, but expected single element tensor")

    value = x.item()
    if value < 0 and negative_to_nan:
        return NaN
    else:
        return value


def make_jsonifiable_singletons(
    record: dict[str, torch.Tensor], *, negative_to_nan: bool
) -> dict[str, float]:
    """Convert values of input dict to single floats.

    It assumes they are all tensors with exactly one element, otherwise will raise an error.

    Resulting dict can be passed through json.dumps()
    """
    return {k: _process_singleton(x, negative_to_nan=negative_to_nan) for k, x in record.items()}


def custom_collate_dicts(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, list]:
    """Given a list of size n of dictionaries with the same keys...

    return a single dictionary with the same keys, each value being a list of size n

    e.g:
    custom_collate_dicts([
      {"a": 1, "b": 2, "c": 3},
      {"a": 4, "b": 5, "c": 6},
    ]) == {"a": [1, 4], "b": [2, 5], "c": [3, 6]}
    """
    # print("input to collate:\n", summarize(batch))
    keys = batch[0].keys()
    return {k: [item[k] for item in batch] for k in keys}


def zip_dict(a_dict_of_lists: dict[str, list]) -> list[dict]:
    """The reverse operation of the above.

    Given a dict of lists of the same size (n),
    return a list of size n of dicts all having the same keys.

    zip_dict({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    == [
        {"a": 1, "b": 4, "c": 7},
        {"a": 2, "b": 5, "c": 8},
        {"a": 3, "b": 6, "c": 9},
    ]
    """
    keys = list(a_dict_of_lists.keys())
    n = len(a_dict_of_lists[keys[0]])
    return [{k: a_dict_of_lists[k][i] for k in keys} for i in range(n)]


def filter_bboxes_for_classes(
    boxes0: list[list], label_strs: list[str], cls_name_to_id: dict[str, int]
) -> tuple[list[list], list[int]]:
    """Filter bboxes and their corresponding labels.

    Keep only boxes corresponding to labels that are keys in cls_name_to_id.
    Also map string labels to integer label_ids
    """
    # Filter bboxes only for classes that are in class_name_to_id
    boxes: list[list[int]] = []
    labels: list[int] = []
    for bbox, class_name in zip(boxes0, label_strs, strict=False):
        if class_name not in cls_name_to_id:
            continue
        else:
            boxes.append(bbox)
            labels.append(cls_name_to_id[class_name])

    return boxes, labels
