import torch


def make_jsonifiable(record: dict[str, torch.Tensor]) -> dict[str, list]:
    """Convert values of input dict to list.

    Resulting dict can be passed through json.dumps()
    """
    return {k: v.tolist() for k, v in record.items()}


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
