from pathlib import Path
from typing import TypeAlias, TypedDict

import torch


class Target(TypedDict):
    """An image its path together, bboxes and labels.

    boxes, labels and label_strs are parallel sequences of size N.
    """

    image_path: Path
    image_pt: torch.Tensor  # image as a tensor, dimension [3, W, H]
    boxes: torch.Tensor  # dimension should be [N, 4], for some N (N==0 is also valid)
    labels: torch.Tensor  # dimension should be [N]
    label_strs: list[str]  # classes as strings, length should be [N]


class Prediction(TypedDict):
    """Represents what comes out of FasterRCNN model.

    These are parallel tensors with one element per detection box.
    """

    boxes: torch.Tensor  # dimension should be [N, 4], for some N (N==0 is also valid)
    scores: torch.Tensor  # dimension N
    labels: torch.Tensor  # class ids of detected objects
