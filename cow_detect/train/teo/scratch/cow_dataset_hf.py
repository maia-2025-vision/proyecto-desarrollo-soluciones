import os
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from cow_detect.datasets.sky_v1 import TargetType
from cow_detect.utils.annotations import parse_json_annotations_file

TransformType: type = Callable[[Image.Image, TargetType], tuple[torch.Tensor, TargetType]]


# Define the dataset class
class CowDatasetHF(Dataset):
    """Read images and annotations from SKY dataset.

    Also process them so that they can be fed into a HuffingFace model
    """

    def __init__(
        self,
        image_dir: Path,
        annot_dir: Path,
        image_processor,
        transform: TransformType | None = None,  # Pending: right type?
        class_name_to_id: dict[str, int] | None = None,
    ) -> None:
        self.image_processor = image_processor
        self.transform = transform
        self.image_dir = image_dir
        self.annotation_dir = annot_dir
        # self.annotation_format = annot_format
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".JPG")]
        # self.class_name = 'cow'
        # self.label_map = {self.class_name: 0}

        self.class_name_to_id = class_name_to_id or {"cattle": 0}

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Get the idx-th item."""
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        annotation_path = self.annotation_dir / (img_path.name + ".json")

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        if annotation_path.exists():
            boxes, labels = parse_json_annotations_file(
                annotation_path, class_name_to_id=self.class_name_to_id
            )
            # To Parse yolo format
            # boxes, labels = parse_yolo_annotation_file(
            #   annotation_path, img_width=width, img_height=height
            # )
        else:  # no objects in this image...
            boxes, labels = [], []

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
            target["boxes"][:, 2] - target["boxes"][:, 0]
        )
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transform is not None:
            image, target = self.transform(image, target)

        # Format the target for the Hugging Face model
        encoded_inputs = self.image_processor(
            images=image,
            annotations=[{"boxes": boxes, "class_labels": labels}],
            return_tensors="pt",
        )

        # The Hugging Face model expects a specific format for the target
        target_hf = {
            "pixel_values": encoded_inputs["pixel_values"].squeeze(0),
            "labels": encoded_inputs["labels"][0],
        }

        return target_hf
