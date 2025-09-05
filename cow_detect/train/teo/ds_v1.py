# Define the dataset class for a custom dataset
import math
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import torch
import torchvision.transforms  # type: ignore[import-untyped]
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from cow_detect.utils.annotations import parse_json_annotations_file

# Type produced by a detection model in train mode
TargetType: TypeAlias = dict[str, torch.Tensor]
TransformType: TypeAlias = Callable[[torch.Tensor, TargetType], tuple[torch.Tensor, TargetType]]


class SkyDataset(Dataset):
    """Dataset class currently adapted to read the sky dataset and json annotations."""

    def __init__(
        self,
        name: str,
        root_dir: Path,
        image_paths: list[Path],
        annot_subdir: str = "ann",
        transforms: TransformType | None = None,  # TODO: what should the type really be?
    ) -> None:
        self.name = name
        self.root_dir = root_dir
        self.transforms = transforms
        self.annotation_dir = root_dir / annot_subdir

        # all_paths = list(self.image_dir.glob(f"*.{ext}"))
        logger.info(f"dataset: {self.name} - image_paths has : {len(image_paths)}")
        fnames = [path.name for path in image_paths]
        fnames_first5 = " ".join(fnames[0:5])
        fnames_last5 = " ".join(fnames[-5:])
        logger.info(f"dataset: {self.name} - fnames : [{fnames_first5}...{fnames_last5}]")

        broken_imgs_file = root_dir / "broken-imgs.txt"
        if broken_imgs_file.exists():
            broken_imgs = {Path(line) for line in broken_imgs_file.read_text().split("\n")}
            logger.info(
                f"Found {broken_imgs_file}, containing {len(broken_imgs)} broken image paths"
            )
        else:
            broken_imgs = set()

        self.image_paths = [fp for fp in image_paths if fp not in broken_imgs]
        logger.info(
            f"dataset: {self.name} - after excluding broken images: {len(self.image_paths)}"
        )

        self.class_name_to_id = {"cattle": 1}  # Class 0 is reserved for background
        self.img_to_tensor = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.image_paths)

    def num_batches(self, batch_size: int) -> int:
        """Get the number of batches per epoch."""
        return int(math.ceil(len(self.image_paths) / batch_size))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], Path]:
        """Get the i-th item from this dataset."""
        img_path = self.image_paths[idx]
        # This simple rule works for SKY but it doesnt work for ICAERUS
        annotation_path = self.annotation_dir / (img_path.name + ".json")

        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            logger.info(f"Could not open image file: {img_path!s}")
            raise

        image_pt = self.img_to_tensor(image)
        del image

        bboxes0, label_strs = parse_json_annotations_file(annotation_path)
        boxes, labels = filter_bboxes_for_classes(bboxes0, label_strs, self.class_name_to_id)
        del bboxes0, label_strs

        target = {}

        if len(boxes) > 0:
            boxes_pt = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes_pt = torch.zeros((0, 4), dtype=torch.float32)

        target["boxes"] = boxes_pt
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes_pt[:, 3] - boxes_pt[:, 1]) * (boxes_pt[:, 2] - boxes_pt[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms:
            image_pt, target = self.transforms(image_pt, target)

        return image_pt, target, img_path


def filter_bboxes_for_classes(
    boxes0: list, label_strs: list[str], cls_name_to_id: dict[str, int]
) -> tuple[list[list[int]], list[str]]:
    """Filter bboxes and their corresponding labels.

    Keep only boxes corresponding to labels that are keys in cls_name_to_id.
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
