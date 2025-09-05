import math
from pathlib import Path

import torch
import torchvision.transforms  # type: ignore[import-untyped]
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from cow_detect.utils.standardize import Annot, AnnotatedImageRecord, AnnotationsFileInfo


class AnnotatedImagesDataset(Dataset):
    """Dataset class used for evaluation."""

    def __init__(
        self,
        std_data_path: Path,
        label_to_id: dict[str, int],
        force_resize: bool = True,
        limit: int | None = None,
        limit_fraction: float | None = None,
        name: str = "<unnamed>",
    ) -> None:
        """A torch dataset that provides images as tensors together with their paths.

        :param std_data_path: a path to a jsonl file such produced by cow_detect.utils.standardize
            see for instance files under data/standardized/
        :param force_resize: Whether to force resizing of images
        :param limit: If not None, limit to this maximum number of images to return
        (for quicker testing)
        """
        self.std_data_path = std_data_path
        self.label_to_id = label_to_id
        self.name = name

        data_lines = self.std_data_path.read_text().splitlines()
        logger.info(f"{len(data_lines)} read from {std_data_path}")
        records = [
            AnnotatedImageRecord.model_validate_json(line)
            for line in data_lines
            if line.strip() != ""
        ]
        # filter ok records only
        self.records: list[AnnotatedImageRecord] = [
            rec for rec in records if rec.image.status == "ok" and rec.annotations.status == "ok"
        ]

        logger.info(f"Ok records: {len(self.records)}")

        if limit_fraction is not None:
            limit = int(math.ceil(len(self.records) * limit_fraction))
            logger.info(f"Limiting dataset to {limit} records (fraction: {limit_fraction})")
        elif limit is not None:
            logger.info(f"Limiting dataset to {limit} records")
        else:  # do not limit
            limit = len(self.records)

        self.records = self.records[:limit]
        self.img_to_tensor = torchvision.transforms.ToTensor()
        self.target_size: tuple[int, int] | None = None
        self.force_resize: bool = force_resize

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.records)

    def num_batches(self, batch_size: int) -> int:
        """Get the number of batches per epoch."""
        return int(math.ceil(len(self.records) / batch_size))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | Path | list[str]]:
        """Get the i-th image from this dataset as torch tensor along with its path."""
        img_path = self.records[idx].image.path

        try:
            image = Image.open(img_path).convert("RGB")
            if self.target_size is None:
                self.target_size = image.size
                logger.info(
                    f"After successfully loaded image, set target_size to: {self.target_size}"
                )
            else:
                if image.size != self.target_size:
                    if self.force_resize:
                        logger.info(f"Resizing image from {image.size} to {self.target_size}")
                        image = image.resize(self.target_size)
                    else:
                        logger.warning(
                            f"Image loaded from {img_path!s} has size {image.size} "
                            f"which differs from {self.target_size}. "
                            f"If you are processing batches of size > 1, "
                            f"this will cause an error downstream."
                        )

        except OSError as err:
            logger.warning(
                f"Could not open image file: {img_path!s}, error: {err!s}, instead, will return "
                f"fully default black image of size {self.target_size}"
            )
            assert self.target_size is not None
            image = Image.new("RGB", self.target_size, (0, 0, 0))

        image_tensor = self.img_to_tensor(image)
        del image

        annot_info: AnnotationsFileInfo = self.records[idx].annotations
        annots: list[Annot] | None = annot_info.annots
        assert (
            annots is not None
        ), f"annots is not populated in annot_info: {annot_info.model_dump_json()}"
        bboxes: list[list[float]] = [annot.coords for annot in annots]
        label_strs: list[str] = [annot.label for annot in annots]
        label_ids: list[int] = [self.label_to_id[lbl] for lbl in label_strs]

        if len(bboxes) > 0:
            boxes = torch.tensor(bboxes, dtype=torch.float32)
        else:  # make sure boxes has dim=2 even if first dimension is zero..
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "image_path": img_path,
            "image_pt": image_tensor,
            "boxes": boxes,
            "label_strs": label_strs,
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
