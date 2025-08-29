# Define the dataset class for a custom dataset
from pathlib import Path

import torch
import torchvision.transforms
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from cow_detect.utils.annotations import parse_json_annotations_file


class SkyDataset(Dataset):
    """Dataset class currently adapted to read the sky dataset and json annotations."""

    def __init__(
        self,
        name: str,
        root_dir: Path,
        image_paths: list[Path],
        annot_subdir: str = "ann",
        transforms=None,
    ):
        self.name = name
        self.root_dir = root_dir
        self.transforms = transforms
        # self.image_dir = root_dir / img_subdir
        self.annotation_dir = root_dir / annot_subdir

        # all_paths = list(self.image_dir.glob(f"*.{ext}"))
        logger.info(f"dataset: {self.name} - image_paths has : {len(image_paths)}")

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

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
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

        assert annotation_path.exists()
        boxes, labels = parse_json_annotations_file(annotation_path, self.class_name_to_id)

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

        return image_pt, target
