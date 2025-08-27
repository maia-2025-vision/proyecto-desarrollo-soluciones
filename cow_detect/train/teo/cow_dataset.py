from pathlib import Path
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from cow_detect.utils.annotations import parse_json_annotations_file


# Define the dataset class
class CowDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        annot_dir: Path,
        image_processor,
        transform=None
    ) -> None:
        self.image_processor = image_processor
        self.transform = transform
        self.image_dir = image_dir
        self.annotation_dir = annot_dir
        # self.annotation_format = annot_format
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.JPG')]
        self.class_name = 'cow'
        self.label_map = {self.class_name: 0}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name
        annotation_path = self.annotation_dir /  img_path.'.json'

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        if annotation_path.exists():
            boxes, labels = parse_json_annotations_file(annotation_path)
        else: # no objects in this image...
            boxes, labels = [], []
            # Parse yolo format
            # parse_yolo_annotation_file(annotation_path, img_width=width, img_height=height)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                    target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transform:
            image, target = self.transform(image, target)

        # Format the target for the Hugging Face model
        encoded_inputs = self.image_processor(
            images=image,
            annotations=[{'boxes': boxes, 'class_labels': labels}],
            return_tensors='pt'
        )

        # The Hugging Face model expects a specific format for the target
        target_hf = {
            'pixel_values': encoded_inputs['pixel_values'].squeeze(0),
            'labels': encoded_inputs['labels'][0]
        }

        return target_hf


# Custom collate function to handle the variable number of boxes
def custom_collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = [item['labels'] for item in batch]
    return {'pixel_values': pixel_values, 'labels': labels}



