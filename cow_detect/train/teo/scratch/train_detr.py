from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

from cow_detect.train.teo.scratch.cow_dataset_hf import CowDataset
from cow_detect.utils.pytorch import auto_detect_device


def train_detr(
    image_dir: Path = Path("data/sky/Dataset2/img"),
    annot_dir: Path = Path("data/sky/Dataset2/ann"),
    learning_rate: float = 1e-5,
    num_epochs: int = 10,
    batch_size: int = 2,
):
    """Trains a Detr model.

    Note: which is not really appropriate for detecting objects that
    are very small compared to the dimensions of the image
    """
    # Set up the model, optimizer, and dataset
    device = auto_detect_device()

    # Use DetrForObjectDetection from Hugging Face
    model_name = "facebook/detr-resnet-50"
    image_processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_epochs = 10

    # Create the dataset and dataloader
    dataset = CowDataset(image_dir=image_dir, annot_dir=annot_dir, image_processor=image_processor)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=detr_custom_collate_fn
    )

    # Fine-tuning loop
    model.train()
    for epoch in range(num_epochs):
        for i, batch in enumerate(data_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "detr_cow_detection.pth")
    print("Fine-tuning complete. Model saved.")


# Custom collate function to handle the variable number of boxes
def detr_custom_collate_fn(batch):
    """Custom collate function for HF Detr model."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}
