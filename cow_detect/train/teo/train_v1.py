from pathlib import Path

import torch
import torchvision
import typer
import yaml
from loguru import logger
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)
from typer import Typer

from cow_detect.train.teo.cow_ds_v1 import CowDataset
from cow_detect.utils.config import DataLoaderParams, OptimizerParams
from cow_detect.utils.metrics import calculate_iou
from cow_detect.utils.pytorch import auto_detect_device

# %%

cli = Typer(pretty_exceptions_show_locals=False)


def get_model(num_classes: int) -> FasterRCNN:
    """Get a faster-rcnn model with a box_predictor head for the given number of classes."""
    # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one that has the number of classes we need
    # (plus 1 for the background class)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def faster_rcnn_custom_collate_fn(batch):
    """Custom batch collate function.

    TODO: check this does the right thing...
    """
    return tuple(zip(*batch, strict=False))


def _interactive_test():
    # %%
    from PIL import Image

    img = Image.open("data/sky/Dataset1/img/auto2_DJI_0065.JPG").convert("RGB")
    print(img)
    # %%


class TrainCfg(BaseModel):
    """Parameters fort training."""

    data_root_dir: Path = Path("data/sky/Dataset1")
    data_loader: DataLoaderParams
    num_epochs: int
    optimizer: OptimizerParams


@cli.command()
def train_faster_rcnn(
    train_cfg_path: Path = typer.Option(..., "--cfg", help="where to get the config from"),
    save_path: Path = typer.Option(..., "--save-path", "-o", help="where to leave the final model"),
    print_every_batches: int = typer.Option(3, "-p", help="print error metrics these many batches"),
):
    """Train a faster rcnn model with a given config and leaving result in a given model_path."""
    # Set up the device
    # device = auto_detect_device()
    device = torch.device("cpu")

    cfg_dict = yaml.load(train_cfg_path.open("rt"), Loader=yaml.Loader)
    train_cfg: TrainCfg = TrainCfg.model_validate(cfg_dict)

    # Create the dataset and dataloader
    dataset = CowDataset(
        root_dir=train_cfg.data_root_dir,
    )

    dl_params = train_cfg.data_loader
    data_loader = DataLoader(
        dataset,
        batch_size=dl_params.batch_size,
        shuffle=dl_params.data_shuffle,
        num_workers=dl_params.num_workers,
        collate_fn=faster_rcnn_custom_collate_fn,
    )

    # Load the model
    # Define number of classes (cow + background)
    num_classes = 2

    model = get_model(num_classes)
    model.to(device)

    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    opt_params = train_cfg.optimizer
    optimizer = torch.optim.SGD(
        params,
        lr=opt_params.learning_rate,
        momentum=opt_params.momentum,
        weight_decay=opt_params.weight_decay,
    )

    # Training loop
    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        for i, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Make predictions in forward mode so that we can calculate IOU metric
            with torch.no_grad():
                model.eval()
                predictions = model(images)
                model.train()

            mean_iou = calculate_iou(predictions, targets)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % print_every_batches == 0:
                logger.info(
                    f"Epoch: {epoch}, Batch: {i}, Loss: {losses.item():.4f},"
                    f" Mean IoU: {mean_iou:.4f}"
                )

    # Save the fine-tuned model
    torch.save(model.state_dict(), save_path)
    logger.info(f"Fine-tuning complete. Model saved to: {save_path!s}")


if __name__ == "__main__":
    cli()
