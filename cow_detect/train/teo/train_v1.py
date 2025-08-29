import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torchvision
import tqdm
import typer
import yaml
from loguru import logger
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)
from typer import Typer

from cow_detect.train.teo.ds_v1 import SkyDataset
from cow_detect.utils.config import DataLoaderParams, OptimizerParams
from cow_detect.utils.metrics import calculate_iou
from cow_detect.utils.train import train_validation_split

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

    train_fraction: float
    valid_fraction: float
    data_loader: DataLoaderParams
    num_epochs: int
    optimizer: OptimizerParams


class Trainer:
    """Provides more ergonomic training and validation loops."""

    def __init__(
        self,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        # print_every_batches: int = 10,
    ) -> None:
        self.device = device
        self.optimizer = optimizer
        # self.print_every_batches = print_every_batches

    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        train_data_loader: DataLoader,
    ) -> None:
        """Run loop over train data for one epoch."""
        train_losses = []
        train_ious = []

        n_batches = len(train_data_loader.dataset) // train_data_loader.batch_size
        pbar = tqdm.tqdm(train_data_loader, total=n_batches)
        for images, targets in pbar:
            # logger.info(f"batch: {i} images: {len(images)} targets: {len(targets)}")
            model.train()
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            # Make predictions in forward mode so that we can calculate IOU metric
            with torch.no_grad():
                model.eval()
                predictions = model(images)
                model.train()

            mean_iou = calculate_iou(predictions, targets)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            train_losses.append(total_loss.item())
            train_ious.append(mean_iou)

            pbar.set_description(
                f"Epoch {epoch}: training: avg.loss={np.mean(train_losses):.4f}"
                f", avg.mean.iou={np.mean(train_ious):.4f}"
            )
            # if i % self.print_every_batches == 0:
            #    logger.info(
            #        f"Epoch {epoch} - Batch: {i} - Total Loss: {total_loss.item():.4f},"
            #        f" Mean IoU: {mean_iou:.4f}"
            #    )

        avg_train_loss = np.mean(train_losses)
        avg_train_iou = np.mean(train_ious)
        logger.info(f"Epoch {epoch}: {avg_train_loss=:.4}, {avg_train_iou=:.4f}")

    def validate_epoch(
        self,
        epoch: int,
        model: nn.Module,
        valid_data_loader: DataLoader,
    ) -> None:
        """Run loop over validation data after an epoch."""
        model.eval()
        valid_scores = []
        valid_ious = []

        n_batches = len(valid_data_loader.dataset) // valid_data_loader.batch_size
        with torch.no_grad():
            pbar = tqdm.tqdm(valid_data_loader, total=n_batches)
            for images, targets in pbar:
                # logger.info(f"batch: {i} images: {len(images)} targets: {len(targets)}")
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                prediction_dicts = model(images, targets)
                # In eval mode there are no losses but: boxes, labels, confidence scores
                try:
                    scores = [torch.mean(d["scores"]).item() for d in prediction_dicts]
                    avg_score = np.nanmean(scores)
                except (KeyError, TypeError, ValueError, RuntimeError):
                    pprint(prediction_dicts)
                    raise
                # predictions = model(images)
                mean_iou = calculate_iou(prediction_dicts, targets)

                valid_scores.append(avg_score)
                valid_ious.append(mean_iou)
                pbar.set_description(
                    f"Epoch {epoch}: Validation: avg.score={np.mean(valid_scores):.4f}"
                    f", avg.mean.iou={np.mean(valid_ious):.4f}"
                )

        avg_valid_score = np.mean(valid_scores)
        avg_valid_iou = np.mean(valid_ious)
        logger.info(f"Epoch {epoch}: VALIDATION : {avg_valid_score=:.4}, {avg_valid_iou=:.4f}")


@cli.command()
def train_faster_rcnn(
    train_cfg_path: Path = typer.Option(..., "--cfg", help="where to get the config from"),
    train_data: Path = typer.Option(..., "--train-data", help="where to get the data from"),
    save_path: Path = typer.Option(..., "--save-path", "-o", help="where to leave the final model"),
    print_every_batches: int = typer.Option(3, "-p", help="print error metrics these many batches"),
):
    """Train a faster rcnn model with a given config and leaving result in a given model_path."""
    # Set up the device
    # device = auto_detect_device()
    device = torch.device("cpu")

    cfg_dict = yaml.load(train_cfg_path.open("rt"), Loader=yaml.Loader)
    train_cfg: TrainCfg = TrainCfg.model_validate(cfg_dict)

    train_img_paths, valid_img_paths = train_validation_split(
        imgs_dir=train_data / "img",
        train_fraction=train_cfg.train_fraction,
        valid_fraction=train_cfg.valid_fraction,
    )
    # Create the dataset and dataloader
    train_data_set = SkyDataset(
        name="train",
        root_dir=train_data,
        image_paths=train_img_paths,
    )
    valid_data_set = SkyDataset(
        name="valid",
        root_dir=train_data,
        image_paths=valid_img_paths,
    )

    dl_params = train_cfg.data_loader
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=dl_params.batch_size,
        shuffle=dl_params.data_shuffle,
        num_workers=dl_params.num_workers,
        collate_fn=faster_rcnn_custom_collate_fn,
    )

    valid_data_loader = DataLoader(
        valid_data_set,
        batch_size=10,
        shuffle=False,
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

    trainer = Trainer(
        device=device,
        optimizer=optimizer,
        # print_every_batches=print_every_batches,
    )

    # Training loop
    num_epochs = train_cfg.num_epochs

    for epoch in range(num_epochs):
        # Train loop:
        trainer.train_epoch(epoch, model, train_data_loader)
        trainer.validate_epoch(epoch, model, valid_data_loader)

    # Save the fine-tuned model
    torch.save(model.state_dict(), save_path)
    logger.info(f"Fine-tuning complete. Model saved to: {save_path!s}")
    sys.exit(0)


if __name__ == "__main__":
    cli()
