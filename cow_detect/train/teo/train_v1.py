#!/usr/bin/env python
import gc
import json
import os
from hashlib import md5
from pathlib import Path
from pprint import pprint
from typing import Annotated

import mlflow
import numpy as np
import torch
import torchvision  # type: ignore[import-untyped]
import tqdm
import typer
import yaml
from loguru import logger
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (  # type: ignore[import-untyped]
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)
from typer import Typer

from cow_detect.datasets.sky_v1 import SkyDataset
from cow_detect.utils.config import DataLoaderParams, OptimizerParams
from cow_detect.utils.metrics import calculate_iou
from cow_detect.utils.train import get_num_batches, train_validation_split
from cow_detect.utils.versioning import get_cfg_hash, get_git_revision_hash

# %%

cli = Typer(pretty_exceptions_show_locals=False)

# Set mlflow backed file store under data dir
mlflow.set_tracking_uri(f"file://{os.getcwd()}/data/mlruns")


def get_model(num_classes: int) -> FasterRCNN:  # type: ignore[no-any-unimported]
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


def faster_rcnn_custom_collate_fn(batch: list[object]) -> tuple:
    """Custom batch collate function.

    Verified!
    Run: marimo edit research/try-out-dataloading.py
    """
    return tuple(zip(*batch, strict=True))


def _interactive_test() -> None:
    # %%
    from PIL import Image

    img = Image.open("data/sky/Dataset1/img/auto2_DJI_0065.JPG").convert("RGB")
    print(img)
    # %%


class TrainCfg(BaseModel):
    """Parameters fort training."""

    experiment_name: str
    sort_paths: Annotated[
        bool,
        Field(
            description="whether to sort paths from input dataset before splitting,"
            "default true for greater reproducibility",
        ),
    ] = True

    train_fraction: float | None = None
    valid_fraction: float | None = None
    data_loader: DataLoaderParams
    num_epochs: int
    optimizer: OptimizerParams
    device: str = "cpu"
    num_classes: int = 2
    max_detection_thresholds: Annotated[
        list[int],
        Field(
            default_factory=lambda: [3, 5, 10, 100],
            description="thresholds used for calculating mAR metrics",
        ),
    ]


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

        n_batches = get_num_batches(train_data_loader)
        pbar = tqdm.tqdm(train_data_loader, total=n_batches)
        pbar.set_description("Epoch 0: Training: avg.loss=..... avg.mean.iou=.....")
        for images, targets, _file_paths in pbar:
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

            mean_iou, num_iou = calculate_iou(predictions, targets)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            train_losses.append(total_loss.detach().item())
            train_ious.append(mean_iou)

            pbar.set_description(
                f"Epoch {epoch}: training: avg.loss={np.mean(train_losses):.4f}"
                f", avg.mean.iou={np.mean(train_ious):.4f} num_iou={num_iou}"
            )

        avg_train_loss = np.mean(train_losses)
        avg_train_iou = np.mean(train_ious)
        mlflow.log_metric("avg_train_loss", float(avg_train_loss), step=epoch)
        mlflow.log_metric("avg_train_iou", float(avg_train_iou), step=epoch)
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

        n_batches = get_num_batches(valid_data_loader)
        with torch.no_grad():
            pbar = tqdm.tqdm(valid_data_loader, total=n_batches)
            pbar.set_description("Epoch 0: Validation: avg.loss=..... avg.mean.iou=.....")

            for images, targets, _fpaths in pbar:
                # logger.info(f"batch: {i} images: {len(images)} targets: {len(targets)}")
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                prediction_dicts = model(images, targets)
                # In eval mode there are no losses but: boxes, labels, confidence scores
                try:
                    scores = [torch.mean(d["scores"]).detach().item() for d in prediction_dicts]
                    avg_score = np.nanmean(scores)
                except (KeyError, TypeError, ValueError, RuntimeError):
                    pprint(prediction_dicts)
                    raise

                mean_iou, num_iou = calculate_iou(prediction_dicts, targets)

                valid_scores.append(avg_score)
                valid_ious.append(mean_iou)
                pbar.set_description(
                    f"Epoch {epoch}: Validation: avg.score={np.mean(valid_scores):.4f}"
                    f", avg.mean.iou={np.mean(valid_ious):.4f}, num_iou={num_iou}"
                )

        avg_valid_score = np.mean(valid_scores)
        avg_valid_iou = np.mean(valid_ious)
        mlflow.log_metric("avg_valid_score", float(avg_valid_score), step=epoch)
        mlflow.log_metric("avg_valid_iou", float(avg_valid_iou), step=epoch)
        logger.info(f"Epoch {epoch}: VALIDATION : {avg_valid_score=:.4}, {avg_valid_iou=:.4f}")


def save_model_and_version(
    model: nn.Module, train_cfg: TrainCfg, git_revision: str, save_path: Path
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model.pth")
    (save_path / "train-config.yaml").write_text(yaml.dump(train_cfg))
    (save_path / "versioning.txt").write_text(
        json.dumps(
            {
                "git_revision": git_revision,
                "cfg_hash": get_cfg_hash(train_cfg.model_dump_json()),
            }
        )
    )
    logger.info(f"Fine-tuning complete. Model saved to: {save_path!s}")


@cli.command()
def train_faster_rcnn(
    train_cfg_path: Path = typer.Option(..., "--cfg", help="where to get the config from"),
    train_data_path: Path = typer.Option(..., "--train-data", help="where to get the data from"),
    save_path: Path | None = typer.Option(
        None,
        "--save-path",
        "-o",
        help="directory where to save model, code revision and full-params file.",
    ),
) -> None:
    """Train a faster rcnn model with a given config and leaving result in a given model_path."""
    logger.info(f"Reading train_cfg from: {train_cfg_path}")
    cfg_dict = yaml.load(train_cfg_path.open("rt"), Loader=yaml.Loader)
    train_cfg: TrainCfg = TrainCfg.model_validate(cfg_dict)

    logger.info(f"Using device={train_cfg.device} as specified in train_cfg.")
    device = torch.device(train_cfg.device)
    assert train_cfg.valid_fraction is not None

    train_img_paths, valid_img_paths = train_validation_split(
        imgs_dir=train_data_path / "img",
        sort_paths=train_cfg.sort_paths,
        train_fraction=train_cfg.train_fraction,
        valid_fraction=train_cfg.valid_fraction,
    )
    # Create the dataset and dataloader
    train_data_set = SkyDataset(
        name="train",
        root_dir=train_data_path,
        image_paths=train_img_paths,
    )
    valid_data_set = SkyDataset(
        name="valid",
        root_dir=train_data_path,
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
        batch_size=2,
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
    )

    # Training loop
    num_epochs = train_cfg.num_epochs

    git_revision = get_git_revision_hash()
    experiment = mlflow.set_experiment(train_cfg.experiment_name)
    cfg_md5 = md5(train_cfg_path.read_bytes()).hexdigest()

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_param("data_set", str(train_data_path))
        mlflow.log_param("git_revision_12", git_revision[:12])
        mlflow.log_param("cfg_md5", cfg_md5)
        mlflow.log_param("cfg_path", str(train_cfg_path))
        mlflow.log_param("cfg_full", str(train_cfg_path.read_text()))
        mlflow.log_param("model_class", type(model).__name__)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("optimizer_class", type(optimizer).__name__)
        mlflow.log_param("lr", opt_params.learning_rate)
        mlflow.log_param("momentum", opt_params.momentum)
        mlflow.log_param("weight_decay", opt_params.weight_decay)
        mlflow.log_param("batch_size", dl_params.batch_size)
        mlflow.log_param("num_workers", dl_params.num_workers)
        mlflow.log_param("device", str(device))

        for epoch in range(num_epochs):
            # Train loop:
            trainer.train_epoch(epoch, model, train_data_loader)
            torch.cuda.empty_cache()
            gc.collect()
            trainer.validate_epoch(epoch, model, valid_data_loader)
            torch.cuda.empty_cache()
            gc.collect()

    # Save the fine-tuned model
    if save_path is not None:
        save_model_and_version(
            model, train_cfg=train_cfg, git_revision=git_revision, save_path=save_path
        )


if __name__ == "__main__":
    cli()
