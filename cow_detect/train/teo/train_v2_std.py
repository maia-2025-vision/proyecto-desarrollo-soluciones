import gc
import io
import json
from pathlib import Path

import mlflow
import numpy as np
import torch
import tqdm
import typer
import yaml
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.detection import intersection_over_union, mean_average_precision

from cow_detect.datasets.std import AnnotatedImagesDataset
from cow_detect.train.teo.train_v1 import TrainCfg, get_model, save_model_and_version
from cow_detect.utils.data import custom_collate_dicts, make_jsonifiable_singletons, zip_dict
from cow_detect.utils.debug import summarize

# from cow_detect.utils.metrics import calculate_iou
from cow_detect.utils.mlflow_utils import log_mapr_metrics, log_params_v1
from cow_detect.utils.pytorch import detach_dicts
from cow_detect.utils.train import get_num_batches
from cow_detect.utils.versioning import get_git_revision_hash

cli = typer.Typer(pretty_exceptions_show_locals=False, no_args_is_help=True)


class TrainerV2:
    """Provides more ergonomic training and validation loops.

    Now using standardized input dataset (from jsonl file)
    """

    def __init__(
        self,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        max_detection_thresholds: list[int],  # used to measure mAR
    ) -> None:
        self.device = device
        self.optimizer = optimizer
        self.max_detection_thresholds = max_detection_thresholds

    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        train_data_loader: DataLoader,
    ) -> None:
        """Run loop over train data for one epoch."""
        train_losses = []  # one per batch
        train_ious: list[float] = []  # one per image
        all_targets: list[dict] = []
        all_predictions: list[dict] = []

        n_batches = get_num_batches(train_data_loader)
        pbar = tqdm.tqdm(train_data_loader, total=n_batches)

        for batch in pbar:
            # pbar.set_description("Epoch 0: Training: avg.loss=..... avg.mean.iou=.....")
            model.train()
            images = [image.to(self.device) for image in batch["image_pt"]]
            targets: list[dict] = zip_dict(batch)  # type: ignore[arg-type]
            all_targets.extend(targets)

            loss_dict = model(images, targets)
            total_loss: torch.Tensor = sum(loss for loss in loss_dict.values())

            # Make predictions in forward mode so that we can calculate IOU and mAP/R metrics
            with torch.no_grad():
                model.eval()
                predictions = detach_dicts(model(images))
                all_predictions.extend(predictions)
                model.train()

            # Old iou calculation
            # mean_iou = calculate_iou(predictions, targets)
            ious_batch = [
                intersection_over_union(pred["boxes"], tgt["boxes"]).detach().item()
                for pred, tgt in zip(predictions, targets, strict=False)
            ]
            # print("ious_batch:", summarize(ious_batch))
            train_ious.extend(ious_batch)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            train_losses.append(total_loss.detach().item())

            pbar.set_description(
                f"Epoch {epoch}: training: mean.loss={np.mean(train_losses):.4f}, "
                f"mean.train.iou={np.nanmean(train_ious):.4f}"
            )

        mapr_metrics_raw = mean_average_precision(  # type: ignore[arg-type]
            preds=all_predictions,
            target=all_targets,
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=self.max_detection_thresholds,
        )

        mapr_metrics = make_jsonifiable_singletons(
            mapr_metrics_raw,  # type: ignore[arg-type]
            negative_to_nan=True
        )

        mean_train_loss = np.mean(train_losses)
        mean_train_iou = np.nanmean(train_ious)
        mlflow.log_metric("mean_train_loss", float(mean_train_loss), step=epoch)
        mlflow.log_metric("mean_train_iou", float(mean_train_iou), step=epoch)
        logged_maprs = log_mapr_metrics(
            mapr_metrics, prefix="train", max_detect_thresholds=self.max_detection_thresholds
        )
        logger.info(f"Epoch {epoch}: {mean_train_loss=:.4}, {mean_train_iou=:.4f}, {logged_maprs=}")

    def validate_epoch(
        self,
        epoch: int,
        model: nn.Module,
        valid_data_loader: DataLoader,
    ) -> None:
        """Run loop over validation data after an epoch."""
        model.eval()
        # valid_scores = []
        valid_ious = []

        # list of targets and predictions to use torchmetrics api below
        # each of these lists should have as many elements as images in validation_dataset
        all_targets: list[dict[str, torch.Tensor]] = []
        all_predictions: list[dict[str, torch.Tensor]] = []

        n_batches = get_num_batches(valid_data_loader)
        with torch.no_grad():
            pbar = tqdm.tqdm(valid_data_loader, total=n_batches)
            pbar.set_description("Epoch 0: VALIDATION: avg.mean.iou=.....")
            for batch in pbar:
                images = [image.to(self.device) for image in batch["image_pt"]]
                targets: list[dict] = zip_dict(batch)  # type: ignore[arg-type]
                all_targets.extend(targets)

                # PREDICTION happens here:
                predictions: list[dict] = detach_dicts(model(images, targets))
                all_predictions.extend(predictions)

                ious_batch = [
                    intersection_over_union(pred["boxes"], tgt["boxes"]).detach().item()
                    for pred, tgt in zip(predictions, targets, strict=False)
                ]
                valid_ious.extend(ious_batch)

                pbar.set_description(
                    f"Epoch {epoch}: VALIDATION:  avg.mean.iou={np.mean(valid_ious):.4f}"
                )

        mapr_metrics_raw = mean_average_precision(
            preds=all_predictions,
            target=all_targets,
            box_format="xyxy",
            iou_type="bbox",
            max_detection_thresholds=self.max_detection_thresholds,
        )

        mapr_metrics = make_jsonifiable_singletons(
            mapr_metrics_raw,  # type: ignore[arg-type]
            negative_to_nan=True
        )

        logger.info(
            f"Epoch {epoch} - VALIDATION mAP/R metrics:\n{json.dumps(mapr_metrics, indent=2)}"
        )

        mean_valid_iou = np.nanmean(valid_ious)
        non_nan_ious = int((~np.isnan(valid_ious)).sum())

        mlflow.log_metric("mean_valid_iou", float(mean_valid_iou), step=epoch)
        logged_mapr = log_mapr_metrics(
            mapr_metrics, prefix="valid", max_detect_thresholds=self.max_detection_thresholds
        )
        logger.info(
            f"Epoch {epoch}: VALIDATION : {mean_valid_iou=:.4}, {non_nan_ious=}, {logged_mapr=}"
        )


DEFAULT_LABEL_TO_ID: dict[str, int] = {"cow": 1, "cattle": 1, "calf": 1}


@cli.command()
def train_v2_std_cli(
    train_cfg_path: Path = typer.Option(..., "--cfg", help="where to get the config from"),
    train_data_path: Path = typer.Option(..., "--train-data", help="where to get train data from"),
    valid_data_path: Path = typer.Option(..., "--valid-data", help="where to get valid. data from"),
    device: str | None = typer.Option(
        None,
        "--device",
        help="accelerator device to use, if specified overrides the one from train_cfg",
    ),
    num_epochs: int | None = typer.Option(
        None,
        "--num_epochs",
        help="num_epochs to run, if specified overrides the value from train_cfg",
    ),
    save_path: Path | None = typer.Option(
        None,
        "--save-path",
        "-o",
        help="directory where to save model, code revision and full-params file.",
    ),
) -> None:
    """Train a faster rcnn model with a given config and leaving result in a given model_path."""
    if save_path is None:
        logger.warning("save_path not set, model won't be saved to disk at the end!")

    logger.info(f"Reading train_cfg from: {train_cfg_path}")
    train_cfg_text = train_cfg_path.read_text()

    cfg_dict = None
    try:
        cfg_dict = yaml.safe_load(io.StringIO(train_cfg_text))
        train_cfg: TrainCfg = TrainCfg.model_validate(cfg_dict)
    except Exception:
        logger.error(f"Failed to parse load or parse train_cfg:\n{json.dumps(cfg_dict, indent=2)}")
        raise

    logger.info(f"Train cfg (full including defaults):\n{train_cfg.model_dump_json(indent=2)}\n")
    device = device or train_cfg.device
    assert device is not None
    logger.info(f"Using device: {device}")
    # TODO: make this configurable?
    label_to_id = DEFAULT_LABEL_TO_ID
    logger.info(f"Using label to id: {json.dumps(label_to_id)}")

    # Create the dataset and dataloader
    train_data_set = AnnotatedImagesDataset(
        name="train",
        std_data_path=train_data_path,
        label_to_id=label_to_id,
        limit_fraction=train_cfg.train_fraction,
    )
    valid_data_set = AnnotatedImagesDataset(
        name="valid",
        std_data_path=valid_data_path,
        label_to_id=label_to_id,
        limit_fraction=train_cfg.valid_fraction,
    )

    dl_params = train_cfg.data_loader
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=dl_params.batch_size,
        shuffle=dl_params.data_shuffle,
        num_workers=dl_params.num_workers,
        collate_fn=custom_collate_dicts,
    )
    valid_data_loader = DataLoader(
        valid_data_set,
        batch_size=dl_params.valid_batch_size,
        shuffle=False,
        num_workers=dl_params.num_workers,
        collate_fn=custom_collate_dicts,
    )

    # Load the model
    # Define number of classes (cow + background)
    logger.info(f"Training model with num_classes={train_cfg.num_classes}")

    model = get_model(train_cfg.num_classes)
    model.to(device)

    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    opt_params = train_cfg.optimizer
    # TODO: experiment with other optimizers? Adam, AdamW?
    assert opt_params.optimizer_class == "SGD", f"{opt_params.optimizer_class=} not yet supported"
    optimizer = torch.optim.SGD(
        params,
        lr=opt_params.learning_rate,
        momentum=opt_params.momentum,
        weight_decay=opt_params.weight_decay,
    )
    trainer = TrainerV2(
        device=torch.device(device),
        optimizer=optimizer,
        max_detection_thresholds=train_cfg.max_detection_thresholds,
    )

    # Training loop
    num_epochs = num_epochs or train_cfg.num_epochs
    git_revision = get_git_revision_hash()

    logger.info(f"Setting mlflow.experiment_name={train_cfg.experiment_name}")
    experiment = mlflow.set_experiment(train_cfg.experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        log_params_v1(
            device=device,
            git_revision=git_revision,
            train_cfg_path=train_cfg_path,
            train_data_path=train_data_path,
            valid_data_path=valid_data_path,
            num_epochs=num_epochs,
            model_type=type(model),
            opt_params=opt_params,
            dl_params=dl_params,
        )

        for epoch in range(num_epochs):
            # Train loop:
            trainer.train_epoch(epoch, model, train_data_loader)
            torch.cuda.empty_cache()
            gc.collect()
            # validation loop:
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
