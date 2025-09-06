#!/usr/bin/env python
import gc
import io
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import tqdm
import typer
import yaml
from loguru import logger
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.detection import mean_average_precision

from cow_detect.datasets.std import AnnotatedImagesDataset
from cow_detect.train.teo.train_v1 import TrainCfg, get_model
from cow_detect.train.train_utils import save_model_and_version
from cow_detect.utils.data import custom_collate_dicts, make_jsonifiable_singletons, zip_dict
from cow_detect.utils.metrics import mean_iou
from cow_detect.utils.mlflow_utils import log_mapr_metrics, log_params_v1
from cow_detect.utils.pytorch import detach_dicts, dict_to_device
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
        train_cfg: BaseModel,
        key_metric: str,  # used for deciding whether to
        max_detection_thresholds: list[int],  # used to measure mAR
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        save_path: Path | None,
    ) -> None:
        """Put together resoruces and params used in training.

        :param device: What accelerator device to train on
        :param optimizer: Optimizer used in trainin loop
        :param key_metric: what metric is used to decide whether to save a model,
                e.g "mar_10"
        :param max_detection_thresholds: Used to compute "mar_{mdt}" metrics,
                mdt will range over these values
        :param save_path: if not None model will be saved to this path
        :param train_data_loader: DataLoader for training data
        :param valid_data_loader: DataLoader for validation data
        """
        self.device = device
        self.optimizer = optimizer
        self.key_metric = key_metric
        self.train_cfg = train_cfg
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.max_detection_thresholds = max_detection_thresholds
        self.save_path = save_path

    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
    ) -> None:
        """Run loop over train data for one epoch."""
        train_losses = []  # one per batch
        all_targets: list[dict] = []
        all_predictions: list[dict] = []

        n_batches = get_num_batches(self.train_data_loader)
        pbar = tqdm.tqdm(self.train_data_loader, total=n_batches)
        for batch in pbar:
            # pbar.set_description("Epoch 0: Training: avg.loss=..... avg.mean.iou=.....")
            model.train()
            images = [image.to(self.device) for image in batch["image_pt"]]
            targets: list[dict] = zip_dict(batch)  # type: ignore[arg-type]
            targets = [dict_to_device(tgt, self.device) for tgt in targets]
            all_targets.extend(targets)

            loss_dict = model(images, targets)
            total_loss: torch.Tensor = sum(loss for loss in loss_dict.values())

            # Make predictions in forward mode so that we can calculate IOU and mAP/R metrics
            with torch.no_grad():
                model.eval()
                predictions = detach_dicts(model(images))
                all_predictions.extend(predictions)
                model.train()

            running_train_iou, _ = mean_iou(all_predictions, all_targets)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            train_losses.append(total_loss.detach().item())

            pbar.set_description(
                f"Epoch {epoch}: training: mean.loss={np.mean(train_losses):.4f}, "
                f"mean.train.iou={running_train_iou:.4f}"
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
            negative_to_nan=True,
        )

        mean_train_loss = np.mean(train_losses)
        train_iou, _ = mean_iou(all_predictions, all_targets)
        mlflow.log_metric("mean_train_loss", float(mean_train_loss), step=epoch)
        mlflow.log_metric("mean_train_iou", float(train_iou), step=epoch)
        logged_maprs = log_mapr_metrics(
            mapr_metrics,
            prefix="train",
            step=epoch,
            max_detect_thresholds=self.max_detection_thresholds,
        )
        logger.info(
            f"Epoch {epoch} (train): {mean_train_loss=:.4}, {train_iou=:.4f}, {logged_maprs=}"
        )

    def validate_epoch(
        self,
        epoch: int,
        model: nn.Module,
    ) -> dict[str, float]:
        """Run loop over validation data after an epoch.

        Return a dictionary with validation metrics.
        """
        model.eval()
        metrics: dict[str, float] = {}  # to be returned

        # list of targets and predictions to use torchmetrics api below
        # each of these lists should have as many elements as images in validation_dataset
        all_targets: list[dict[str, torch.Tensor]] = []
        all_predictions: list[dict[str, torch.Tensor]] = []

        n_batches = get_num_batches(self.valid_data_loader)
        with torch.no_grad():
            pbar = tqdm.tqdm(self.valid_data_loader, total=n_batches)
            pbar.set_description("Epoch 0: VALIDATION: mean.iou=....., n_iou_preds=...")
            for batch in pbar:
                images = [image.to(self.device) for image in batch["image_pt"]]
                targets: list[dict] = zip_dict(batch)  # type: ignore[arg-type]
                targets = [dict_to_device(tgt, self.device) for tgt in targets]
                all_targets.extend(targets)

                # PREDICTION happens here:
                predictions: list[dict] = detach_dicts(model(images, targets))
                all_predictions.extend(predictions)
                t0 = time.perf_counter()
                running_mean_iou, iou_preds = mean_iou(all_predictions, all_targets)
                elapsed = time.perf_counter() - t0
                pbar.set_description(
                    f"Epoch {epoch}: VALIDATION:  mean.iou={running_mean_iou:.4f}, "
                    f"n_iou_preds={iou_preds}, elapsed={elapsed:.4f}"
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
            negative_to_nan=True,
        )

        mean_valid_iou, num_iou_preds = mean_iou(all_predictions, all_targets)

        mlflow.log_metric("mean_valid_iou", mean_valid_iou, step=epoch)
        logged_mapr = log_mapr_metrics(
            mapr_metrics,
            prefix="valid",
            step=epoch,
            max_detect_thresholds=self.max_detection_thresholds,
        )
        logger.info(
            f"Epoch {epoch}: VALIDATION : {mean_valid_iou=:.4}, {num_iou_preds=}, "
            f"{logged_mapr=}"
        )

        metrics["mean_valid_iou"] = mean_valid_iou
        metrics["num_iou_preds"] = num_iou_preds
        metrics |= mapr_metrics

        return metrics

    def run_experiment(
        self,
        max_epochs: int,
        model: nn.Module,
    ) -> None:
        """Run a number of epochs of train an validation."""
        git_revision = get_git_revision_hash()

        best_key_metric_value = -float("inf")
        for epoch in range(max_epochs):
            # Train loop:
            self.train_epoch(epoch, model)
            torch.cuda.empty_cache()
            gc.collect()

            # validation loop:
            metrics = self.validate_epoch(epoch, model)
            key_metric_value = metrics[self.key_metric]

            torch.cuda.empty_cache()
            gc.collect()
            mlflow.log_metric("last_finished_epoch", epoch, step=epoch)

            # Save the fine-tuned model
            if self.save_path is not None and key_metric_value > best_key_metric_value:
                logger.info(
                    f"Saving model at epoch {epoch}, key_metric={self.key_metric}, "
                    f"value={key_metric_value:.4f}, "
                    f"previous best value= {best_key_metric_value:.4f}"
                )
                best_key_metric_value = key_metric_value
                save_model_and_version(
                    model,
                    train_cfg=self.train_cfg,
                    git_revision=git_revision,
                    save_path=self.save_path,
                )


DEFAULT_LABEL_TO_ID: dict[str, int] = {"cow": 1, "cattle": 1, "calf": 1}


@cli.command()
def train_v2_std_cli(
    train_cfg_path: Path = typer.Option(..., "--cfg", help="where to get the config from"),
    train_data_path: Path = typer.Option(..., "--train-data", help="where to get train data from"),
    train_data_fraction: float | None = typer.Option(
        None,
        "--train-data-fraction",
        help="what fraction of the training data to use. "
        "If specified will take precedence over the on in the config. "
        "Otherwise, it must be specified in config ",
    ),
    valid_data_path: Path = typer.Option(..., "--valid-data", help="where to get valid. data from"),
    valid_data_fraction: float | None = typer.Option(
        None,
        "--valid-data-fraction",
        help="what fraction of the validation data to use. "
        "If specified will take precedence over the on in the config. "
        "Otherwise, it must be specified in config",
    ),
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

    if device is not None:
        train_cfg.device = device
        logger.info(f"Overriding train_cfg.device={device} (from CLI option)")
    del device
    assert train_cfg.device is not None

    logger.info(f"Using device: {train_cfg.device}")
    # TODO: make this configurable?
    label_to_id = DEFAULT_LABEL_TO_ID
    logger.info(f"Using label to id: {json.dumps(label_to_id)}")

    if train_data_fraction is not None:
        train_cfg.train_fraction = train_data_fraction
        logger.info(f"Overriding {train_cfg.train_fraction=} (from CLI option)")
    assert (
        train_cfg.train_fraction is not None
    ), "train_data_fraction must be specified in config or as CLI option"
    del train_data_fraction

    if valid_data_fraction is not None:
        train_cfg.valid_fraction = valid_data_fraction
        logger.info(f"Overriding {train_cfg.valid_fraction=}(from CLI option) ")
    assert (
        train_cfg.valid_fraction is not None
    ), "valid_data_fraction must be specified in config or as CLI option"
    del valid_data_fraction

    logger.info(
        f"Train cfg (full, including defaults, and overrides):\n"
        f"{train_cfg.model_dump_json(indent=2)}\n"
    )

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
    model.to(train_cfg.device)

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
        device=torch.device(train_cfg.device),
        train_cfg=train_cfg,
        optimizer=optimizer,
        max_detection_thresholds=train_cfg.max_detection_thresholds,
        key_metric=train_cfg.key_metric,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        save_path=save_path,
    )

    # Training loop
    num_epochs = num_epochs or train_cfg.num_epochs
    git_revision = get_git_revision_hash()

    logger.info(f"Setting mlflow.experiment_name={train_cfg.experiment_name}")
    experiment = mlflow.set_experiment(train_cfg.experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        log_params_v1(
            device=train_cfg.device,
            train_cfg=train_cfg.model_dump(),
            git_revision=git_revision,
            train_cfg_path=train_cfg_path,
            train_data_path=train_data_path,
            train_data_fraction=train_cfg.train_fraction,
            valid_data_path=valid_data_path,
            valid_data_fraction=train_cfg.valid_fraction,
            num_epochs=num_epochs,
            model_type=type(model),
            opt_params=opt_params,
            dl_params=dl_params,
        )

        trainer.run_experiment(max_epochs=num_epochs, model=model)


if __name__ == "__main__":
    cli()
