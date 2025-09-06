from hashlib import md5
from pathlib import Path

import mlflow

from cow_detect.utils.config import DataLoaderParams, OptimizerParams


def log_params_v1(
    *,
    device: str,
    git_revision: str,
    train_cfg_path: Path,
    train_data_path: Path,
    train_data_fraction: float,
    valid_data_path: Path,
    valid_data_fraction: float,
    num_epochs: int,
    model_type: type,
    opt_params: OptimizerParams,
    dl_params: DataLoaderParams,
) -> None:
    cfg_md5 = md5(train_cfg_path.read_bytes()).hexdigest()

    mlflow.log_param("data_set", str(train_data_path))
    mlflow.log_param("train_data_fraction", train_data_fraction)
    mlflow.log_param("valid_data_set", str(valid_data_path))
    mlflow.log_param("valid_data_fraction", valid_data_fraction)
    mlflow.log_param("git_revision_12", git_revision[:12])
    mlflow.log_param("cfg_md5", cfg_md5)
    mlflow.log_param("cfg_path", str(train_cfg_path))
    mlflow.log_param("cfg_full", str(train_cfg_path.read_text()))
    mlflow.log_param("model_class", model_type.__name__)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("optimizer_class", opt_params.optimizer_class)
    mlflow.log_param("lr", opt_params.learning_rate)
    mlflow.log_param("momentum", opt_params.momentum)
    mlflow.log_param("weight_decay", opt_params.weight_decay)
    mlflow.log_param("batch_size", dl_params.batch_size)
    mlflow.log_param("num_workers", dl_params.num_workers)
    mlflow.log_param("device", str(device))


def log_mapr_metrics(
    mapr_metrics: dict[str, float],
    prefix: str,
    step: int,
    max_detect_thresholds: list[int],
    decimals: int = 4
) -> dict[str, float]:
    logged = {}
    for metric in ["map", "map_50", "map_75", "map_medium"]:
        value = round(mapr_metrics[metric], decimals)
        mlflow.log_metric(f"{prefix}_{metric}", value, step=step)
        logged[metric] = value

    for mdt in max_detect_thresholds:
        key = f"mar_{mdt}"
        value = round(mapr_metrics[key], decimals)
        mlflow.log_metric(f"{prefix}_{metric}", value, step=step)
        logged[key] = value

    return logged
