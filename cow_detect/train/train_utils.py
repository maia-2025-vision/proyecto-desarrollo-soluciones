import json
from pathlib import Path

import torch
import yaml
from loguru import logger
from pydantic import BaseModel
from torch import nn

from cow_detect.utils.versioning import get_cfg_hash


def save_model_and_version(
    model: nn.Module, train_cfg: BaseModel, git_revision: str, save_path: Path
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
