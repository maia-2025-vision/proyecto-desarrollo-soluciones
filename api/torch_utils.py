from pathlib import Path

import torch
import torchvision
from loguru import logger
from torch import nn
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)


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


def get_prediction_model(weights_path: Path, model_type: str = "teo") -> nn.Module:
    """Restore and return a prediction model from a weights file."""
    if model_type == "teo":
        model = get_model(num_classes=2)
        logger.info(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        assert isinstance(model, nn.Module)
        return model
    else:
        raise NotImplementedError(f"model_type=`{model_type}` not implemented yet")
