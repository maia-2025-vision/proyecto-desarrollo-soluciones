import torch
from torchvision.ops import box_iou


def calculate_iou(predictions: list[dict], targets: list[dict]) -> float:
    """Calculate IoU (intersection over union) between predictions and targets."""
    total_iou = 0
    num_iou = 0

    for i in range(len(predictions)):
        pred_boxes = predictions[i]["boxes"]
        target_boxes = targets[i]["boxes"]

        if pred_boxes.shape[0] > 0 and target_boxes.shape[0] > 0:
            iou_matrix = box_iou(pred_boxes, target_boxes)
            max_iou_per_pred = torch.max(iou_matrix, dim=1)[0]
            total_iou += torch.sum(max_iou_per_pred)
            num_iou += pred_boxes.shape[0]

    return total_iou / num_iou if num_iou > 0 else 0
