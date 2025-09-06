import torch
from torchvision.ops import box_iou  # type: ignore[import-untyped]


def mean_iou(predictions: list[dict], targets: list[dict]) -> tuple[float, int]:
    """Calculates mean "best" IoU (intersection over union) between predictions and targets.

    Compute the best IoU for each predicted box (each element of predictions may contain many boxes)
    Finally divide the total sum of all IoUs by the total number of predicted boxes.
    If total number of predicted boxes is 0, return 0.
    """
    total_iou: float = 0.0
    num_iou: int = 0

    for i in range(len(predictions)):
        pred_boxes = predictions[i]["boxes"]
        target_boxes = targets[i]["boxes"]

        if pred_boxes.shape[0] > 0 and target_boxes.shape[0] > 0:
            iou_matrix = box_iou(pred_boxes, target_boxes)
            max_iou_per_pred = torch.max(iou_matrix, dim=1)[0]
            total_iou += float(torch.sum(max_iou_per_pred).detach().item())
            num_iou += pred_boxes.shape[0]

    return (total_iou / num_iou if num_iou > 0 else 0), num_iou
