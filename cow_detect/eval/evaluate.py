#!/usr/bin/env python
import itertools
import json
import time
from importlib import reload
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
import torch
import tqdm
import typer
from loguru import logger
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.detection import mean_average_precision

import cow_detect.datasets.std as ds
from cow_detect.predict.batch import get_prediction_model
from cow_detect.train.types import Prediction, Target
from cow_detect.utils.data import (
    custom_collate_dicts,
    make_jsonifiable,
    make_jsonifiable_singletons,
    zip_dict,
)
from cow_detect.utils.metrics import max_ious_for_preds

reload(ds)

# This can be overriden via CLI argument for models that
# distinguish cow from calf. See doc of cli.eval for details.
# e.g --label_to_id: '{"cow": 1, "calf": 2}'
DEFAULT_LABEL_TO_ID = {"cow": 1, "cattle": 1, "calf": 1}

DEFAULT_SCORE_THRESHOLDS = torch.arange(0, 1.00, 0.02)
# %%

cli = typer.Typer(pretty_exceptions_show_locals=False)


class BatchForEvaluation(TypedDict):
    """Parallel lists of stuff, each with one element per input image."""

    image_pt: list[torch.Tensor]  # each element is an individual image as tensor
    boxes: list[
        torch.Tensor
    ]  # each element has all target/true boxes in one image, i.e. shape [n_boxes, 4]


class OneImageEvaluationResult(BaseModel):
    """Result of evaluating a detection model on one image."""

    image_path: Annotated[Path, Field(desc="Path to the input image image")]
    true_count: Annotated[int, Field(desc="Number ground-truth boxes for this image")]
    counting_err_by_score_thresh: Annotated[
        dict[float, int], Field("See counting_errors_by_score_threshold function for details")
    ]
    prediction: Annotated[
        dict[str, list],
        Field(
            desc="The prediction result for this image including bboxes, class " "labels and scores"
        ),
    ]
    max_ious_for_preds: Annotated[
        list[float], Field("Maximum IoU against all targets for each predicted box")
    ]
    prediction_time_secs: Annotated[float, Field(desc="Elapsed time in seconds for this image")]


class EvaluationResults(BaseModel):
    """Aggregated evaluation results over a whole dataset."""

    by_image_results: list[OneImageEvaluationResult]
    mapr_metrics: Annotated[dict[str, float], Field(desc="The mAP and mAR metrics for this image")]
    total_detections: Annotated[int, Field(desc="Total number of detections for all images")]
    mean_count_err_by_thresh: Annotated[
        dict[float, float], Field(desc="mean of counting error for each score threshold")
    ]
    std_dev_count_err_by_thresh: Annotated[
        dict[float, float], Field(desc="std. dev. of counting error for each score threshold")
    ]
    mean_max_ious: Annotated[float, Field(desc="Mean of the Max IoU over all preds.")]


def counting_errors_by_score_threshold(
    true_count: int, *, scores: torch.Tensor, score_thresholds: torch.Tensor
) -> dict[float, int]:
    """Return counting errors vs. a true count for each score threshold.

    That is, for each score threshold thrsh, count how many scores are above it.
    Call that count(thrsh). Compute the counting error as count(thrsh) - true_count

    Return the dictionary
        { thrsh[i]: count(thrsh[i]) - count | i in range(len(score_thresholds)) }
    """

    def count(thrsh: float) -> int:
        return int((scores >= thrsh).sum().item())

    return {thrsh: count(thrsh) - true_count for thrsh in score_thresholds}


def eval_one_batch(
    batch: BatchForEvaluation, model: nn.Module
) -> tuple[list[OneImageEvaluationResult], list[Prediction], list[Target]]:
    """Run evaluation on one batch of data.

    Return a list of OneImageEvaluationResult, one per each image in the input batch.
    """
    images_batch: list[torch.Tensor] = batch["image_pt"]

    t0 = time.perf_counter()
    # PREDICTION happens here:
    preds_batch: list[Prediction] = model(images_batch)
    pred_elapsed = time.perf_counter() - t0

    assert len(preds_batch) == len(images_batch), f"{len(preds_batch)} != {len(images_batch)}"
    targets_batch: list[Target] = zip_dict(batch)  # type: ignore[arg-type, assignment]
    batch_size = len(preds_batch)

    output_records = []
    for target, prediction in zip(targets_batch, preds_batch, strict=False):
        image_path = target["image_path"]
        true_bboxes = target["boxes"]
        # print("precision:", mapr_metrics["map"])  # , " recall:", metrics["mar"])

        true_count = len(true_bboxes)
        output_record = OneImageEvaluationResult(
            image_path=image_path,
            true_count=true_count,
            counting_err_by_score_thresh=counting_errors_by_score_threshold(
                true_count, scores=prediction["scores"], score_thresholds=DEFAULT_SCORE_THRESHOLDS
            ),
            max_ious_for_preds=max_ious_for_preds(prediction, target),
            prediction=make_jsonifiable(prediction),  # type: ignore[arg-type]
            prediction_time_secs=pred_elapsed / batch_size,
        )

        output_records.append(output_record)

    return output_records, preds_batch, targets_batch


def eval_on_whole_dataset(
    predict_ds: ds.AnnotatedImagesDataset, model: nn.Module, batch_size: int
) -> EvaluationResults:
    # Run evaluation on each image:
    predict_data_loader = DataLoader(
        predict_ds, batch_size=batch_size, collate_fn=custom_collate_dicts, num_workers=0
    )

    all_eval_results: list[OneImageEvaluationResult] = []
    all_preds: list[Prediction] = []
    all_targets: list[Target] = []
    pbar = tqdm.tqdm(predict_data_loader, total=predict_ds.num_batches(batch_size))
    for batch in pbar:
        eval_results, preds_batch, targets_batch = eval_one_batch(batch, model)
        all_eval_results.extend(eval_results)
        all_preds.extend(preds_batch)
        all_targets.extend(targets_batch)

    mapr_metrics = mean_average_precision(
        preds=all_preds,  # type: ignore[arg-type]
        target=all_targets,  # type: ignore[arg-type]
        box_format="xyxy",
        iou_type="bbox",
    )
    assert isinstance(mapr_metrics, dict)

    mean_count_err_by_thresh, std_dev_count_err_by_thresh = compute_counting_error_stats_by_thresh(
        [r.counting_err_by_score_thresh for r in all_eval_results]
    )

    all_max_ious_for_preds = list(
        itertools.chain.from_iterable(r.max_ious_for_preds for r in all_eval_results)
    )
    # print(list(all_max_ious_for_preds))
    mean_max_ious = float(
        torch.mean(torch.tensor(all_max_ious_for_preds, dtype=torch.float32)).item()
    )
    num_max_ious = len(all_max_ious_for_preds)
    total_detections = sum(len(r.prediction["boxes"]) for r in all_eval_results)
    assert num_max_ious == total_detections, f"{num_max_ious=} != {total_detections=}"

    return EvaluationResults(
        by_image_results=all_eval_results,
        mapr_metrics=make_jsonifiable_singletons(mapr_metrics, negative_to_nan=True),
        mean_count_err_by_thresh=mean_count_err_by_thresh,
        std_dev_count_err_by_thresh=std_dev_count_err_by_thresh,
        mean_max_ious=mean_max_ious,
        total_detections=total_detections,
    )


def compute_counting_error_stats_by_thresh(
    counting_errs_by_thresh_all_imgs: list[dict[float, int]],
) -> tuple[dict[float, float], dict[float, float]]:
    """Compute mean and std.deviation of counting errors over all each, for each threshhold.

    Input: counting_errs_by_thresh results for each image
    Output: (dict of mean by thresh, dict std-dev by thresh)
    """
    df = pd.DataFrame(counting_errs_by_thresh_all_imgs)
    threshs = df.columns.values

    means_by_thresh: dict[float, float] = {}
    stds_by_thresh: dict[float, float] = {}

    for thresh in threshs:
        means_by_thresh[thresh] = df[thresh].mean()
        stds_by_thresh[thresh] = df[thresh].std()

    return means_by_thresh, stds_by_thresh


@cli.command()
def eval_from_std_dataset(
    model_weights: Path = typer.Argument(help="Path to model weights"),
    std_data_path: Path = typer.Argument(
        help="standardized data set containing image paths and ground truth annotations "
        "against which to run evaluation"
    ),
    model_type: str = "teo",
    output_path: Path = typer.Option(
        ..., "--out", help="path of output directory, several files will be written into it"
    ),
    batch_size: int = 2,  # kept low to avoid memory overflow on small gpu's
    label_to_id_json: str | None = typer.Option(
        None,
        "--label-to-id",
        help='e.g. {"cow": 1, "calf": 2}, '
        "for a model that can distinguish both classes."
        f"If left unspecified, we will use {json.dumps(DEFAULT_LABEL_TO_ID)}.",
    ),
    limit: None | int = None,
) -> None:
    """Run evaluation of a model on a given dataset."""
    label_to_id = DEFAULT_LABEL_TO_ID if label_to_id_json is None else json.loads(label_to_id_json)
    predict_ds = ds.AnnotatedImagesDataset(
        std_data_path, limit=limit, label_to_id=label_to_id, name=std_data_path.name
    )
    model = get_prediction_model(model_weights, model_type=model_type)
    model.eval()

    logger.info(f"Evaluation starts... Results will be saved to {output_path!s}")
    eval_results = eval_on_whole_dataset(predict_ds, model, batch_size)

    output_path.mkdir(parents=True, exist_ok=True)
    # Write evaluation results, one line per image
    by_image_path = output_path / "eval_results_by_image.jsonl"
    with torch.no_grad(), by_image_path.open("wt", encoding="utf-8", newline="\n") as file_out:
        for record in eval_results.by_image_results:
            file_out.write(record.model_dump_json() + "\n")

    summary_path = output_path / "summary.json"
    eval_results.by_image_results = []
    logger.info(f"Summary results will be saved to {summary_path!s}")
    summary_path.write_text(eval_results.model_dump_json(indent=4))


def _interactive_testing() -> None:
    # %%
    eval_from_std_dataset(
        # std_data_path=Path("data/standardized/sky.dataset2.jsonl"),
        std_data_path=Path("data/standardized/icaerus.derval.jsonl"),
        model_weights=Path("data/training/teo/v1/faster-rcnn/model.pth"),
        model_type="teo/v1",
        batch_size=4,
        output_path=Path("data/evaluation/faster-rcnn-v1.json"),
        limit=None,
    )
    # %%
    example_output = """
        {"image_path": "data/sky/Dataset2/img/DJI_0036.JPG",
        "total_detections": 3, "counts_by_label": {"1": 3},
        "avg_score": 0.9464678168296814, "prediction": {
        "boxes": [[587.2669677734375, 1636.1265869140625, 686.401123046875, 1731.468017578125],
                  [761.794189453125, 2817.29345703125, 854.7379150390625, 2952.552490234375],
                  [315.4936828613281, 2670.287841796875, 419.01641845703125, 2742.505615234375]],
                  "labels": [1, 1, 1],
        "scores": [0.996434211730957, 0.9948186278343201, 0.8481504917144775]}, "error": null}
    """
    obj = json.loads(example_output)
    print(json.dumps(obj, indent=4))

    # %%


if __name__ == "__main__":
    cli()
