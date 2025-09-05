import json
import time
from importlib import reload
from pathlib import Path

import torch
import tqdm
import typer
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.detection import mean_average_precision

import cow_detect.datasets.std as ds
from cow_detect.predict.batch import get_prediction_model
from cow_detect.utils.data import custom_collate_dicts, make_jsonifiable, zip_dict

reload(ds)

# This can be overriden via CLI argument for models that
# distinguish cow from calf. See doc of cli.eval for details.
# e.g --label_to_id: '{"cow": 1, "calf": 2}'
DEFAULT_LABEL_TO_ID = {"cow": 1, "cattle": 1, "calf": 1}

cli = typer.Typer()


def eval_one_batch(
    batch: dict[str, list[torch.Tensor | str | Path]], model: nn.Module
) -> list[dict]:
    """Run evaluation on one batch of data."""
    images_batch: list[torch.Tensor] = batch["image_pt"]

    t0 = time.perf_counter()
    preds_batch: list[dict] = model(images_batch)
    pred_elapsed = time.perf_counter() - t0

    assert len(preds_batch) == len(images_batch), f"{len(preds_batch)} != {len(images_batch)}"
    targets: list[dict] = zip_dict(batch)
    batch_size = len(preds_batch)

    output_records = []
    for target, prediction in zip(targets, preds_batch, strict=False):
        image_path = target["image_path"]
        true_bboxes = target["boxes"]
        total_detections = len(prediction["boxes"])

        mapr_metrics = mean_average_precision(
            preds=[prediction],
            target=[{"boxes": true_bboxes, "labels": target["labels"]}],
            box_format="xyxy",
            iou_type="bbox",
        )
        # print("precision:", mapr_metrics["map"])  # , " recall:", metrics["mar"])

        output_record = {
            "image_path": str(image_path),
            "total_detections": total_detections,
            "true_count": len(true_bboxes),
            # "counts_by_label": counts_by_label,
            "prediction": make_jsonifiable(prediction),
            "mapr_metrics": make_jsonifiable(mapr_metrics),
            "prediction_time_secs": pred_elapsed / batch_size,
        }

        output_records.append(output_record)

    return output_records


@cli.command()
def eval_from_std_dataset(
    std_data_path: Path = typer.Argument(
        help="standardized data set containing image paths and annotations "
        "on which we will run evaluation"
    ),
    model_type: str = "teo/v1",
    model_weights: Path = typer.Option(..., help="Path to model weights."),
    output_path: Path = typer.Option(..., help="path of output file, should have .jsonl extension"),
    batch_size: int = 2,
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
    predict_ds = ds.AnnotatedImagesDataset(std_data_path, limit=limit, label_to_id=label_to_id)
    model = get_prediction_model(model_weights, model_type=model_type)
    model.eval()

    predict_data_loader = DataLoader(
        predict_ds, batch_size=batch_size, collate_fn=custom_collate_dicts
    )

    logger.info(f"Evaluation starts... Results will be saved to {output_path!s}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), output_path.open("wt", encoding="utf-8", newline="\n") as file_out:
        pbar = tqdm.tqdm(predict_data_loader, total=predict_ds.num_batches(batch_size))
        for batch in pbar:
            output_records = eval_one_batch(batch, model)
            for record in output_records:
                file_out.write(json.dumps(record) + "\n")


# %%
def _interactive_testing():
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


# %%
