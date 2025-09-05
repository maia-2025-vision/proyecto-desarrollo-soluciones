import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torchvision  # type: ignore[import-untyped]
import tqdm
from loguru import logger
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cow_detect.utils.data import make_jsonifiable


class ImagesPredictDataset(Dataset):
    """Dataset class used for prediction/inference."""

    def __init__(
        self,
        root_dir: Path,
        extensions: tuple[str, ...] = (".jpg", ".png", ".jpeg"),
        force_resize: bool = True,
        limit: None | int = None,
    ) -> None:
        """A torch dataset that provides tuples (images: tensor, path: Path).

        :param root_dir: Directory to find images (recursively)
        :param extensions: Valid image extensions
        :param force_resize: Whether to force resize of images to size of first image read
        :param limit: If not None, limit to this maximum number of images to return
           (for quicker testing)
        """
        self.root_dir = root_dir
        # self.transforms = transforms

        all_paths_raw = list(self.root_dir.rglob("*.*"))
        logger.info(f"Found {len(all_paths_raw)} files under {self.root_dir!s}")
        self.image_paths = [
            fp for fp in all_paths_raw if fp.is_file() and fp.suffix.lower() in extensions
        ]

        logger.info(
            f"{len(self.image_paths)} images found under {self.root_dir} (and its subfolders), "
            f"extensions included: {extensions}"
        )

        if limit is not None:
            logger.info(f"Limiting to {limit} images")
            self.image_paths = self.image_paths[:limit]

        self.img_to_tensor = torchvision.transforms.ToTensor()
        self.target_size: tuple[int, int] | None = None
        self.force_resize: bool = force_resize

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.image_paths)

    def num_batches(self, batch_size: int) -> int:
        """Get the number of batches per epoch."""
        return int(math.ceil(len(self.image_paths) / batch_size))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Path, str | None]:
        """Get the i-th tuple (image, path, error).

        image will be a torch tensor of shape (3, W, H)
        Error will be not None if image is not found or is unreadable by PIL.
        """
        img_path = self.image_paths[idx]
        # This simple rule works for SKY but it doesnt work for ICAERUS

        error: str | None = None  # only set if we fail to load the imge
        try:
            image = Image.open(img_path).convert("RGB")
            if self.target_size is None:
                self.target_size = image.size
                logger.info(
                    f"After successfully loaded image, set target_size to: {self.target_size}"
                )
            else:
                if image.size != self.target_size:
                    if self.force_resize:
                        logger.info(f"Resizing image from {image.size} to {self.target_size}")
                        image = image.resize(self.target_size)
                    else:
                        logger.warning(
                            f"Image loaded from {img_path!s} has size {image.size} "
                            f"which differs from {self.target_size}. "
                            f"If you are processing batches of size > 1, "
                            f"this will cause an error downstream."
                        )

        except OSError as err:
            logger.warning(
                f"Could not open image file: {img_path!s}, error: {err!s}, instead, will return "
                f"fully default black image of size {self.target_size}"
            )
            assert self.target_size is not None
            image = Image.new("RGB", self.target_size, (0, 0, 0))
            # original_size = None
            error = str(err)

        image_tensor = self.img_to_tensor(image)
        del image

        return image_tensor, img_path, error


def get_prediction_model(weights_path: Path, model_type: str = "teo/v1") -> nn.Module:
    """Restore and return a prediction model from a weights file."""
    if model_type == "teo/v1":
        from cow_detect.train.teo.train_v1 import get_model

        model = get_model(num_classes=2)
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        assert isinstance(model, nn.Module)
        return model
    else:
        raise NotImplementedError(f"model_type={model_type} not implemented yet")


def custom_collate_fun(batch: list[tuple]) -> tuple[list, ...]:
    """Given a list of tuples, build a tuple of lists.

    E.g.
    custom_collate_fun([(1, "a", tensor1), (3, "b", tensor2)])
    == ([1, 2], ["a", "b"], [tensor1, tensor2])
    """
    return tuple(zip(*batch, strict=True))  # type: ignore[arg-type]  #  mypy false positive, this works


def predict_one_batch(
    batch: tuple[torch.Tensor, list[Path], list[str | None]], model: nn.Module
) -> list[dict]:
    """Run prediction on a single batch of images.

    Return a list of records with the same size as the batch,
    each containing predictions for one image, plus some stats.
    """
    images_tensor, image_paths, errors = batch
    # images_tensor, image_paths, errors = (
    #     batch["images_tensor"], batch["image_paths"], batch["errors"]
    # )

    t0 = time.perf_counter()
    predictions: list[dict] = model(images_tensor)
    pred_elapsed = time.perf_counter() - t0

    output_tups = zip(image_paths, predictions, errors, strict=True)
    n_records = len(predictions)

    output_records = []
    for tup in output_tups:
        image_path, prediction, error = tup
        total_detections = len(prediction["boxes"])
        avg_score = prediction["scores"].mean().item()
        counts_by_label = dict(Counter(prediction["labels"].tolist()))

        prediction_json = make_jsonifiable(prediction)
        output_record = {
            "image_path": str(image_path),
            "total_detections": total_detections,
            "counts_by_label": counts_by_label,
            "avg_score": avg_score,
            "prediction": prediction_json,
            "error": error,
            "prediction_time_secs": pred_elapsed / n_records,
        }

        output_records.append(output_record)

    return output_records


def predict_from_path_by_batches(
    images_path: Path,
    model_weights: Path,
    output_path: Path,
    model_type: str = "teo/v1",
    batch_size: int = 1,
    limit: None | int = None,
) -> None:
    """Run model inference on the given model on an image path.

    This is meant for pure inference, images are NOT assumed
    to have ground truth annotations.
    """
    predict_ds = ImagesPredictDataset(images_path, limit=limit)
    model = get_prediction_model(model_weights, model_type=model_type)
    model.eval()

    predict_data_loader = DataLoader(
        predict_ds, batch_size=batch_size, collate_fn=custom_collate_fun
    )

    logger.info(f"Prediction starts... Results will be saved to {output_path!s}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wt", encoding="utf-8", newline="\n") as file_out:
        with torch.no_grad():
            pbar = tqdm.tqdm(predict_data_loader, total=predict_ds.num_batches(batch_size))
            for batch in pbar:
                output_records = predict_one_batch(batch, model)
                for record in output_records:
                    file_out.write(json.dumps(record) + "\n")


def _interactive_testing() -> None:
    # %%
    predict_from_path_by_batches(
        images_path=Path("data/sky/Dataset1"),
        model_weights=Path("data/training/teo/v1/faster-rcnn/model.pth"),
        model_type="teo/v1",
        batch_size=16,
        output_path=Path("data/prediction/teo/v1/sky-ds1-faster-rcnn.jsonl"),
        limit=None,
    )
    # %%
    example_output = """
        {"image_path": "data/sky/Dataset2/img/DJI_0036.JPG",
        "total_detections": 3, "counts_by_label": {"1": 3},
        "avg_score": 0.9464678168296814, "prediction": {
        "boxes": [[587.2669, 1636.12, 686.40112, 1731.46801],
                  [761, 2817.293125, 854.7379150390625, 2952.552490234375],
                  [315.49368, 2670.287841, 419.01643125, 2742.505615234375]], "labels": [1, 1, 1],
        "scores": [0.996434211730957, 0.9948186278343201, 0.8481504917144775]}, "error": null}
    """
    obj = json.loads(example_output)
    print(json.dumps(obj, indent=4))

    # %%


# %%
