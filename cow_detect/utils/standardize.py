#!/usr/bin/env python
"""Script to standardize training/evaluation data into a flat format."""

from collections import Counter
from pathlib import Path
from typing import Literal

import PIL
import tqdm
import typer
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from cow_detect.utils.annotations import parse_json_annotations_file, parse_yolo_annotation_file

cli = typer.Typer(pretty_exceptions_show_locals=False)


class ImageInfo(BaseModel):
    """Info about location and size of an image."""

    path: Path = Field(description="Path to the image.")
    n_bytes: int = Field(description="Image size in bytes")
    status: Literal["ok", "not-found", "unreadable"]
    width: int | None = Field(description="Image width in pixels", nullable=True)
    height: int | None = Field(description="Image height in pixels", nullable=True)
    error: str = Field(
        "", description="Any error encountered while trying to load the image", nullable=True
    )


class Annot(BaseModel):
    """A single annotation i.e. label + bbox coordinates, and possibly a dictionary o tags."""

    label: str = Field(
        description="Class label of the object in the bbox, e.g. 'cow', 'cattle', 'calf' "
    )
    coords: list[float] = Field(
        description="Bbox coordinates in Pascal VOC format [x_min, y_min, x_max, y_max],"
        "where 0 <= x_min < x_max <= image.width and 0 <= y_min < y_max <= image.height"
        "could be ints of floats"
    )
    tags: dict[str, object] = Field(description="an arbitrary dictionary of tags keyed by name")


class AnnotationsFileInfo(BaseModel):
    """Info about an annotations file, including extracted annotations."""

    path: Path
    status: Literal["ok", "not-found", "no-width-height", "unreadable", "unparseable"]
    byte_size: int | None = Field(description="Annotation file size in bytes", nullable=True)
    n_lines: int | None = Field(description="number of lines", nullable=True)
    n_annots: int | None = Field(None, description="number of annotations", nullable=True)
    annots: list[Annot] | None = Field(
        None, description="Bounding boxes and their labels", nullable=True
    )
    error: str = Field(
        "", description="Possible error encountered while trying to generate annotations"
    )


class AnnotatedImageRecord(BaseModel):
    """Record bringing together image file info and annotations."""

    dataset_kind: str = Field(description="Dataset kind of the image, e.g. sky, icaerus")
    images_root_path: Path = Field(description="Root that contains all images")
    image: ImageInfo
    annotations: AnnotationsFileInfo


class SkyTraits:
    """Methods for reading images and annotations from Sky dataset."""

    def __init__(self, images_root_path: Path):
        self.dataset_kind = "sky"
        self.images_root_path = images_root_path

    def annots_path(self, image_path: Path) -> Path:
        """Calculate annotation file path from image path."""
        image_fname = image_path.name
        return image_path.parent.parent / "ann" / (image_fname + ".json")

    def read_annots(self, annots_path: Path) -> AnnotationsFileInfo:
        """Read annotations from a path."""
        # TODO: maybe add tag parsing here?
        bboxes_coords, labels = parse_json_annotations_file(annots_path)

        byte_size = annots_path.stat().st_size
        n_lines = len(annots_path.read_text().splitlines())

        annots = []
        for coords, label in zip(bboxes_coords, labels, strict=False):
            annots.append(Annot(label=label, coords=coords, tags={}))

        return AnnotationsFileInfo(
            path=annots_path,
            status="ok",
            byte_size=byte_size,
            n_lines=n_lines,
            n_annots=len(annots),
            annots=annots,
        )

    def process_one(self, image_path: Path) -> AnnotatedImageRecord:
        """Process one image.

        Try to find annotations for it and parse them.
        """
        img_info = get_image_info(image_path)
        annots_path = self.annots_path(image_path)
        annot_info = self.read_annots(annots_path)

        return AnnotatedImageRecord(
            dataset_kind=self.dataset_kind,
            images_root_path=self.images_root_path,
            image=img_info,
            annotations=annot_info,
        )


class IcaerusTraits:
    """Methods to read image and annotations from Icaerus datasets."""

    def __init__(self, images_root_path: Path):
        self.dataset_kind = "icaerus"
        self.images_root_path = images_root_path

    def _annots_path(self, image_path: Path) -> list[Path]:
        # image_fname = image_path.name
        pparent = image_path.parent.parent
        assert pparent.name == "JPGImages", f"{pparent=}"

        rel_path_parts = image_path.relative_to(pparent).parts
        assert len(rel_path_parts) == 2, f"{rel_path_parts=}"

        annot_fname = image_path.name.rsplit(".", 1)[0] + ".txt"
        annots_path1 = (
            pparent.parent / "YOLO_1.1" / rel_path_parts[0] / "obj_train_data" / annot_fname
        )
        annots_path2 = (
            pparent.parent / "YOLO1.1" / rel_path_parts[0] / "obj_train_data" / annot_fname
        )
        candidates = [annots_path1, annots_path2]
        existing = [p for p in candidates if p.exists()]

        if len(existing) > 0:
            return existing[0]
        else:
            raise FileNotFoundError(f"None of these exists: {[str(p) for p in candidates]}")

    def _read_annots(self, annots_path: Path, *, img_width: int, img_height) -> AnnotationsFileInfo:
        bboxes_coords, label_ids = parse_yolo_annotation_file(
            annots_path, img_width=img_width, img_height=img_height
        )
        obj_names_path = annots_path.parent / "../obj.names"
        id_2_name: dict[int, str] = dict(enumerate(obj_names_path.read_text().splitlines()))
        label_strs: list[str] = [id_2_name[id_] for id_ in label_ids]

        byte_size = annots_path.stat().st_size
        n_lines = len(annots_path.read_text().splitlines())

        annots = []
        for coords, label in zip(bboxes_coords, label_strs, strict=False):
            annots.append(Annot(label=label, coords=coords, tags={}))

        return AnnotationsFileInfo(
            path=annots_path,
            status="ok",
            byte_size=byte_size,
            n_lines=n_lines,
            n_annots=len(annots),
            annots=annots,
        )

    def process_one(self, image_path: Path) -> AnnotatedImageRecord:
        """Process one image path and its associated annotations file."""
        img_info = get_image_info(image_path)

        annot_info: AnnotationsFileInfo | None = None
        annots_path: Path | None = None

        try:
            annots_path = self._annots_path(image_path)
        except FileNotFoundError as err:
            annot_info = AnnotationsFileInfo(
                path=Path("notfound.txt"),
                status="not-found",
                error=f"FileNotFoundError: {err}",
                byte_size=None,
                n_lines=None,
                annots=None,
            )

        if annots_path is not None:
            width, height = img_info.width, img_info.height
            if width is not None and height is not None:
                annot_info = self._read_annots(annots_path, img_width=width, img_height=height)
            else:
                annot_info = AnnotationsFileInfo(
                    path=annots_path,
                    byte_size=annots_path.stat().st_size,
                    n_lines=None,
                    annots=None,
                    status="no-width-height",
                    error="No width/height provided, error loading image?",
                )

        return AnnotatedImageRecord(
            dataset_kind=self.dataset_kind,
            images_root_path=self.images_root_path,
            image=img_info,
            annotations=annot_info,
        )


def get_image_info(image_path: Path) -> ImageInfo:
    """Extract image info from an image at path.

    If image is not found or it is not readable a record with status='unreadable'
    and error message, and width, height set to None will be returned
    """
    try:
        img = Image.open(image_path).convert("RGB")
        n_bytes = image_path.stat().st_size
        width, height = img.size

        return ImageInfo(
            path=image_path,
            n_bytes=n_bytes,
            status="ok",
            width=width,
            height=height,
            error="",
        )
    except (OSError, PIL.UnidentifiedImageError) as e:
        n_bytes = image_path.stat().st_size
        return ImageInfo(
            path=image_path,
            n_bytes=n_bytes,
            status="unreadable",
            width=None,
            height=None,
            error=f"{type(e).__name__}: {e}",
        )


@cli.command()
def main(
    images_root_dir: Path,
    kind: str = typer.Option(
        ...,
        help="what kind of dataset is this, e.g. 'sky', 'icaerus', "
        "other ones to be implemented in the future?",
    ),
    output_path: Path = typer.Option(
        ..., "-o", "--out", help="output will go here, should have .jsonl extension!"
    ),
):
    """High level standardization logic."""
    assert output_path.suffix == ".jsonl", f"{output_path.suffix=} is not .jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assert images_root_dir.is_dir(), f"{images_root_dir=} is not a directory"

    if kind == "sky":
        traits = SkyTraits(images_root_dir)
    elif kind == "icaerus":
        traits = IcaerusTraits(images_root_dir)
    else:
        raise ValueError(f"Unknown kind: `{kind}`")

    image_paths = sorted(images_root_dir.rglob("*.JPG"))

    logger.info(f"{len(image_paths)} images found.")

    image_stats: Counter[str] = Counter()
    annot_stats: Counter[str] = Counter()

    records_written = 0
    with output_path.open("wt") as f_out:
        for image_path in tqdm.tqdm(image_paths):
            record_out = traits.process_one(image_path)
            image_stats[record_out.image.status.lower()] += 1
            annot_stats[record_out.annotations.status.lower()] += 1
            print(record_out.model_dump_json(), file=f_out)
            records_written += 1

    logger.info(f"{image_stats=}")
    logger.info(f"{annot_stats=}")
    logger.info(f"{records_written=} records written to {output_path!s}")


if __name__ == "__main__":
    cli()
