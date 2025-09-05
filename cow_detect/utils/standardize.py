"""Script to standardize training data into a flat format"""

from collections import Counter
from typing import Literal
from pathlib import Path

import PIL
from pydantic import BaseModel, Field
import typer
from loguru import logger
from PIL import Image


from cow_detect.utils.annotations import parse_json_annotations_file

cli = typer.Typer(pretty_exceptions_show_locals=False)


class ImageInfo(BaseModel):
    path: Path
    n_bytes: int = Field(description="Image size in bytes")
    status: Literal["ok", "not-found", "unreadable"]
    width: int | None = Field(description="Image width in pixels", nullable=True)
    height: int | None = Field(description="Image height in pixels", nullable=True)
    error: str = Field(
        "", description="Any error encountered while trying to load the image", nullable=True
    )


class BBox(BaseModel):
    label: str = Field(
        description="Class label of the object in the bbox, e.g. 'cow', 'cattle', 'calf' "
    )
    coords: list[float] = Field(
        description="Bbox coordinates in Pascal VOC format [x_min, y_min, x_max, y_max],"
       "where 0 <= x_min < x_max <= image.width and 0 <= y_min < y_max <= image.height"
      "could be ints of floats"
    )
    tags: dict[str, object] = Field(description="an arbitrary dictionary of tags keyed by name")


class AnnotInfo(BaseModel):
    path: Path
    status: Literal["ok", "not-found", "unreadable", "unparseable"]
    size: int | None = Field(description="Annotation file size in bytes", nullable=True)
    n_lines: int | None = Field(description="number of lines", nullable=True)
    bboxes: list[BBox] = Field(description="Bounding boxes", nullable=True)


class AnnotatedImageRecord(BaseModel):
    image: ImageInfo
    annotations: AnnotInfo


class SkyTraits:
    def __init__(self, images_root_path: Path):
        assert images_root_path.name == "img", f"{images_root_path=}, {images_root_path.name=} is not 'img'"
        self.images_root_path = images_root_path

    def annots_path(self, image_path: Path) -> Path:
        image_fname = image_path.name
        return (image_path.parent.parent / image_fname).with_suffix(".json")

    def read_annots(self, annots_path: Path) -> AnnotInfo:
        # TODO: maybe add tag parsing here?
        bboxes_coords, labels = parse_json_annotations_file(annots_path)

        size = annots_path.stat().st_size
        n_lines = len(annots_path.read_text().splitlines())

        bboxes = []
        for coords, label in zip(bboxes_coords, labels):
            bboxes.append(BBox(label=label, coords=coords, tags={}))

        return AnnotInfo(
            path=annots_path,
            status="ok",
            size=size,
            n_lines=n_lines,
            bboxes=bboxes,
        )

    def process_one(self, image_path: Path) -> AnnotatedImageRecord:
        img_info =get_image_info(image_path)
        annots_path = self.annots_path(image_path)
        annot_info = self.read_annots(annots_path)

        return AnnotatedImageRecord(image=img_info, annotations=annot_info)


def get_image_info(image_path: Path) -> ImageInfo:
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
            error=str(e),
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
       ..., "-o", "--out",
        help="output will go here, should have .jsonl extension!"
    )
):
    assert output_path.suffix == ".jsonl", f"{output_path.suffix=} is not .jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assert images_root_dir.is_dir(), f"{images_root_dir=} is not a directory"

    if kind == "sky":
        traits = SkyTraits(images_root_dir)
    elif kind == "icaerus":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown kind: `{kind}`")

    image_paths = sorted(images_root_dir.rglob("*.JPG"))

    logger.info(f"{len(image_paths)} images found.")

    image_stats: Counter[str] = Counter()
    annot_stats: Counter[str] = Counter()

    records_written = 0
    with output_path.open("wt") as f_out:
        for image_path in image_paths:
            record_out = traits.process_one(image_path)
            image_stats[record_out.image.status.lower()] += 1
            annot_stats[record_out.annotations.status.lower()] += 1
            f_out.write(record_out.model_dump_json())
            records_written += 1

    logger.info(f"{image_stats=}")
    logger.info(f"{image_stats=}")
    logger.info(f"{records_written=} records written to {output_path!s}")


if __name__ == "__main__":
    cli()
