import json
from collections import Counter
from pathlib import Path

from loguru import logger
from pydantic import BaseModel
from typer import Typer

cli = Typer()


class BboxMinMax(BaseModel):
    """A bounding box defined by a top-left corner and a bottom-right corner."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def as_list(self) -> list[int]:
        """Convert myself to a list."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]


def parse_yolo_annotation_line(
    line: str, *, img_width: int, img_height: int
) -> tuple[int, BboxMinMax]:
    """Parse a yolo formatted annotation line.

    I.e. we assume the format
    <class_id> <x_center> <y_center> <bbox_width> <bbox_height>

    Returns:
        class_id (int), BBox
    """
    parts = line.strip().split()
    if len(parts) == 5:
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
        x_center *= img_width
        y_center *= img_height
        bbox_width *= img_width
        bbox_height *= img_height

        x_min = x_center - bbox_width / 2
        y_min = y_center - bbox_height / 2
        x_max = x_center + bbox_width / 2
        y_max = y_center + bbox_height / 2

        return int(class_id), BboxMinMax(
            x_min=int(x_min), y_min=int(y_min), x_max=int(x_max), y_max=int(y_max)
        )


def parse_yolo_annotation_file(
    annotation_path: Path, img_width: int, img_height: int
) -> tuple[list[list[int]], list[int]]:
    """Parse a plain text file consisting of yolo formatted annotations."""
    boxes: list[list[int]] = []  # [x_min, y_min, x_max, y_max] formatted boxes
    labels: list[int] = []  # class-id for each box

    with annotation_path.open("rt") as f:
        for line in f.readlines():
            class_id, bbox = parse_yolo_annotation_line(
                line, img_width=img_width, img_height=img_height
            )
            boxes.append(bbox.as_list())
            labels.append(class_id)

    return boxes, labels


def parse_json_annotations_file(
    annotation_path: Path, class_name_to_id: dict[str, int]
) -> tuple[list[list[int]], list[int]]:
    """Parse rectangle type annotations assumed to be in "objects" member of the json document."""
    obj = json.loads(annotation_path.read_text())

    annots = obj["objects"]
    ignored_cnts = Counter()

    boxes: list[list[int]] = []
    labels: list[int] = []

    for annot in annots:
        class_name = annot["classTitle"]
        if class_name not in class_name_to_id:
            ignored_cnts[class_name] += 1
            continue

        class_id = class_name_to_id[class_name]
        labels.append(class_id)

        assert annot["geometryType"] == "rectangle", f"{annot['geometryType']} != 'rectangle'"
        p_min, p_max = annot["points"]["exterior"]
        x_min, y_min = p_min
        x_max, y_max = p_max

        assert x_min < x_max, f"it's not the case that {x_min} < {x_max}"
        assert y_min < y_max, f"it's not the case that {y_min} < {y_max}"

        boxes.append([x_min, y_min, x_max, y_max])

    if len(ignored_cnts) > 0:
        logger.warning(f"{annotation_path!s} - ignored: {ignored_cnts}")

    assert len(labels) == len(boxes), f"{len(labels)} != {len(boxes)}"

    return boxes, labels


@cli.command()
def scan_json_annots(annots_dir: Path):
    """Test parsing over a directory of json annotations."""
    import tqdm

    annot_files = list(annots_dir.glob("*.json"))
    class_name_to_id: dict[str, int] = {"cattle": 0}

    n_files = 0
    n_boxes = 0
    n_labels = 0
    for file in tqdm.tqdm(annot_files):
        boxes, labels = parse_json_annotations_file(file, class_name_to_id)
        n_boxes += len(boxes)
        n_labels += len(labels)
        n_files += 1

    logger.info(f"Parsed {n_files} json files, found {n_boxes} boxes and {n_labels} labels")


if __name__ == "__main__":
    cli()
