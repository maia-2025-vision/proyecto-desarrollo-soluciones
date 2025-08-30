from pathlib import Path

import tqdm
import typer
from loguru import logger
from PIL import Image

cli = typer.Typer()


@cli.command()
def identify_broken_images(
    images_dir: Path,
    output_path: Path,  # list of broken images will go here
    ext: str = typer.Option("JPG"),
) -> None:
    """Identify corrupted image files that can't be read.

    List of paths is written to output_path
    """
    img_paths = list(images_dir.glob(f"*.{ext}"))

    logger.info(f"images_dir: {images_dir} - found {len(img_paths)} images")
    broken_images = []
    for img_path in tqdm.tqdm(img_paths):
        try:
            Image.open(img_path).convert("RGB")
        except OSError:
            logger.info(f"OSError for: {img_path!s} ")
            broken_images.append(img_path)

    logger.info(f"{len(broken_images)} broken images found, " f"writing list to: {output_path!s}")
    with open(output_path, "w") as f_out:
        for img_path in broken_images:
            print(str(img_path), file=f_out)


if __name__ == "__main__":
    cli()
