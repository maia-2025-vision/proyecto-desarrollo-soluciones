import math
from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader


def train_validation_split(
    imgs_dir: Path,
    *,
    sort_paths: bool = True,
    valid_fraction: float,
    train_fraction: float | None = None,
    ext: str = "JPG",
    strategy: str = "simple",
) -> tuple[list[Path], list[Path]]:
    """Produce two lists of paths to images relative contained in imgs_dir.

    (imgs_dir is assumed to be an absolute path or a path relative to current working directory).
    Fractions are relative to the number of images in `imgs_dir`
    We allow to vary train and valid fractions independently so that we can run quick tests
    with fewer train and valid images (e.g train_fraction = 0.1, valid_fraction = 0.05)

    :param imgs_dir: A directory of images
    :param valid_fraction: A fraction of the total number of images in the dir
    :param train_fraction: A
    :param ext: the images extension, e.g. JPG
    :param strategy:
        - "simple" take the train images from the beginning of the listing and
          the valid images from the end this diminishes changes of overlap between
          images in both sets.
        - "random" to be implemented
        - more to come later maybe...

    :return: (list of train paths, list of valid paths)
    """
    all_imgs = list(imgs_dir.rglob(f"*.{ext}"))
    if sort_paths:
        logger.info(f"Sorting {len(all_imgs)} images from {imgs_dir!s}")
        all_imgs = sorted(all_imgs, key=lambda x: x.name)
    n_imgs = len(all_imgs)
    assert (
        0.0 < valid_fraction < 1.0
    ), f"Invalid validation fraction: {valid_fraction}, should be between 0.0 and 1.0"

    logger.info(f"{n_imgs} images with {ext} extension found in {imgs_dir}")

    if train_fraction is None:
        train_fraction = 1.0 - valid_fraction

    assert (
        0.0 < train_fraction < 1.0
    ), f"Invalid training fraction: {train_fraction}, should be > 0.0"

    logger.info(
        f"Splitting train/valid = {train_fraction}/{valid_fraction} with strategy: {strategy}"
    )

    if strategy == "simple":
        n_valid = int(math.floor(n_imgs * valid_fraction))
        n_train = int(math.floor(n_imgs * train_fraction))
        assert 0 < n_train < n_imgs, f"Invalid {n_train=}: {train_fraction=}, {n_imgs=}"
        assert 0 < n_valid < n_imgs, f"Invalid {n_valid=}: {valid_fraction=}, {n_imgs=}"

        return all_imgs[:n_train], all_imgs[-n_valid:]
    else:
        raise NotImplementedError(f"Strategy={strategy!r} not yet implemented")


def get_num_batches(dl: DataLoader) -> int:
    """Estimate number of batches produced by a dataloader."""
    n_items = len(dl.dataset)  # type: ignore[arg-type]
    batch_size = dl.batch_size
    assert isinstance(batch_size, int)

    return int(math.ceil(n_items / batch_size))
