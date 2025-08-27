class DataLoaderParams:
    """Parameters for the data loader."""

    batch_size: int
    data_shuffle: bool
    num_workers: int


class OptimizerParams:
    """Parameters for the optimizer."""

    learning_rate: float
    momentum: float
    weight_decay: float
