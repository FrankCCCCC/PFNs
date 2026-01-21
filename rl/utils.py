from typing import Tuple

import torch
from pfns.model import transformer
from pfns.train import MainConfig


def load_config_and_model(
    path: str = "tree/pfns/runs/singletaskgp2_clusteredx_0/checkpoint.pt",
    map_location: str | None = None,
) -> Tuple[MainConfig, transformer.TableTransformer]:
    """
    Load a config and model from a checkpoint file on the local filesystem.

    Args:
        path: The path to the checkpoint file.
        map_location: The device to load the tensors to. Same as torch.load, e.g. "cpu" or "cuda:0".

    Returns:
        A tuple of (config, model).
    """
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(path, map_location=map_location)

    config_dict = checkpoint["config"]

    c = MainConfig.from_dict(config_dict)
    model = c.model.create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return c, model.to(map_location)
