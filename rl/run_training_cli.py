#!/usr/bin/env python3
"""
Command-line interface for training PFNs models.
"""

import argparse
import io
import os
import sys
from functools import partial
from pathlib import Path

import pfns.run_training_cli as original_cli
import pfns.train
import torch

from manifold.clients.python import ManifoldClient

from .discrete_eval import evaluate_bo_on_hpob
from .discrete_pfns_bayesopt import get_acquisition_values_pfn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a PFNs model using configuration from a Python file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the Python configuration file that defines a 'config' variable or `get_config` function. Path is relative to PFNs/fb.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cuda:0', 'cpu', 'mps'). If not specified, will auto-detect cuda, but not mps.",
    )

    parser.add_argument(
        "--checkpoint-save-load-prefix",
        type=str,
        default=None,
        help="Path to save/load checkpoint and for tensorboard.",
    )

    parser.add_argument(
        "--checkpoint-save-load-suffix",
        type=str,
        default="",
        help="Suffix to add to the checkpoint save/load path. This can e.g. be the seed.",
    )

    parser.add_argument(
        "--tensorboard-path",
        type=str,
        default=None,
        help=(
            "Path to save tensorboard. If not provided, will use the "
            "checkpoint save/load prefix or the path in the config file."
        ),
    )

    parser.add_argument(
        "--config-index",
        type=int,
        default=0,
        help="Index of the config to use. This is used to select a config from the config file.",
    )

    return parser.parse_args()


def manifold_load(path: str, map_location: str | None = None) -> object:
    """
    A wrapper around torch.load with the same API.
    Loads from manifold instead of local disk, though.

    Args:
        path: The path to the file to load. The path has the format: manifold://<bucket>/<path>.
        map_location: The device to load the tensors to. Same as torch.load, e.g. "cpu" or "cuda:0".

    Returns:
        The loaded object.
    """

    with ManifoldClient.get_client("ae_generic") as client:
        stream = io.BytesIO()
        client.sync_get(path, stream)
        stream.seek(0)
        return torch.load(stream, map_location=map_location, weights_only=True)


def manifold_exists(path: str) -> bool:
    """
    A replacement for os.path.exists that works for manifold paths.

    Args:
        path: The path to check. The path has the format: manifold://<bucket>/<path>.

    Returns:
        True if the path exists, False otherwise.
    """

    with ManifoldClient.get_client("ae_generic") as client:
        return client.sync_exists(path)


def manifold_save(obj, path: str):
    """
    A wrapper around torch.save with the same API that saves to manifold.

    Args:
        obj: The object to save.
        path: The path to save the object to.
            The path has the format: manifold://<bucket>/<path>.

    Returns:
        None
    """
    dir_path = os.path.dirname(path)

    assert dir_path != "", "dir_path must not be empty"

    with ManifoldClient.get_client("ae_generic") as client:
        if not client.sync_exists(dir_path):
            client.sync_mkdirs(dir_path)
            print("made path")

        stream = io.BytesIO()
        torch.save(obj, stream)
        stream.seek(0)
        client.sync_put(
            path, stream, predicate=ManifoldClient.Predicates.AllowOverwrite
        )


def main():
    """Main CLI entry point."""
    args = parse_args()

    config_file = args.config_file
    config_file = config_file[3:] if config_file.startswith("fb/") else config_file

    # Load configuration from Python file
    config = original_cli.load_config_from_python(
        config_file, args.config_index, config_base_path=Path(__file__).parent
    )

    def get_filename(config_file):
        return Path(config_file).stem

    if args.checkpoint_save_load_suffix:
        assert (
            args.checkpoint_save_load_prefix is not None
        ), "checkpoint_save_load_prefix is required when checkpoint_save_load_suffix is provided"

    config_tensorboard_path_is_none = config.tensorboard_path is None

    # Override checkpoint paths if specified via CLI
    if args.checkpoint_save_load_prefix is not None:
        assert (
            config.train_state_dict_save_path is None
        ), "train_state_dict_save_path is already set"
        assert (
            config.train_state_dict_load_path is None
        ), "train_state_dict_load_path is already set"
        assert config_tensorboard_path_is_none, "tensorboard_path is already set"

        # Add suffix if it exists
        suffix = f"_{args.config_index}"
        if args.checkpoint_save_load_suffix:
            suffix += f"_{args.checkpoint_save_load_suffix}"

        path = f"{args.checkpoint_save_load_prefix}/{get_filename(config_file)}{suffix}"

        config = config.__class__(
            **{
                **config.__dict__,
                "train_state_dict_save_path": path + "/checkpoint.pt",
                "train_state_dict_load_path": path + "/checkpoint.pt",
                "tensorboard_path": "manifold://ae_generic/" + path + "/tensorboard",
            }
        )

    if args.tensorboard_path is not None:
        assert config_tensorboard_path_is_none, "tensorboard_path is already set"
        config = config.__class__(
            **{
                **config.__dict__,
                "tensorboard_path": args.tensorboard_path,
            }
        )

    # We overwrite the config with the one from the checkpoint if it exists
    # as there is some randomness in the config and we want to use the exact
    # same config again.
    if pfns.train.should_load_checkpoint(
        config, check_path_exists_function=manifold_exists
    ):
        config = pfns.train.load_config(
            config.train_state_dict_load_path, load_function=manifold_load
        )

    print("Starting training with configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Steps per epoch: {config.steps_per_epoch}")
    print(f"  Device: {args.device or 'auto-detect'}")
    print(f"  Mixed precision: {config.train_mixed_precision}")

    try:
        result = pfns.train.train(
            c=config,
            device=args.device,
            # overrides for filesystem things
            save_object_function=manifold_save,
            load_object_function=manifold_load,
            check_path_exists_function=manifold_exists,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

    print("\nTraining completed successfully!")
    print(f"Total training time: {result['total_time']:.2f} seconds")
    print(f"Final loss: {result['total_loss']:.6f}")

    if config.train_state_dict_save_path is not None:
        print(f"Model saved to: {config.train_state_dict_save_path}")
        # run eval
        # todo use manifold_save and manifold_load instead of torch
        device = args.device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = result["model"].to(device)
        acq_function = partial(get_acquisition_values_pfn, model=model, device=device)
        results = evaluate_bo_on_hpob(acq_function, verbose=True)
        torch.save(
            results,
            "manifold://ae_generic/"
            + str(Path(config.train_state_dict_save_path).parent)
            + "/hpob_results.pt",
        )


if __name__ == "__main__":
    main()
