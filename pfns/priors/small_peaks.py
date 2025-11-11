#!/usr/bin/env python3
"""
Example configuration file for PFN training.
This is a Hebo+ prior configuration, as found in the PFNs4BO paper.
This file demonstrates how to configure the MainConfig for training using Python.
"""

import math

import torch
from pfns.model import bar_distribution
from pfns.model.encoders import EncoderConfig
from pfns.priors.prior import AdhocPriorConfig
from pfns.train import (
    BatchShapeSamplerConfig,
    MainConfig,
    OptimizerConfig,
    TransformerConfig,
)
from pfns.utils import product_dict

from tqdm import tqdm

config_dicts = product_dict(
    {
        "emsize": [256],
        "nlayers": [12],
        "epochs": [200],
        "lr": [2e-4],
        "batch_size": [256],
        "batch_size_per_gp_sample": [8],
        "num_workers": [6],  # while more workers would be good they lead to segfaults
        "max_seq_len": [60, 120],
        "num_buckets": [1000, 5000],
        "encoder_hidden_size": [1024, None],
    }
)


def get_config(config_index: int):
    config_dict = list(config_dicts)[config_index]

    emsize = config_dict["emsize"]
    epochs = config_dict["epochs"]
    lr = config_dict["lr"]
    nlayers = config_dict["nlayers"]
    batch_size = config_dict["batch_size"]
    num_workers = config_dict["num_workers"]
    max_seq_len = config_dict["max_seq_len"]

    steps_per_epoch = 1000
    num_features = 2
    hyperparameters = {}

    def get_prior_config(plotting=False):
        prior_config = AdhocPriorConfig(
            prior_names=["small_peaks"],
            prior_kwargs={
                "num_features": 1 if plotting else num_features,
                "hyperparameters": {**hyperparameters},
                "batch_size_per_gp_sample": config_dict["batch_size_per_gp_sample"],
            },
        )
        return prior_config, hyperparameters

    prior_config, hps = get_prior_config()

    gb = prior_config.create_get_batch_method()

    ys = []
    for nf in tqdm(list(range(1, num_features)) * 200):
        ys.append(gb(batch_size=128, seq_len=1000, num_features=nf).target_y.flatten())

    ys = torch.cat(ys)
    print(f"{len(ys)=} for {config_dict['num_buckets']=}")

    borders = bar_distribution.get_bucket_borders(config_dict["num_buckets"], ys=ys)

    print(f"{borders=}")

    return MainConfig(
        priors=[prior_config],
        optimizer=OptimizerConfig("adamw", lr=lr, weight_decay=0.0),
        scheduler="cosine_decay",
        model=TransformerConfig(
            criterion=bar_distribution.BarDistributionConfig(
                borders.tolist(), full_support=True
            ),
            emsize=emsize,
            nhead=emsize // 32,
            nhid=emsize * 4,
            nlayers=nlayers,
            encoder=EncoderConfig(
                variable_num_features_normalization=True,
                constant_normalization_mean=0.5,
                constant_normalization_std=1 / math.sqrt(12),
                hidden_size=config_dict["encoder_hidden_size"],
            ),
            y_encoder=EncoderConfig(
                nan_handling=True,
                constant_normalization_mean=0.5,
                constant_normalization_std=1 / math.sqrt(12),
                hidden_size=config_dict["encoder_hidden_size"],
            ),
            attention_between_features=True,
        ),
        batch_shape_sampler=BatchShapeSamplerConfig(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            fixed_num_test_instances=10,
            max_num_features=num_features,
        ),
        epochs=epochs,
        warmup_epochs=epochs // 10,
        steps_per_epoch=steps_per_epoch,
        num_workers=num_workers,
        train_mixed_precision=True,
        verbose=True,
    )


# View with: tensorboard --logdir=runs
