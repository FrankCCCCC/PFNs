#!/usr/bin/env python3
"""Command-line interface for running RL fine-tuning of PFN models."""

import copy
import math
import os
import random
import time
import typing as tp
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from pfns import base_config
from pfns.model import transformer_config
from pfns.model.encoders import StyleEncoderConfig
from pfns.model.transformer import TableTransformer
from pfns.priors.utils import sample_x_around_points
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from .function_samplers import function_sampler  # noqa: F401
from .utils import load_config_and_model


def local_load(path: str, map_location: str | None = None) -> object:
    """
    Load a checkpoint from the local filesystem.

    Args:
        path: The path to the file to load.
        map_location: The device to load the tensors to. Same as torch.load, e.g. "cpu" or "cuda:0".

    Returns:
        The loaded object.
    """
    return torch.load(path, map_location=map_location, weights_only=True)


def local_exists(path: str) -> bool:
    """
    Check if a path exists on the local filesystem.

    Args:
        path: The path to check.

    Returns:
        True if the path exists, False otherwise.
    """
    return os.path.exists(path)


def local_save(obj, path: str):
    """
    Save an object to the local filesystem.

    Args:
        obj: The object to save.
        path: The path to save the object to.

    Returns:
        None
    """
    dir_path = os.path.dirname(path)

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    torch.save(obj, path)


@dataclass
class PathGenerationResult:
    """Result of path generation for RL training.

    Attributes:
        ys: Tensor of y values, shape [batch_size * sub_batch_size, seq_len]
        target_ys: Tensor of target y values, shape [batch_size * sub_batch_size, seq_len]
        predictions: Tensor of predictions, shape [batch_size * sub_batch_size, seq_len, num_features]
        options: List of option tensors (if choose_next_in_set=True)
        chosen_options: List of chosen option indices (if choose_next_in_set=True)
        choice_probs: List of choice probability tensors (if choose_next_in_set=True)
        current_num_features: Number of features used for this generation
        draw: Random draw tensor for computing regret, shape [batch_size, draw_size]
        y_quantiles: Quantiles of y values, shape [batch_size * sub_batch_size, seq_len]
        draw_size: Size of the random draw
        joint_steps: Number of initial joint steps used during generation
        step_entropies: List of entropy values per step (for tensorboard logging)
        step_max_probs: List of max probability values per step (for tensorboard logging)
        step_sampled_probs: List of sampled probability values per step (for tensorboard logging)
    """

    ys: torch.Tensor
    target_ys: torch.Tensor
    predictions: torch.Tensor
    options: list[torch.Tensor]
    chosen_options: list[torch.Tensor]
    choice_probs: list[torch.Tensor]
    current_num_features: int
    joint_steps: int
    basemodel_ei_values: list[
        torch.Tensor
    ]  # EI values from unfinetuned basemodel for each step
    step_entropies: list[torch.Tensor]  # Entropy of distribution at each step
    step_max_probs: list[torch.Tensor]  # Max probability at each step
    step_sampled_probs: list[torch.Tensor]  # Probability of sampled action at each step
    binary_features_mask: (
        torch.Tensor | None
    )  # Mask indicating which features are binary, shape [batch_size, num_features]
    bo_batch_size: int  # The bo_batch_size used for this generation
    seq_len: int  # The seq_len used for this generation (for random horizon training)
    y_style: torch.Tensor | None  # The y_style tensor used for this generation

    # placeholders to be filled by subsequent computations
    normalized_avg_rewards: torch.Tensor | None = None
    unnormalized_avg_rewards: torch.Tensor | None = None
    draw: torch.Tensor | None = None
    y_quantiles: torch.Tensor | None = None
    target_y_quantiles: torch.Tensor | None = None
    standardized_ys: torch.Tensor | None = None
    standardized_target_ys: torch.Tensor | None = None
    draw_size: int | None = None

    def to_device(self, device):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device))
            elif isinstance(value, list):
                new_list = [
                    item.to(device) if isinstance(item, torch.Tensor) else item
                    for item in value
                ]
                setattr(self, field.name, new_list)
        return self


@dataclass(frozen=True)
class RewardConfig(base_config.BaseConfig):
    reward_type: tp.Literal[
        "raw", "quantile", "standardized", "log_quantile", "rs_equivalent"
    ] = "raw"
    standardization_source: tp.Literal["batch", "draw"] = "draw"
    only_future: bool = False
    aggregation: str = (
        "max"  # we can sum instead of mean, because everything is standardized anyways
        # Options: "sum", "max", "avgmax", "max_imp", "max_sparse", "myopic_X" (where X is 0, 1, 2, ...)
        # myopic_0: only current position, myopic_1: current + next, etc.
    )
    standardization: tp.Literal[
        "none",
        "per_step_and_function",
        "divide_per_step_and_function",
        "mean_divide_per_step_and_function",
        "mean_sub_per_step_and_function",
        "mean_divide_per_function",
        "top_0.1_per_function",
        "top_0.2_per_function",
    ] = "per_step_and_function"
    standardization_eps: float = 1e-8
    reward_on_targets: bool = False
    no_reward_after_peak: bool | tp.Literal["global"] = False

    @classmethod
    def _loading_kwarg_transform(cls, kwargs):
        if "quantile_reward" in kwargs:
            qr = kwargs.pop("quantile_reward")
            kwargs["reward_type"] = "quantile" if qr else "raw"
        return kwargs

    def __post_init__(self):
        if self.aggregation in ("max_imp", "max_sparse"):
            assert (
                self.only_future
            ), f"{self.aggregation} does only make sense with future rewards."
        if self.aggregation.startswith("myopic_"):
            assert (
                self.only_future
            ), f"{self.aggregation} does only make sense with future rewards."
            try:
                window = int(self.aggregation.split("_")[1])
                assert window >= 0, f"myopic window must be non-negative, got {window}"
            except (IndexError, ValueError):
                raise ValueError(
                    f"Invalid myopic aggregation format: {self.aggregation}. Expected 'myopic_X' where X is a non-negative integer."
                )

    @torch.no_grad()
    def compute_reward(
        self,
        ys: torch.Tensor,
        target_ys: torch.Tensor,
        quantile_ys: torch.Tensor,
        quantile_target_ys: torch.Tensor,
        standardized_ys: torch.Tensor,
        standardized_target_ys: torch.Tensor,
    ):
        batch_size, sub_batch_size, seq_len = ys.shape

        if self.reward_on_targets:
            ys = target_ys
            quantile_ys = quantile_target_ys
            standardized_ys = standardized_target_ys

        if self.reward_type == "quantile":
            rewards_curr_pos = quantile_ys
        elif self.reward_type == "standardized":
            rewards_curr_pos = standardized_ys
        elif self.reward_type == "log_quantile":
            quantile_regret = (1 - quantile_ys).clamp(
                min=1 / 10_000
            )  # clamp s.t. we don't get log(0.) errors
            log_quantile_regret = torch.log(quantile_regret)
            rewards_curr_pos = -log_quantile_regret
        elif self.reward_type == "rs_equivalent":
            # Transform quantile to equivalent random search size
            # 1/(1-q) represents how many random samples would be needed on average
            # to find a value at least this good (e.g., q=0.99 -> 100 samples)
            quantile_regret = (1 - quantile_ys).clamp(
                min=1 / 10_000
            )  # clamp s.t. we don't get division by 0
            rewards_curr_pos = 1.0 / quantile_regret
        else:
            assert self.reward_type == "raw", f"Unknown reward_type: {self.reward_type}"
            rewards_curr_pos = ys

        if self.only_future:
            if self.aggregation == "sum":
                timestep_rewards = torch.cumsum(rewards_curr_pos.flip(-1), dim=-1).flip(
                    -1
                )
            elif self.aggregation == "avgmax":
                max_so_far = torch.cummax(rewards_curr_pos, dim=-1)[0]
                timestep_rewards = torch.cumsum(max_so_far.flip(-1), dim=-1).flip(-1)
                # normalize sums to be averages
                num_remaining = torch.arange(
                    seq_len, 0, -1, device=timestep_rewards.device
                )
                timestep_rewards = timestep_rewards / num_remaining.view(1, 1, -1)
            else:
                timestep_rewards = torch.cummax(rewards_curr_pos.flip(-1), dim=-1)[
                    0
                ].flip(-1)
                if self.aggregation == "max_imp":  # try this with future rewards
                    # reward for the first one is a little random
                    # as it is against a baseline that is 0 starting
                    # for the quantiles it makes sense, but without not really
                    max_so_far = torch.cummax(rewards_curr_pos, dim=-1)[0]
                    timestep_improvement = timestep_rewards
                    timestep_improvement[..., 1:] -= max_so_far[..., :-1]
                    average_y_first_guess = rewards_curr_pos[:, :, 0].mean(
                        1, keepdim=True
                    )
                    timestep_improvement[..., 0] -= average_y_first_guess
                    timestep_rewards = timestep_improvement.clamp(min=0.0)
                elif self.aggregation == "max_sparse":
                    # todo: think about this more
                    max_so_far = torch.cummax(rewards_curr_pos, dim=-1)[0]
                    timestep_rewards[..., 1:] = torch.where(
                        max_so_far[..., :-1] > 0.0, 0.0, timestep_rewards[..., 1:]
                    )
                elif self.aggregation.startswith("myopic_"):
                    # myopic_X: only look at current position and the next X positions
                    window = int(self.aggregation.split("_")[1])
                    # For each position i, compute max over positions i to min(i+window, seq_len-1)
                    timestep_rewards = torch.zeros_like(rewards_curr_pos)
                    for i in range(seq_len):
                        end_idx = min(i + window + 1, seq_len)
                        timestep_rewards[..., i] = rewards_curr_pos[..., i:end_idx].max(
                            dim=-1
                        )[0]
                else:
                    assert self.aggregation == "max", self.aggregation
        else:
            if self.aggregation == "sum":
                timestep_rewards = rewards_curr_pos.sum(-1, keepdim=True).repeat(
                    1, 1, seq_len
                )
            elif self.aggregation == "avgmax":
                # Average of cumulative max across the sequence
                max_so_far = torch.cummax(rewards_curr_pos, dim=-1)[0]
                avg_max = max_so_far.mean(-1, keepdim=True)
                timestep_rewards = avg_max.repeat(1, 1, seq_len)
            else:
                assert self.aggregation == "max", self.aggregation
                timestep_rewards = rewards_curr_pos.max(-1, keepdim=True)[0].repeat(
                    1, 1, seq_len
                )

        if self.standardization == "per_step_and_function":
            normalized_avg_rewards_future = (
                timestep_rewards - timestep_rewards.mean(1, keepdim=True)
            ) / (timestep_rewards.std(1, keepdim=True) + self.standardization_eps)
        elif self.standardization == "mean_sub_per_step_and_function":
            normalized_avg_rewards_future = timestep_rewards - timestep_rewards.mean(
                1, keepdim=True
            )
        elif self.standardization == "divide_per_step_and_function":  # try this
            normalized_avg_rewards_future = timestep_rewards / (
                timestep_rewards.std(1, keepdim=True) + self.standardization_eps
            )
        elif self.standardization == "mean_divide_per_step_and_function":  # try this
            normalized_avg_rewards_future = timestep_rewards / (
                timestep_rewards.mean(1, keepdim=True) + self.standardization_eps
            )
        elif self.standardization == "mean_divide_per_function":
            normalized_avg_rewards_future = timestep_rewards / (
                timestep_rewards.mean(1, keepdim=True).mean(2, keepdim=True)
                + self.standardization_eps
            )
        elif self.standardization.startswith("top_") and self.standardization.endswith(
            "per_function"
        ):
            # generalize to any quantile, e.g. "top_0.1_per_function"
            try:
                quantile = float(self.standardization.split("_")[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f"Invalid standardization format: {self.standardization}"
                )
            # cutoff top quantile
            quantile_cutoffs = timestep_rewards.view(batch_size, -1).sort(-1)[0][
                :, -round(quantile * sub_batch_size * seq_len)
            ]
            normalized_avg_rewards_future = torch.where(
                timestep_rewards > quantile_cutoffs[:, None, None],
                timestep_rewards,
                0.0,
            )
        else:
            assert self.standardization == "none", self.standardization
            normalized_avg_rewards_future = timestep_rewards
        # normalized_avg_rewards_future = normalized_avg_rewards_future.view(
        #     batch_size * sub_batch_size, -1
        # )

        if self.no_reward_after_peak:
            # Set rewards to 0 for all timesteps after the peak evaluation
            position_indices = torch.arange(seq_len, device=rewards_curr_pos.device)

            if self.no_reward_after_peak == "global":
                # Find the global peak across all trajectories in each sub-batch
                # and use it to cut off rewards for all items
                # Flatten sub_batch and seq_len to find global max per batch
                flat_rewards = rewards_curr_pos.view(
                    batch_size, sub_batch_size * seq_len
                )
                global_peak_flat_indices = flat_rewards.argmax(dim=-1)  # [batch_size]
                # Convert flat index to seq_len index (the timestep of the global peak)
                global_peak_timestep = (
                    global_peak_flat_indices % seq_len
                )  # [batch_size]
                # Create mask: positions after global peak timestep are True for all trajectories
                after_peak_mask = position_indices.view(
                    1, 1, -1
                ) > global_peak_timestep.view(
                    batch_size, 1, 1
                )  # [batch_size, sub_batch_size, seq_len]
            else:
                assert self.no_reward_after_peak is True
                # Original behavior: find peak per trajectory
                peak_indices = rewards_curr_pos.argmax(
                    dim=-1
                )  # [batch_size, sub_batch_size]
                # Create a mask where positions after the peak are True
                after_peak_mask = position_indices.view(
                    1, 1, -1
                ) > peak_indices.unsqueeze(-1)  # [batch_size, sub_batch_size, seq_len]

            # Zero out rewards after the peak
            normalized_avg_rewards_future = torch.where(
                after_peak_mask,
                torch.zeros_like(normalized_avg_rewards_future),
                normalized_avg_rewards_future,
            )

        return (
            normalized_avg_rewards_future,
            timestep_rewards,
        )  # both shape: [batch_size, sub_batch_size, seq_len]


@dataclass(frozen=True)
class RLConfig(base_config.BaseConfig):
    model_path: str
    function_sampler: function_sampler.FunctionSamplerConfig
    reward: RewardConfig = RewardConfig()
    batch_size: int = 16
    sub_batch_size: int = 32  # in the scaling RL paper, they use 16
    seq_len: int = 5
    algorithm: tp.Literal["grpo", "cispo"] = "grpo"
    eps: float = 0.2
    eps_low: float | None = None  # If None, use eps for lower bound (symmetric)
    filter_rewards_up_to_magnitude: float | None = (
        None  # Zero out rewards for functions with avg magnitude <= this threshold
    )
    num_batches: int = 100
    experience_repetitions: int = 10
    learning_rate: float = 1e-5
    min_learning_rate: float | None = 0.0  # No LR schedule, when None
    lr_schedule: tp.Literal["cosine", "linear"] = "cosine"  # LR decay schedule type
    warmup_batches: int | None = None  # Optional linear LR warmup for this many batches
    opt_beta2: float = 0.99
    opt_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    device: str | None = None
    standardize_y: None | str = None
    seed: int = 21415
    independent_noise_in_function_draws: bool = False

    # For choice based RL
    choose_next_in_set: bool = False
    choice_set_size: int = 100
    super_choice_set_factor: float = 1.0
    choice_set_top_share: float = (
        0.5  # only relevant when super_choice_set_factor > 1.0
    )
    ei_selector: bool = False
    keep_head: bool = False
    num_features: int = 1  # Number of input features for choose_next_in_set mode
    mix_k_features_in_opt: int = 1
    basemodel_ei_input: bool = (
        False  # Pass EI from unfinetuned basemodel as input feature
    )
    binary_feature_likelihood: float = (
        0.0  # Probability that each feature is binary (0 or 1) in the choice set
    )

    # Around train point sampling: sample some options near previous training points
    around_train_point_share: float = (
        0.0  # Fraction of options to sample around training points (0.0 = disabled)
    )
    around_train_point_std: float = 0.01  # Standard deviation for Gaussian noise when sampling around training points

    # Joint rollout training: keep trajectories identical for a random number of initial steps
    # None: disabled (default), train on all positions independently
    # "single": train only on the first position after joint steps (the split point)
    # "remaining": train on all positions from the split point onwards
    joint_rollout_training: tp.Literal["single", "remaining"] | None = None

    # Batched BO: select multiple points per batch with NaN y values for pending evaluations
    # When bo_batch_size > 1, points within a batch don't see each other's y values
    # and rewards are copied from the last position in each batch to all positions
    bo_batch_size: int = 1
    randomize_bo_batch_size: bool = False  # If True, sample bo_batch_size uniformly from 1 to bo_batch_size for each rollout

    # Random horizon: sample seq_len uniformly at random up to the specified value for each rollout
    randomize_seq_len: bool = False
    # When randomize_seq_len is True, add a y_style_encoder that encodes the current seq_len
    # The seq_len is normalized as (curr_seq_len/seq_len)*2-1 to be in range [-1, 1]
    seq_len_y_style_encoder: bool = False

    # Mixed precision for path generation
    mixed_precision_path_generation: bool = False

    # Checkpointing
    checkpoint_save_path: str | None = None  # Path to save checkpoints after each batch
    checkpoint_load_path: str | None = (
        None  # Path to load checkpoint from to resume training
    )

    # Rollback on high loss
    rollback_loss_threshold: float | None = (
        None  # Roll back optimizer step if loss exceeds this threshold
    )

    # Sequence length curriculum: scales seq_len with factors of 2 until reaching target
    # Options:
    #   - None: No curriculum, use seq_len for all batches
    #   - "equal": Train equal number of batches at each curriculum stage
    #   - "exponential": Train exponentially fewer batches at longer sequence lengths
    #                    (double seq_len, half the batches at each stage)
    seq_len_curriculum: tp.Literal["equal", "exponential"] | None = None
    seq_len_curriculum_min: int = 8  # Minimum sequence length for curriculum stages

    # Filled in automatically
    tensorboard_path: str | None = None
    model: transformer_config.TransformerConfig | None = None

    # make it backwards compatible
    @classmethod
    def _loading_kwarg_transform(cls, kwargs):
        kwargs.pop("output_checkpoint_path", None)

        if kwargs.pop("single_position_training", None):
            kwargs["joint_rollout_training"] = "single"

        return kwargs

    def __post_init__(self):
        if self.bo_batch_size > 1 and not self.choose_next_in_set:
            raise ValueError(
                "bo_batch_size > 1 is only supported when choose_next_in_set=True"
            )
        if self.binary_feature_likelihood > 0.0 and not self.choose_next_in_set:
            raise ValueError(
                "binary_feature_likelihood > 0 is only supported when choose_next_in_set=True"
            )
        if self.around_train_point_share > 0.0 and not self.choose_next_in_set:
            raise ValueError(
                "around_train_point_share > 0 is only supported when choose_next_in_set=True"
            )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Get the underlying module from a DDP-wrapped model or return the model itself."""
    if isinstance(model, DDP):
        return model.module
    return model


def transform_logits(model: TableTransformer, logits: torch.Tensor, ei_selector: bool):
    if not ei_selector:
        return logits.squeeze(-1)
    else:
        # Handle DDP-wrapped models
        model_module = unwrap_model(model)
        return model_module.criterion.ei(logits, best_f=0.0) * 1_000_000_000


def preprocess_train_x_and_y(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    standardize_y: str | None,
    bo_batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:  # both input tensors are [b,seq,features / 1]
    """Preprocess training data for model input.

    Args:
        train_x: Training x values, shape [batch, seq_len, num_features]
        train_y: Training y values, shape [batch, seq_len, 1]
        standardize_y: Standardization method ("m0s1" or "m0.5s0.3" or None)
        bo_batch_size: Batch size for batched BO. When > 1, y values for
            previous elements in the current batch are set to NaN (as they
            haven't been evaluated yet in batched BO).

    Returns:
        Tuple of (train_x, train_y) with appropriate preprocessing applied.
    """
    assert train_x.numel() > 0
    current_seq_len = train_y.shape[1]

    # For batched BO, determine which positions have been "evaluated"
    # Positions in previous batches are evaluated, positions in current batch are pending
    if bo_batch_size > 1 and current_seq_len > 0:
        batch_start = (current_seq_len // bo_batch_size) * bo_batch_size
        # Positions 0 to batch_start-1 are evaluated (from previous batches)
        # Positions batch_start to current_seq_len-1 are in current batch (pending)
        evaluated_y = train_y[:, :batch_start, :] if batch_start > 0 else None
    else:
        evaluated_y = train_y
        batch_start = current_seq_len

    if standardize_y:
        # Compute mean/std only over evaluated positions (previous batches)
        if evaluated_y is not None and evaluated_y.shape[1] > 0:
            mean = evaluated_y.mean(1, keepdim=True)
            if evaluated_y.shape[1] == 1:
                std = 1.0
            else:
                std = evaluated_y.std(1, keepdim=True)
                std[std < 1e-8] = 1.0
        else:
            # No evaluated positions yet, use defaults
            mean = 0.0
            std = 1.0

        # Apply standardization (NaNs remain NaN)
        train_y = (train_y - mean) / std

        if standardize_y == "m0.5s0.3":
            train_y = train_y * 0.3 + 0.5
        else:
            assert standardize_y == "m0s1"
    else:
        assert standardize_y is None

    # Still need to NaN out batch positions even without standardization
    if bo_batch_size > 1 and batch_start < current_seq_len:
        train_y = train_y.clone()
        train_y[:, batch_start:, :] = torch.nan

    return train_x, train_y


@torch.no_grad()
def generate_paths(
    model: TableTransformer,
    sampler: tp.Callable,
    batch_size: int,
    seq_len: int,
    sub_batch_size: int,
    choose_next_in_set: bool = False,
    choice_set_size: int = 100,
    super_choice_set_factor: float = 1.0,
    top_share: float = 0.5,  # only relevant when super_choice_set_factor > 1.0
    ei_selector: bool = False,
    standardize_y: str | None = None,
    argmax_selection: bool = False,
    current_num_features: int = 1,
    device: str = "cuda:0",
    joint_steps: int = 0,
    basemodel_ei_input: bool = False,
    basemodel_for_ei: TableTransformer | None = None,
    mixed_precision: bool = False,
    bo_batch_size: int = 1,
    binary_feature_likelihood: float = 0.0,
    around_train_point_share: float = 0.0,
    around_train_point_std: float = 0.01,
    y_style: torch.Tensor | None = None,
) -> PathGenerationResult:
    """Generate paths for a sampler.

    Args:
        model: The model to use for generating predictions
        sampler: Sampler function taking inputs (batch_size, sub_batch_size) returning tuple (batch_size, sub_batch_size)
        batch_size: Number of batches
        seq_len: Number of sequential predictions to make
        sub_batch_size: Number of sub-batches per sampler
        choose_next_in_set: Whether to choose next point from a set of options
        choice_set_size: Size of the choice set when choose_next_in_set=True
        ei_selector: Whether to use EI selector
        standardize_y: Standardization method for y values
        argmax_selection: Whether to use argmax for selection
        current_num_features: Number of features to use for this batch (constant across all steps)
        device: Device to run on
        joint_steps: Number of initial steps where all sub_batches share the same trajectory.
            During these steps, only one sample is taken per batch (not per sub_batch).
            At step joint_steps, trajectories are expanded to sub_batch_size copies each.

    Returns:
        Dictionary with keys:
            - ys: Tensor of shape [batch_size * sub_batch_size, seq_len]
            - predictions: Tensor of predictions
            - options: List of option tensors (if choose_next_in_set=True)
            - chosen_options: List of chosen option indices (if choose_next_in_set=True)
    """
    if current_num_features > 1:
        assert choose_next_in_set

    if argmax_selection:
        assert choose_next_in_set, "We still need to implement the argmax selection."

    if basemodel_ei_input:
        assert (
            choose_next_in_set
        ), "basemodel_ei_input only works with choose_next_in_set"
        assert (
            basemodel_for_ei is not None
        ), "basemodel_for_ei must be provided when basemodel_ei_input=True"

    # Sample which features are binary for this batch (consistent across all steps)
    if binary_feature_likelihood > 0.0:
        binary_features_mask = (
            torch.rand(batch_size, current_num_features, device=device)
            < binary_feature_likelihood
        )
    else:
        binary_features_mask = None

    super_batch_size = batch_size * sub_batch_size

    ys = None
    target_ys = None
    predictions = None
    options = []
    chosen_options = []
    choice_probs = []
    basemodel_ei_values = []
    step_entropies = []
    step_max_probs = []
    step_sampled_probs = []

    for step_idx in range(seq_len):
        # Determine effective batch size for this step
        # During joint_steps, we only sample batch_size trajectories (one per batch)
        # After joint_steps, we sample super_batch_size trajectories (one per sub_batch)
        is_joint_step = step_idx < joint_steps
        effective_batch_size = batch_size if is_joint_step else super_batch_size

        # At the transition from joint to split steps, expand trajectories
        if step_idx == joint_steps and joint_steps > 0:
            # Expand from [batch_size, seq] to [super_batch_size, seq]
            # by repeating each trajectory sub_batch_size times
            ys = (
                ys.unsqueeze(1)
                .expand(-1, sub_batch_size, -1)
                .reshape(super_batch_size, joint_steps)
            )
            target_ys = (
                target_ys.unsqueeze(1)
                .expand(-1, sub_batch_size, -1)
                .reshape(super_batch_size, joint_steps)
            )
            predictions = (
                predictions.unsqueeze(1)
                .expand(-1, sub_batch_size, -1, -1)
                .reshape(super_batch_size, joint_steps, current_num_features)
            )
            # Expand options and chosen_options for PPO training
            for i in range(len(options)):
                options[i] = (
                    options[i]
                    .unsqueeze(1)
                    .expand(-1, sub_batch_size, -1, -1)
                    .reshape(super_batch_size, -1, current_num_features)
                )
                chosen_options[i] = (
                    chosen_options[i]
                    .unsqueeze(1)
                    .expand(-1, sub_batch_size)
                    .reshape(super_batch_size)
                )
                choice_probs[i] = (
                    choice_probs[i]
                    .unsqueeze(1)
                    .expand(-1, sub_batch_size, -1)
                    .reshape(super_batch_size, -1)
                )

        if choose_next_in_set:
            if ys is None:
                x = torch.zeros(
                    effective_batch_size, 0, current_num_features, device=device
                )
                y = torch.zeros(effective_batch_size, 0, 1, device=device)
            else:
                x, y = preprocess_train_x_and_y(
                    predictions,
                    ys[:, :, None],
                    standardize_y,
                    bo_batch_size=bo_batch_size,
                )

            # we make sure the rollouts within each sub-batch get the same choices in each step
            total_opts = round(choice_set_size * super_choice_set_factor)

            # Check if sampler restricts sampling points by checking for get_candidate_points
            if hasattr(sampler, "get_candidate_points"):
                # Validate incompatible options
                if around_train_point_share > 0.0:
                    raise ValueError(
                        "around_train_point_share > 0 is not supported with samplers that provide candidate points"
                    )
                if binary_features_mask is not None:
                    raise ValueError(
                        "binary_feature_likelihood > 0 is not supported with samplers that provide candidate points"
                    )

                # Use candidate points from the sampler
                candidate_points = sampler.get_candidate_points()
                # Build options from the candidate points for each batch element
                opts_list = []
                for batch_idx in range(batch_size):
                    candidates = candidate_points[batch_idx].to(device)
                    n_candidates = candidates.shape[0]
                    if n_candidates >= total_opts:
                        # Sample without replacement
                        perm = torch.randperm(n_candidates, device=device)[:total_opts]
                        opts_list.append(candidates[perm])
                    else:
                        # Sample with replacement if we don't have enough candidates
                        indices = torch.randint(
                            0, n_candidates, (total_opts,), device=device
                        )
                        opts_list.append(candidates[indices])
                opts = torch.stack(
                    opts_list, dim=0
                )  # [batch_size, total_opts, features]
            else:
                # Calculate how many options should be sampled around training points
                num_around_train = (
                    round(total_opts * around_train_point_share)
                    if around_train_point_share > 0.0 and predictions is not None
                    else 0
                )
                num_uniform = total_opts - num_around_train

                # Sample uniform options
                opts = torch.rand(
                    batch_size,
                    num_uniform,
                    current_num_features,
                    device=device,
                )

                # Sample options around training points if applicable
                if num_around_train > 0:
                    # Get unique train points per batch
                    if is_joint_step:
                        # During joint steps, predictions already has shape [batch_size, step_idx, num_features]
                        train_points = predictions
                    else:
                        # After joint steps, take every sub_batch_size-th trajectory
                        train_points = predictions[
                            ::sub_batch_size
                        ]  # [batch_size, step_idx, num_features]

                    # Use shared utility for sampling around training points
                    around_train_opts = sample_x_around_points(
                        batch_size=batch_size,
                        num_samples=num_around_train,
                        num_features=current_num_features,
                        centers=train_points,
                        std=around_train_point_std,
                        device=device,
                    )
                    # Concatenate with uniform options
                    opts = torch.cat([opts, around_train_opts], dim=1)

            # Apply binary feature restriction to options
            if binary_features_mask is not None:
                binary_mask_expanded = binary_features_mask.unsqueeze(1).expand(
                    -1, total_opts, -1
                )
                opts = torch.where(binary_mask_expanded, (opts > 0.5).float(), opts)

            if super_choice_set_factor < 1.0:
                raise NotImplementedError("Please use super_choice_set_factor >= 1.")
            if not is_joint_step:
                opts = (
                    opts.unsqueeze(1)
                    .repeat(
                        1, sub_batch_size, 1, 1
                    )  # using repeat instead of expand as we have to copy for the next line either way
                    .view(super_batch_size, total_opts, current_num_features)
                )

            autocast_ctx = (
                torch.amp.autocast(dtype=torch.float16, device_type="cuda")
                if mixed_precision
                else nullcontext()
            )
            # Compute base model EI and add as additional feature to options
            if basemodel_ei_input:
                with torch.no_grad():
                    # Get logits from the base model (unfinetuned)
                    basemodel_logits = basemodel_for_ei(x=x, y=y, test_x=opts)
                    # Compute EI from base model - using best_f=0.0 as in ei_selector
                    basemodel_ei = (
                        basemodel_for_ei.criterion.ei(basemodel_logits, best_f=0.0)
                        .detach()
                        .view(super_batch_size, total_opts, 1)
                    )  # shape: [batch, num_opts, 1]
                    # Create augmented options with EI as additional feature for model input
                    opts_with_ei = torch.cat([opts, basemodel_ei], dim=-1)
                    # Add 0s to train_x for the EI column
                    x_with_ei = torch.cat(
                        [
                            x,
                            0.5 + torch.zeros(x.shape[0], x.shape[1], 1, device=device),
                        ],
                        dim=-1,
                    )

                # Subselect y_style for joint steps (every sub_batch_size-th element)
                # y_style is already in expanded shape [super_batch_size, 1]
                current_y_style = None
                if y_style is not None:
                    if is_joint_step:
                        # Subselect from [super_batch_size, 1] to [batch_size, 1]
                        current_y_style = y_style[::sub_batch_size]
                    else:
                        current_y_style = y_style

                with autocast_ctx:
                    logits = transform_logits(
                        model,
                        model(
                            x=x_with_ei,
                            y=y,
                            test_x=opts_with_ei,
                            y_style=current_y_style,
                        ),
                        ei_selector,
                    )
            else:
                # Subselect y_style for joint steps (every sub_batch_size-th element)
                # y_style is already in expanded shape [super_batch_size, 1]
                current_y_style = None
                if y_style is not None:
                    if is_joint_step:
                        # Subselect from [super_batch_size, 1] to [batch_size, 1]
                        current_y_style = y_style[::sub_batch_size]
                    else:
                        current_y_style = y_style

                with autocast_ctx:
                    logits = transform_logits(
                        model,
                        model(x=x, y=y, test_x=opts, y_style=current_y_style),
                        ei_selector,
                    )
                basemodel_ei = None

            if super_choice_set_factor > 1.0:
                # Select a super set of options consisting of:
                # - the best top_share * total options by model score
                # - plus the first (1 - top_share) * total options among the remaining non-best ones
                k_best = max(1, round(top_share * choice_set_size))
                k_first_non_best = choice_set_size - k_best

                # Get indices of top-k according to logits
                topk_vals, topk_inds = torch.topk(logits.squeeze(-1), k=k_best, dim=1)

                # Build a mask for selected top-k to find non-best candidates
                top_mask = torch.zeros(
                    effective_batch_size, total_opts, dtype=torch.bool, device=device
                )
                top_mask[torch.arange(effective_batch_size).unsqueeze(1), topk_inds] = (
                    True
                )

                # Indices of non-best (complement of top-k)
                all_inds = (
                    torch.arange(total_opts, device=device)
                    .unsqueeze(0)
                    .expand(effective_batch_size, -1)
                )
                non_best_inds = all_inds[~top_mask].view(effective_batch_size, -1)

                # Take the first k_first_non_best from non-best in their original order
                if k_first_non_best > 0:
                    first_non_best_inds = non_best_inds[:, :k_first_non_best]
                    combined_inds = torch.cat([topk_inds, first_non_best_inds], dim=1)
                else:
                    combined_inds = topk_inds

                # Subselect options and logits to the combined set
                opts = opts[
                    torch.arange(effective_batch_size).unsqueeze(1), combined_inds
                ]
                logits = logits[
                    torch.arange(effective_batch_size).unsqueeze(1), combined_inds
                ]
                # Also subselect EI values if basemodel_ei_input is enabled
                if basemodel_ei_input:
                    basemodel_ei = basemodel_ei[
                        torch.arange(effective_batch_size).unsqueeze(1), combined_inds
                    ]

            logits = logits.squeeze(-1)
            if argmax_selection:
                sampled_inds = logits.argmax(1)
            else:
                sampled_inds = torch.distributions.Categorical(logits=logits).sample()

            # Compute entropy, max prob, and sampled prob for tensorboard logging
            # Entropy: -sum(p * log(p)), using clamp to avoid log(0)
            log_probs = logits.log_softmax(
                dim=-1
            )  # shape: [effective_batch_size, num_opts]
            probs = log_probs.exp()  # shape: [effective_batch_size, num_opts]
            entropy = -(probs * log_probs).sum(dim=-1)  # shape: [effective_batch_size]
            max_prob = probs.max(dim=-1).values  # shape: [effective_batch_size]
            sampled_prob = probs[
                torch.arange(effective_batch_size, device=device), sampled_inds
            ]  # shape: [effective_batch_size]

            step_entropies.append(entropy)
            step_max_probs.append(max_prob)
            step_sampled_probs.append(sampled_prob)

            pred = opts[torch.arange(effective_batch_size), sampled_inds]

            options.append(opts)
            chosen_options.append(sampled_inds)
            choice_probs.append(probs)
            # Store the EI values for reuse in training loop
            basemodel_ei_values.append(basemodel_ei)
        else:
            if ys is None:
                full_train_x = torch.zeros(effective_batch_size, 0, 2, device=device)
            else:
                train_x, train_y = preprocess_train_x_and_y(
                    predictions[:, :],
                    ys[:, :, None],
                    standardize_y,
                    bo_batch_size=bo_batch_size,
                )
                full_train_x = torch.cat((train_x, train_y), -1)

            full_test_x = torch.full(
                (effective_batch_size, 1, 2),
                torch.nan,
                device=device,
            )
            logits = model(x=full_train_x, y=None, test_x=full_test_x)[:, :, 0].squeeze(
                1
            )

            p_cdf = torch.rand(*logits.shape[:-1], device=device)
            pred = torch.stack(
                [
                    unwrap_model(model).criterion.icdf(logits[i, :], p)
                    for i, p in enumerate(p_cdf.tolist())
                ],
            ).clamp(0, 1)

        target_y, y = sampler(
            pred.view(
                batch_size, 1 if is_joint_step else sub_batch_size, current_num_features
            )
        )
        # Flatten the result back to effective_batch_size
        y = y.view(effective_batch_size)
        target_y = target_y.view(effective_batch_size)

        if ys is None:
            ys = y.view(effective_batch_size, 1)
            target_ys = target_y.view(effective_batch_size, 1)
            predictions = pred.view(effective_batch_size, 1, current_num_features)
        else:
            ys = torch.cat((ys, y.view(effective_batch_size, 1)), 1)
            target_ys = torch.cat(
                (target_ys, target_y.view(effective_batch_size, 1)), 1
            )
            predictions = torch.cat(
                (predictions, pred.view(effective_batch_size, 1, current_num_features)),
                1,
            )

    return PathGenerationResult(
        ys=ys,
        target_ys=target_ys,
        predictions=predictions,
        options=options,
        chosen_options=chosen_options,
        choice_probs=choice_probs,
        current_num_features=current_num_features,
        joint_steps=joint_steps,
        basemodel_ei_values=basemodel_ei_values,
        step_entropies=step_entropies,
        step_max_probs=step_max_probs,
        step_sampled_probs=step_sampled_probs,
        binary_features_mask=binary_features_mask,
        bo_batch_size=bo_batch_size,
        seq_len=seq_len,
        y_style=y_style,
    )


def compute_curriculum_schedule(
    num_batches: int,
    target_seq_len: int,
    mode: tp.Literal["equal", "exponential"],
    min_seq_len: int = 8,
) -> list[int]:
    """Compute the sequence length for each batch based on curriculum.

    The curriculum scales the rollout length with factors of 2 until reaching
    the target sequence length.

    Args:
        num_batches: Total number of batches to train
        target_seq_len: Final sequence length to reach
        mode: Curriculum mode:
            - "equal": Train equal number of batches at each curriculum stage
            - "exponential": Train exponentially fewer batches at longer sequence
                lengths (double seq_len, half the batches at each stage)
        min_seq_len: Minimum sequence length for curriculum stages (default: 8)

    Returns:
        List of sequence lengths, one per batch
    """
    # Clamp min_seq_len to not exceed target_seq_len
    min_seq_len = min(min_seq_len, target_seq_len)

    # Generate stages: min_seq_len, min_seq_len*2, min_seq_len*4, ..., up to target_seq_len
    curriculum_stages: list[int] = []
    current = min_seq_len
    while current < target_seq_len:
        curriculum_stages.append(current)
        current *= 2
    curriculum_stages.append(target_seq_len)  # Always end with target

    n_stages = len(curriculum_stages)

    # If we have fewer batches than stages, trim stages to only include
    # the last num_batches stages (prioritize longer sequences)
    if num_batches < n_stages:
        curriculum_stages = curriculum_stages[-num_batches:]
        n_stages = num_batches

    if mode == "equal":
        # Equal batches per stage
        base_batches = num_batches // n_stages
        remainder = num_batches % n_stages
        batches_per_stage = [base_batches] * n_stages
        # Distribute remainder to later stages (longer sequences get slightly more)
        for i in range(remainder):
            batches_per_stage[-(i + 1)] += 1
    elif mode == "exponential":
        # Exponentially decreasing batches for longer sequences
        # First stage (shortest seq) gets the most batches, halving each time
        # Sum of geometric series: 1 + 1/2 + 1/4 + ... + 1/2^(n-1) = 2 - 2^(1-n)
        geometric_sum = sum(1 / (2**i) for i in range(n_stages))
        first_stage_batches = num_batches / geometric_sum
        batches_per_stage = [
            max(1, int(round(first_stage_batches / (2**i)))) for i in range(n_stages)
        ]
        # Adjust total to exactly match num_batches
        total = sum(batches_per_stage)
        diff = num_batches - total
        if diff != 0:
            # Distribute the difference across stages, starting from first stage
            # to maintain the exponential property as much as possible
            for i in range(abs(diff)):
                idx = i % n_stages
                if diff > 0:
                    batches_per_stage[idx] += 1
                else:
                    # Only subtract if stage has more than 1 batch
                    if batches_per_stage[idx] > 1:
                        batches_per_stage[idx] -= 1
                    else:
                        # Find another stage to subtract from
                        for j in range(n_stages):
                            if batches_per_stage[j] > 1:
                                batches_per_stage[j] -= 1
                                break
    else:
        raise ValueError(f"Unknown curriculum mode: {mode}")

    # Build the schedule: list of seq_len for each batch
    schedule: list[int] = []
    for stage_idx, stage_len in enumerate(curriculum_stages):
        schedule.extend([stage_len] * batches_per_stage[stage_idx])

    return schedule


def run_rl_training(
    rl_config: RLConfig,
    device_override: str | None = None,
) -> dict:
    # Detect if we're in distributed mode
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    # Get local rank from environment variable (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if distributed else 0
    torch.cuda.set_device(local_rank)

    print("Training distributed? ", distributed)
    print("Rank?", rank, "Local?", local_rank)
    print("World?", world_size)

    is_main = rank == 0

    if is_main and distributed:
        print(
            f"Running in distributed mode: rank {rank}/{world_size}, local_rank {local_rank}"
        )

    # Set seed (different per rank for diversity in trajectory generation)
    if rl_config.seed is not None:
        seed = rl_config.seed + (rank if distributed else 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    writer_path = rl_config.tensorboard_path

    # Only create writer on main process
    writer: SummaryWriter | None = None
    if writer_path is not None and is_main:
        writer = SummaryWriter(log_dir=writer_path)
        print(f"Tensorboard logging to: {writer_path}")

    if is_main:
        print(f"Loading base model from {rl_config.model_path}")
    base_train_config, model = load_config_and_model(
        rl_config.model_path, map_location="cpu"
    )

    if (
        rl_config.choose_next_in_set
        and not rl_config.ei_selector
        and not rl_config.keep_head
    ):
        # edit model head to be a simple 1 size prediction

        base_train_config = replace(
            base_train_config,
            model=replace(
                base_train_config.model, decoder_dict={"standard": (None, 1)}
            ),
        )
        model_with_single_output = base_train_config.model.create_model()
        og_statedict = model.state_dict()
        filtered_og_statedict = {}
        for n in og_statedict:
            if "decoder_dict" in n:
                print("removing", n)
            else:
                filtered_og_statedict[n] = og_statedict[n]
        model_with_single_output.load_state_dict(filtered_og_statedict, strict=False)
        model = model_with_single_output

    # Add y_style_encoder for seq_len encoding if enabled
    if rl_config.seq_len_y_style_encoder:
        if not rl_config.randomize_seq_len:
            print(
                "Warning: seq_len_y_style_encoder is True but randomize_seq_len is False. "
                "The y_style_encoder will always receive the same normalized seq_len value."
            )
        # Update the model config to include y_style_encoder
        # This ensures the encoder is saved with the model and can be loaded later
        base_train_config = replace(
            base_train_config,
            model=replace(
                base_train_config.model,
                y_style_encoder=StyleEncoderConfig(num_styles=1),
            ),
        )
        # Recreate model with the y_style_encoder
        model_with_y_style_encoder = base_train_config.model.create_model()
        og_statedict = model.state_dict()
        model_with_y_style_encoder.load_state_dict(og_statedict, strict=False)
        model = model_with_y_style_encoder
        print(f"Added y_style_encoder: Linear(1, {model.ninp}) for seq_len encoding")

    # We do this s.t. we can load the model w/o base_train_config
    rl_config = replace(rl_config, model=base_train_config.model)

    sampler_factory = rl_config.function_sampler.function_sampler

    try:
        # Setup device based on distributed training config
        # TODO test which of this is right for distributed, might want to use "cuda" on all.
        if device_override:
            device = device_override
        elif distributed:
            device = f"cuda:{local_rank}"
        elif rl_config.device is not None:
            device = rl_config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if is_main:
            print(f"Using device {device}")

        model.to(device)
        model.train()

        # Create old_model for PPO-style training
        # For DDP, we need to access the underlying module
        old_model = copy.deepcopy(model)
        old_model.requires_grad_(False)
        old_model.to(device)
        old_model.eval()

        # Store a separate copy of the basemodel for EI computation (never updated during training)
        basemodel_for_ei = None
        if rl_config.basemodel_ei_input:
            basemodel_for_ei = copy.deepcopy(model)
            basemodel_for_ei.requires_grad_(False)
            basemodel_for_ei.to(device)
            basemodel_for_ei.eval()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=rl_config.learning_rate,
            eps=rl_config.opt_eps,
            betas=(0.9, rl_config.opt_beta2),
            weight_decay=rl_config.weight_decay,
        )
        # Set up learning rate scheduler with optional warmup
        warmup_batches = rl_config.warmup_batches or 0
        main_batches = max(1, rl_config.num_batches - warmup_batches)
        eta_min = (
            rl_config.learning_rate
            if rl_config.min_learning_rate is None
            else rl_config.min_learning_rate
        )

        # Create main scheduler based on lr_schedule config
        if rl_config.lr_schedule == "linear":
            # Linear decay from learning_rate to eta_min
            main_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=eta_min / rl_config.learning_rate
                if rl_config.learning_rate > 0
                else 1.0,
                total_iters=main_batches,
            )
        else:
            # Default: Cosine annealing
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=main_batches,
                eta_min=eta_min,
            )

        if warmup_batches > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8 / rl_config.learning_rate,  # Start from near-zero
                end_factor=1.0,
                total_iters=warmup_batches,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_batches],
            )
            if is_main:
                print(f"Using linear warmup for {warmup_batches} batches")
        else:
            scheduler = main_scheduler

        if is_main:
            print(f"Using {rl_config.lr_schedule} LR schedule")
        scaler = torch.amp.GradScaler(
            enabled=True,
        )

        # Load checkpoint if available
        start_batch = 0
        if should_load_rl_checkpoint(rl_config):
            start_batch = load_rl_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                checkpoint_load_path=rl_config.checkpoint_load_path,
                device=device,
            )
            if is_main:
                print(f"Resuming training from batch {start_batch}")
        else:
            if rl_config.checkpoint_load_path is not None and is_main:
                print(
                    f"Checkpoint file {rl_config.checkpoint_load_path} not found or load/save paths are identical and file doesn't exist. Starting from scratch."
                )

        # Wrap model with DDP if using distributed training
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            if is_main:
                print("Wrapped model with DistributedDataParallel")
                if writer is not None:
                    writer.add_scalar("config/num_gpus", world_size)

        eps = rl_config.eps
        target_seq_len = rl_config.seq_len
        batch_size = rl_config.batch_size
        sub_batch_size = rl_config.sub_batch_size
        super_batch_size = batch_size * sub_batch_size

        # Compute curriculum schedule if enabled
        if rl_config.seq_len_curriculum is not None:
            curriculum_schedule = compute_curriculum_schedule(
                rl_config.num_batches,
                target_seq_len,
                rl_config.seq_len_curriculum,
                min_seq_len=rl_config.seq_len_curriculum_min,
            )
            if is_main:
                # Log curriculum stages
                stages = []
                current_len = curriculum_schedule[0]
                stage_start = 0
                for i, length in enumerate(curriculum_schedule):
                    if length != current_len:
                        stages.append(
                            (current_len, stage_start, i - 1, i - stage_start)
                        )
                        current_len = length
                        stage_start = i
                stages.append(
                    (
                        current_len,
                        stage_start,
                        len(curriculum_schedule) - 1,
                        len(curriculum_schedule) - stage_start,
                    )
                )
                print(
                    f"Curriculum schedule ({rl_config.seq_len_curriculum} mode): "
                    f"{len(stages)} stages"
                )
                for seq_len_stage, start, end, num_batches_stage in stages:
                    print(
                        f"  seq_len={seq_len_stage}: batches {start}-{end} ({num_batches_stage} batches)"
                    )
        else:
            curriculum_schedule = None

        batch_losses: list[float] = []

        for batch_i in range(start_batch, rl_config.num_batches):
            # Determine seq_len for this batch (curriculum or fixed)
            if curriculum_schedule is not None:
                seq_len = curriculum_schedule[batch_i]
            else:
                seq_len = target_seq_len
            if rl_config.seed is not None:
                # We re-initialize the seed for every batch
                # As randomness might be different in the fitting
                # step.
                seed = rl_config.seed + batch_i + (rank if distributed else 0)
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

            # Save model/optimizer state before this batch for potential rollback
            batch_rolled_back = False
            if rl_config.rollback_loss_threshold is not None:
                model_state_before_batch = {
                    k: v.clone() for k, v in unwrap_model(model).state_dict().items()
                }
                optimizer_state_before_batch = copy.deepcopy(optimizer.state_dict())

            # Calculate current number of features for this batch: (batch_i % num_features) + 1
            if rl_config.mix_k_features_in_opt > 1:
                all_features = list(range(1, rl_config.num_features + 1))
                num_full_copies = rl_config.mix_k_features_in_opt // len(all_features)
                remainder = rl_config.mix_k_features_in_opt % len(all_features)
                # Take all features num_full_copies times
                current_num_features_list = all_features * num_full_copies
                # Sample the remainder
                if remainder > 0:
                    current_num_features_list += random.sample(all_features, remainder)

            else:
                current_num_features_list = [
                    ((batch_i + 1) % rl_config.num_features) + 1
                ]

            start_generation_time = time.time()

            gen_results: list[PathGenerationResult] = []
            for feat_idx, current_num_features in enumerate(current_num_features_list):
                # Sample seq_len for this rollout if randomization is enabled
                # Use synchronized RNG to ensure same seq_len across distributed workers
                if rl_config.randomize_seq_len and seq_len > 3:
                    seq_len_rng = random.Random(
                        (rl_config.seed or 0) + batch_i * 10_000 + feat_idx
                    )
                    current_seq_len = seq_len_rng.randint(3, seq_len)
                else:
                    current_seq_len = seq_len

                # Compute normalized seq_len for y_style_encoder: (curr/max)*2-1 to be in [-1, 1]
                # Create y_style in expanded shape [super_batch_size, 1] upfront
                # and subselect for joint steps in generate_paths
                if rl_config.seq_len_y_style_encoder:
                    normalized_seq_len = (current_seq_len / seq_len) * 2 - 1
                    y_style = torch.full(
                        (batch_size * sub_batch_size, 1),
                        normalized_seq_len,
                        device=device,
                    )  # [super_batch_size, 1]
                else:
                    y_style = None

                # Sample bo_batch_size for this rollout if randomization is enabled
                if rl_config.randomize_bo_batch_size and rl_config.bo_batch_size > 1:
                    current_bo_batch_size = random.randint(1, rl_config.bo_batch_size)
                else:
                    current_bo_batch_size = rl_config.bo_batch_size

                # Compute a synchronized seed for sampler feature selection
                # This ensures all distributed workers use the same dimensionality
                sampler_seed = (
                    (rl_config.seed or 0) + batch_i * 1_000 + current_num_features
                )
                print(
                    f"{sampler_seed=}, {current_num_features=}, {current_bo_batch_size=}, {current_seq_len=}"
                )

                sampler = sampler_factory(
                    batch_size,
                    num_features=current_num_features,
                    device=device,
                    seed=sampler_seed,
                )
                if rl_config.independent_noise_in_function_draws:
                    sampler = partial(sampler, independent_noise=True)

                # Check if the sampler provides its own num_features
                if hasattr(sampler, "num_features"):
                    current_num_features = sampler.num_features

                # Sample joint_steps for joint rollout training
                # Use current_seq_len (not seq_len) to ensure joint_steps < current_seq_len
                joint_steps = 0
                if rl_config.joint_rollout_training is not None:
                    # Sample a random position to train on (0 to current_seq_len-1)
                    # joint_steps = training position, so trajectories are identical until then
                    if rl_config.joint_rollout_training == "remaining" and distributed:
                        # For "remaining" mode in DDP, use a synchronized seed across all workers
                        # to ensure the same joint_steps value, enabling proper gradient synchronization
                        sync_rng = random.Random(
                            (rl_config.seed or 0) + batch_i * 10_000 + feat_idx + 1412
                        )
                        joint_steps = sync_rng.randint(0, current_seq_len - 1)
                    else:
                        joint_steps = random.randint(0, current_seq_len - 1)

                gen_res = generate_paths(
                    model=model,
                    sampler=sampler,
                    batch_size=batch_size,
                    seq_len=current_seq_len,
                    sub_batch_size=sub_batch_size,
                    choose_next_in_set=rl_config.choose_next_in_set,
                    choice_set_size=rl_config.choice_set_size,
                    ei_selector=rl_config.ei_selector,
                    standardize_y=rl_config.standardize_y,
                    current_num_features=current_num_features,
                    device=device,
                    super_choice_set_factor=rl_config.super_choice_set_factor,
                    top_share=rl_config.choice_set_top_share,
                    joint_steps=joint_steps,
                    basemodel_ei_input=rl_config.basemodel_ei_input,
                    basemodel_for_ei=basemodel_for_ei,
                    mixed_precision=rl_config.mixed_precision_path_generation,
                    bo_batch_size=current_bo_batch_size,
                    binary_feature_likelihood=rl_config.binary_feature_likelihood,
                    around_train_point_share=rl_config.around_train_point_share,
                    around_train_point_std=rl_config.around_train_point_std,
                    y_style=y_style,
                )

                generation_time = time.time() - start_generation_time

                start_draw_time = time.time()

                draw_size = 100_000
                draw_x = torch.rand(
                    batch_size, draw_size, current_num_features, device=device
                )
                # Apply binary feature constraint to draw samples if applicable
                if gen_res.binary_features_mask is not None:
                    binary_mask_expanded = gen_res.binary_features_mask.unsqueeze(
                        1
                    ).expand(-1, draw_size, -1)
                    draw_x = torch.where(
                        binary_mask_expanded, (draw_x > 0.5).float(), draw_x
                    )
                draw, _ = sampler(
                    draw_x,
                    independent_noise=True,
                )

                def compute_quantiles(
                    y_version, use_draw: bool = True, draw=draw, draw_size=draw_size
                ):
                    y_view = y_version.view(batch_size, -1)
                    if use_draw:
                        # Use draw samples to compute quantile positions
                        sorted_ref = draw.view(batch_size, draw_size).sort(1).values
                        ref_size = draw_size
                    else:
                        # Use batch data itself to compute quantile positions
                        sorted_ref = y_view.sort(1).values
                        ref_size = y_view.shape[1]

                    quantiles = (
                        torch.searchsorted(sorted_ref, y_view).float() / ref_size
                    )

                    return quantiles.view(batch_size * sub_batch_size, -1)

                def compute_standardized_ys(
                    y_version, use_draw: bool = False, draw=draw, draw_size=draw_size
                ):
                    # Standardize per function (per batch) - z-score normalization
                    if use_draw:
                        # Use draw samples to compute mean and std
                        draw_view = draw.view(batch_size, draw_size)
                        mean = draw_view.mean(dim=1, keepdim=True)
                        std = draw_view.std(dim=1, keepdim=True)
                    else:
                        # Use batch data to compute mean and std
                        y_view = y_version.view(batch_size, -1)
                        mean = y_view.mean(dim=1, keepdim=True)
                        std = y_view.std(dim=1, keepdim=True)
                    std = std.clamp(min=1e-8)
                    y_view = y_version.view(batch_size, -1)
                    standardized = (y_view - mean) / std
                    return standardized.view(batch_size * sub_batch_size, -1)

                use_draw_for_reward = rl_config.reward.standardization_source == "draw"

                # Compute quantiles and standardized values for REWARDS
                # (respects standardization_source setting)
                reward_y_quantiles = compute_quantiles(
                    gen_res.ys, use_draw=use_draw_for_reward
                )
                reward_target_y_quantiles = compute_quantiles(
                    gen_res.target_ys, use_draw=use_draw_for_reward
                )
                reward_standardized_ys = compute_standardized_ys(
                    gen_res.ys, use_draw=use_draw_for_reward
                )
                reward_standardized_target_ys = compute_standardized_ys(
                    gen_res.target_ys, use_draw=use_draw_for_reward
                )

                # Compute quantiles and standardized values for PLOTS
                # (always uses draw for quantiles, batch for standardized)
                plot_y_quantiles = compute_quantiles(gen_res.ys, use_draw=True)
                plot_target_y_quantiles = compute_quantiles(
                    gen_res.target_ys, use_draw=True
                )
                plot_standardized_ys = compute_standardized_ys(
                    gen_res.ys, use_draw=True
                )
                plot_standardized_target_ys = compute_standardized_ys(
                    gen_res.target_ys, use_draw=True
                )

                draw_time = time.time() - start_draw_time

                # These rewards make sense, but they do have the problem that randomly roll-out might just be disadvantaged by earlier mishaps in it
                # This could be fixed by keeping all trajectories the same until we break out into sub trajectories and then only train at the break-out point

                gen_res.normalized_avg_rewards, gen_res.unnormalized_avg_rewards = (
                    rl_config.reward.compute_reward(
                        gen_res.ys.view(batch_size, sub_batch_size, -1),
                        gen_res.target_ys.view(batch_size, sub_batch_size, -1),
                        reward_y_quantiles.view(batch_size, sub_batch_size, -1),
                        reward_target_y_quantiles.view(batch_size, sub_batch_size, -1),
                        reward_standardized_ys.view(batch_size, sub_batch_size, -1),
                        reward_standardized_target_ys.view(
                            batch_size, sub_batch_size, -1
                        ),
                    )
                )

                # For batched BO, copy reward from last position in each batch to all positions
                if gen_res.bo_batch_size > 1:
                    reward_seq_len = gen_res.normalized_avg_rewards.shape[-1]
                    normalized = gen_res.normalized_avg_rewards.clone()
                    unnormalized = gen_res.unnormalized_avg_rewards.clone()

                    for batch_start in range(0, reward_seq_len, gen_res.bo_batch_size):
                        batch_end = min(
                            batch_start + gen_res.bo_batch_size, reward_seq_len
                        )
                        # Copy reward from last position in batch to all positions in batch
                        normalized[..., batch_start:batch_end] = normalized[
                            ..., batch_end - 1 : batch_end
                        ]
                        unnormalized[..., batch_start:batch_end] = unnormalized[
                            ..., batch_end - 1 : batch_end
                        ]

                    gen_res.normalized_avg_rewards = normalized
                    gen_res.unnormalized_avg_rewards = unnormalized
                gen_res.draw = draw
                # Store plot versions (not reward versions) for tensorboard logging
                gen_res.y_quantiles = plot_y_quantiles
                gen_res.target_y_quantiles = plot_target_y_quantiles
                gen_res.standardized_ys = plot_standardized_ys
                gen_res.standardized_target_ys = plot_standardized_target_ys
                gen_res.draw_size = draw_size

                gen_results.append(gen_res)

            # TODO put all the relevant var's, that is for stuff without plotting just normalized_avg_rewards, I believe, into the gen_res and then generate multiple gen_res and cycle through them during training
            # when we do the features, we should enable it to just be a subset of the features
            # and when we run the trainings with this setting we should maybe think about reducing the batch size!?

            if writer is not None:
                # Log current curriculum seq_len
                writer.add_scalar("curriculum/seq_len", seq_len, batch_i)

                def avg(
                    values: list[float],
                ) -> float:
                    return sum(values) / len(values)

                # Aggregated metrics (average across all gen_results)
                avg_reward = avg(
                    [gr.unnormalized_avg_rewards.mean().item() for gr in gen_results]
                )
                writer.add_scalar(
                    "avg_reward",
                    avg_reward,
                    batch_i,
                )

                writer.add_scalar(
                    "reward_metrics/std_mean",
                    avg(
                        [
                            gr.unnormalized_avg_rewards.std(1, keepdim=True)
                            .mean()
                            .item()
                            for gr in gen_results
                        ]
                    ),
                    batch_i,
                )
                writer.add_scalar(
                    "reward_metrics/std_min",
                    min(
                        gr.unnormalized_avg_rewards.std(1, keepdim=True).min().item()
                        for gr in gen_results
                    ),
                    batch_i,
                )
                writer.add_scalar(
                    "retrieved_y/mean",
                    avg([gr.ys.mean().item() for gr in gen_results]),
                    batch_i,
                )
                writer.add_scalar(
                    "retrieved_y/last",
                    avg([gr.ys[:, -1].mean().item() for gr in gen_results]),
                    batch_i,
                )
                writer.add_scalar(
                    "retrieved_y/max",
                    avg([gr.ys.max(1).values.mean().item() for gr in gen_results]),
                    batch_i,
                )

                # Regret metrics
                for name in ["noisy", "noiseless"]:
                    regrets = []
                    for gr in gen_results:
                        max_draw_per_function = gr.draw.max(1).values
                        used_ys = gr.ys if name == "noisy" else gr.target_ys
                        regret = (
                            (
                                max_draw_per_function[:, None]
                                - used_ys.max(1).values.view(batch_size, sub_batch_size)
                            )
                            .mean()
                            .item()
                        )
                        regrets.append(regret)
                    writer.add_scalar(
                        f"retrieved_y/final_{name}_regret_v_{draw_size}rs",
                        avg(regrets),
                        batch_i,
                    )

                # Noiseless regret at step increments of 5
                step_increments = list(range(5, seq_len + 1, 5))
                for step_cutoff in step_increments:
                    regrets_at_step = []
                    for gr in gen_results:
                        max_draw_per_function = gr.draw.max(1).values
                        used_ys = gr.target_ys[:, :step_cutoff]
                        max_y_up_to_step = used_ys.max(1).values.view(
                            batch_size, sub_batch_size
                        )
                        regret = (
                            (max_draw_per_function[:, None] - max_y_up_to_step)
                            .mean()
                            .item()
                        )
                        regrets_at_step.append(regret)
                    writer.add_scalar(
                        f"retrieved_y/noiseless_regret_at_step_{step_cutoff}",
                        avg(regrets_at_step),
                        batch_i,
                    )

                # Distance to incumbent (Euclidean distance to best point seen so far)
                # Averaged over groups of 5 steps
                # Skip this logging when using randomize_seq_len as sequence lengths vary
                if not rl_config.randomize_seq_len:
                    for group_start in range(0, seq_len, 5):
                        group_end = min(group_start + 5, seq_len)
                        all_distances = []
                        for gr in gen_results:
                            for step_i in range(group_start, group_end):
                                if step_i == 0:
                                    # At step 0, there's no previous incumbent
                                    continue
                                # Incumbent is the argmax of target_ys among steps 0 to step_i-1
                                incumbent_indices = gr.target_ys[:, :step_i].argmax(
                                    dim=1
                                )
                                batch_indices = torch.arange(
                                    gr.predictions.shape[0],
                                    device=gr.predictions.device,
                                )
                                incumbent_x = gr.predictions[
                                    batch_indices, incumbent_indices
                                ]
                                current_x = gr.predictions[:, step_i]
                                # Euclidean distance
                                distance = (
                                    (current_x - incumbent_x).pow(2).sum(dim=-1).sqrt()
                                )
                                all_distances.append(distance.mean().item())

                        if all_distances:
                            avg_distance = sum(all_distances) / len(all_distances)
                            writer.add_scalar(
                                f"incumbent_distance/steps_{group_start + 1}_to_{group_end}",
                                avg_distance,
                                batch_i,
                            )

                # Quantile metrics
                writer.add_scalar(
                    "retrieved_y/mean_quantile",
                    avg([gr.target_y_quantiles.mean().item() for gr in gen_results]),
                    batch_i,
                )
                writer.add_scalar(
                    "retrieved_y/last_quantile",
                    avg(
                        [
                            gr.target_y_quantiles[:, -1].mean().item()
                            for gr in gen_results
                        ]
                    ),
                    batch_i,
                )
                max_quantile = avg(
                    [
                        gr.target_y_quantiles.max(1).values.mean().item()
                        for gr in gen_results
                    ]
                )
                writer.add_scalar(
                    "retrieved_y/max_quantile",
                    max_quantile,
                    batch_i,
                )

                print(
                    "average max quantile",
                    max_quantile,
                )
                print("avereage reward", avg_reward)
                print("generation took ", generation_time, "seconds")

                writer.add_scalar(
                    "retrieved_y/max_standardized",
                    avg(
                        [
                            gr.standardized_target_ys.max(1).values.mean().item()
                            for gr in gen_results
                        ]
                    ),
                    batch_i,
                )

                # Per-feature metrics
                for gr in gen_results:
                    num_feats = gr.current_num_features
                    local_draw_size = gr.draw_size
                    max_draw_per_function = gr.draw.max(1).values

                    writer.add_scalar(
                        f"{num_feats}_features/avg_reward",
                        gr.unnormalized_avg_rewards.mean().item(),
                        batch_i,
                    )
                    for name, used_ys in [
                        ("noisy", gr.ys),
                        ("noiseless", gr.target_ys),
                    ]:
                        writer.add_scalar(
                            f"{num_feats}_features/retrieved_y/final_{name}_regret_v_{local_draw_size}rs",
                            (
                                max_draw_per_function[:, None]
                                - used_ys.max(1).values.view(batch_size, sub_batch_size)
                            )
                            .mean()
                            .item(),
                            batch_i,
                        )
                    writer.add_scalar(
                        f"{num_feats}_features/retrieved_y/max_quantile",
                        gr.target_y_quantiles.max(1).values.mean().item(),
                        batch_i,
                    )

                # Rollout distribution metrics (entropy, max_prob, sampled_prob)
                # Averaged across all steps and all samples in the batch
                # Note: tensors may have different shapes (batch_size during joint steps,
                # super_batch_size after), so we concatenate and compute mean safely
                all_entropies = []
                all_max_probs = []
                all_sampled_probs = []
                for gr in gen_results:
                    if gr.step_entropies:
                        # Concatenate all step tensors (handles different shapes)
                        entropies = torch.cat([e.flatten() for e in gr.step_entropies])
                        max_probs = torch.cat([m.flatten() for m in gr.step_max_probs])
                        sampled_probs = torch.cat(
                            [s.flatten() for s in gr.step_sampled_probs]
                        )
                        # Mean across all samples
                        all_entropies.append(entropies.mean().item())
                        all_max_probs.append(max_probs.mean().item())
                        all_sampled_probs.append(sampled_probs.mean().item())

                if all_entropies:
                    writer.add_scalar(
                        "rollout/entropy",
                        sum(all_entropies) / len(all_entropies),
                        batch_i,
                    )
                    writer.add_scalar(
                        "rollout/max_prob",
                        sum(all_max_probs) / len(all_max_probs),
                        batch_i,
                    )
                    writer.add_scalar(
                        "rollout/sampled_prob",
                        sum(all_sampled_probs) / len(all_sampled_probs),
                        batch_i,
                    )

            # training actually starts here
            start_training_loop_time = time.time()

            old_model.load_state_dict(copy.deepcopy(unwrap_model(model).state_dict()))

            losses: list[float] = []
            nan_encountered = False

            # Tracking for eps clamping analysis - per repetition
            per_rep_stats: list[dict[str, int]] = []

            # Tracking for zero variance filtering
            zero_variance_filtered_samples = 0
            total_filtering_samples = 0

            # Build list of (gen_res_idx, seq_idx) pairs to iterate over
            if rl_config.joint_rollout_training == "single":
                # Only train on the first non-joint position (the split point)
                training_steps = [
                    (gr_idx, gr.joint_steps) for gr_idx, gr in enumerate(gen_results)
                ]
            elif rl_config.joint_rollout_training == "remaining":
                # Train on all positions from the split point onwards
                training_steps = [
                    (gr_idx, sep)
                    for gr_idx, gr in enumerate(gen_results)
                    for sep in range(gr.joint_steps, gr.predictions.shape[1])
                ]
            else:
                # No joint rollout training: train on all positions
                training_steps = [
                    (gr_idx, sep)
                    for gr_idx, gr in enumerate(gen_results)
                    for sep in range(gr.predictions.shape[1])
                ]

            for _rep_idx in range(rl_config.experience_repetitions):
                # Per-repetition tracking
                rep_total_samples = 0
                rep_ratio_above_eps = 0
                rep_ratio_below_eps = 0
                rep_clamp_active = 0
                random.shuffle(training_steps)

                for gr_idx, i in training_steps:
                    gen_res = gen_results[gr_idx]
                    current_num_features = gen_res.current_num_features

                    with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                        optimizer.zero_grad(set_to_none=True)
                        if i == 0:
                            train_x = gen_res.predictions[:, :i]
                            train_y = gen_res.ys[:, :i, None]
                            full_train_x = torch.cat((train_x, train_y), -1)
                        else:
                            train_x, train_y = preprocess_train_x_and_y(
                                gen_res.predictions[:, :i],
                                gen_res.ys[:, :i, None],
                                rl_config.standardize_y,
                                bo_batch_size=gen_res.bo_batch_size,
                            )
                            full_train_x = torch.cat((train_x, train_y), -1)
                        full_test_x = torch.full(
                            (batch_size * sub_batch_size, 1, current_num_features + 1),
                            torch.nan,
                            device=device,
                        )
                        if rl_config.choose_next_in_set:
                            # Prepare options with base model EI if enabled
                            if rl_config.basemodel_ei_input:
                                # Reuse the precomputed EI values from generation
                                basemodel_ei = gen_res.basemodel_ei_values[i]
                                # Create augmented options with EI
                                opts = torch.cat(
                                    [gen_res.options[i], basemodel_ei], dim=-1
                                )
                                # Add 0s to train_x for the EI column
                                train_x = torch.cat(
                                    [
                                        train_x,
                                        torch.zeros(
                                            train_x.shape[0],
                                            train_x.shape[1],
                                            1,
                                            device=device,
                                        )
                                        + 0.5,
                                    ],
                                    dim=-1,
                                )
                            else:
                                opts = gen_res.options[i]

                            # Use stored y_style from path generation (already in super_batch_size shape)
                            with torch.no_grad():
                                logits_old = transform_logits(
                                    old_model,
                                    old_model(
                                        x=train_x,
                                        y=train_y,
                                        test_x=opts,
                                        y_style=gen_res.y_style,
                                    ),
                                    rl_config.ei_selector,
                                )  # shape: superb x num options

                            logits_new = transform_logits(
                                model,
                                model(
                                    x=train_x,
                                    y=train_y,
                                    test_x=opts,
                                    y_style=gen_res.y_style,
                                ),
                                rl_config.ei_selector,
                            )  # shape: superb x num options

                            log_p_new = logits_new.log_softmax(1)[
                                torch.arange(super_batch_size),
                                gen_res.chosen_options[i],
                            ]
                            log_p_old = logits_old.log_softmax(1)[
                                torch.arange(super_batch_size),
                                gen_res.chosen_options[i],
                            ]
                            pred_ratio = (log_p_new - log_p_old).exp()

                        else:
                            logits_new = model(
                                x=full_train_x, y=None, test_x=full_test_x
                            )[:, :, 0].squeeze(1)
                            with torch.no_grad():
                                logits_old = (
                                    old_model(
                                        x=full_train_x, y=None, test_x=full_test_x
                                    )[:, :, 0]
                                    .squeeze(1)
                                    .detach()
                                )
                            # criterion computes the neg log likelihood, that is why the ratio is inverted
                            # Note: old_model is always unwrapped, but model might be DDP-wrapped
                            nll_old = old_model.criterion(
                                logits_old, gen_res.predictions[:, i]
                            )
                            nll_new = unwrap_model(model).criterion(
                                logits_new, gen_res.predictions[:, i]
                            )
                            log_p_new = -nll_new
                            pred_ratio = (nll_old - nll_new).exp()
                        normalized_avg_rewards = gen_res.normalized_avg_rewards.view(
                            batch_size * sub_batch_size, -1
                        )

                        # Filter low-magnitude functions per time step
                        # Compute average magnitude (L1 / sub_batch_size) per function
                        rewards_at_step = normalized_avg_rewards[:, i].view(
                            batch_size, sub_batch_size
                        )
                        avg_magnitude_per_func = rewards_at_step.abs().mean(
                            dim=1
                        )  # [batch_size]

                        # Zero out rewards for functions with low average magnitude
                        # and scale up remaining rewards to maintain gradient magnitude
                        if rl_config.filter_rewards_up_to_magnitude is not None:
                            low_magnitude_mask = (
                                avg_magnitude_per_func
                                <= rl_config.filter_rewards_up_to_magnitude
                            )  # [batch_size]
                            effective_batch_size = (
                                batch_size - low_magnitude_mask.sum().item()
                            )

                            # Track zero variance filtered samples
                            zero_variance_filtered_samples += (
                                low_magnitude_mask.sum().item()
                            )
                            total_filtering_samples += batch_size

                            # Expand mask to super_batch_size and zero out rewards
                            full_zero_mask = (
                                low_magnitude_mask.unsqueeze(1)
                                .expand(-1, sub_batch_size)
                                .reshape(-1)
                            )  # [batch_size * sub_batch_size]
                            # Create a copy of rewards and zero out low-magnitude functions
                            rewards_for_loss = normalized_avg_rewards[:, i].clone()
                            rewards_for_loss[full_zero_mask] = 0.0
                            # Scale up by batch_size / effective_batch_size to maintain gradient magnitude
                            if effective_batch_size > 0:
                                rewards_for_loss = (
                                    rewards_for_loss * batch_size / effective_batch_size
                                )
                        else:
                            rewards_for_loss = normalized_avg_rewards[:, i]

                        # Compute eps bounds (asymmetric if eps_low is specified)
                        eps_lower = (
                            rl_config.eps_low if rl_config.eps_low is not None else eps
                        )
                        eps_upper = eps

                        # Track eps clamping statistics for this repetition
                        with torch.no_grad():
                            n_samples = pred_ratio.numel()
                            rep_total_samples += n_samples
                            rep_ratio_above_eps += (
                                (pred_ratio > 1 + eps_upper).sum().item()
                            )
                            rep_ratio_below_eps += (
                                (pred_ratio < 1 - eps_lower).sum().item()
                            )
                            # Clamp is active when pred_ratio is outside bounds
                            rep_clamp_active += (
                                (
                                    (pred_ratio > 1 + eps_upper)
                                    | (pred_ratio < 1 - eps_lower)
                                )
                                .sum()
                                .item()
                            )

                        if rl_config.algorithm == "grpo":
                            # GRPO: min(ratio * advantage, clamp(ratio) * advantage)
                            goal = pred_ratio * rewards_for_loss
                            clamped_goal = (
                                pred_ratio.clamp(1 - eps_lower, 1 + eps_upper)
                                * rewards_for_loss
                            )
                            # GRPO formulation is a maximization and we minimize
                            loss = -torch.min(goal, clamped_goal)
                        else:
                            # CISPO: stop_grad(clamp(ratio)) * advantage * log_p_new
                            assert rl_config.algorithm == "cispo"
                            clamped_weight = pred_ratio.detach().clamp(
                                1 - eps_lower, 1 + eps_upper
                            )
                            # Maximize weighted log probability, so negate for minimization
                            loss = -clamped_weight * rewards_for_loss * log_p_new

                        loss = loss.mean()

                        scaler.scale(loss).backward()

                    losses.append(loss.detach().item())

                    if torch.isnan(loss):
                        if is_main:
                            print("nan loss")
                        nan_encountered = True
                        break

                    if (
                        rl_config.grad_clip_norm is not None
                        and rl_config.grad_clip_norm > 0
                    ):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), rl_config.grad_clip_norm
                        )

                    scaler.step(optimizer)
                    scaler.update()

                if nan_encountered:
                    break

                # Store this repetition's stats
                per_rep_stats.append(
                    {
                        "total_samples": rep_total_samples,
                        "ratio_above_eps": rep_ratio_above_eps,
                        "ratio_below_eps": rep_ratio_below_eps,
                        "clamp_active": rep_clamp_active,
                    }
                )

            training_loop_time = time.time() - start_training_loop_time
            mean_loss = sum(losses) / len(losses) if losses else float("nan")

            # Synchronize mean_loss across all workers for coordinated rollback decision
            if distributed:
                mean_loss_tensor = torch.tensor(mean_loss, device=device)
                dist.all_reduce(mean_loss_tensor, op=dist.ReduceOp.AVG)
                mean_loss = mean_loss_tensor.item()

            # Check if batch should be rolled back due to high loss
            if (
                rl_config.rollback_loss_threshold is not None
                and mean_loss > rl_config.rollback_loss_threshold
            ):
                # Rollback model and optimizer state to before this batch
                unwrap_model(model).load_state_dict(model_state_before_batch)
                optimizer.load_state_dict(optimizer_state_before_batch)
                batch_rolled_back = True
                if is_main:
                    print(
                        f"Batch {batch_i} rolled back: mean loss {mean_loss:.4f} > threshold {rl_config.rollback_loss_threshold}"
                    )

            batch_losses.append(mean_loss)
            if is_main:
                print("mean loss", mean_loss, "after batch", batch_i)

            if writer is not None and not math.isnan(mean_loss):
                print(f"rank {rank} is pushing times for batch {batch_i}")
                writer.add_scalar("loss/mean_batch_loss", mean_loss, batch_i)
                writer.add_scalar(
                    "optimizer/lr", optimizer.param_groups[0]["lr"], batch_i
                )
                writer.add_scalar("time/generation_time", generation_time, batch_i)
                writer.add_scalar("time/draw_time", draw_time, batch_i)
                writer.add_scalar(
                    "time/training_loop_time", training_loop_time, batch_i
                )

                # Log eps clamping statistics - per repetition
                for rep_idx, stats in enumerate(per_rep_stats):
                    if stats["total_samples"] > 0:
                        ratio_above_eps_frac = (
                            stats["ratio_above_eps"] / stats["total_samples"]
                        )
                        ratio_below_eps_frac = (
                            stats["ratio_below_eps"] / stats["total_samples"]
                        )
                        clamp_active_frac = (
                            stats["clamp_active"] / stats["total_samples"]
                        )

                        writer.add_scalar(
                            f"ppo_clipping/rep_{rep_idx}/ratio_above_eps_frac",
                            ratio_above_eps_frac,
                            batch_i,
                        )
                        writer.add_scalar(
                            f"ppo_clipping/rep_{rep_idx}/ratio_below_eps_frac",
                            ratio_below_eps_frac,
                            batch_i,
                        )
                        writer.add_scalar(
                            f"ppo_clipping/rep_{rep_idx}/clamp_active_frac",
                            clamp_active_frac,
                            batch_i,
                        )
                        writer.add_scalar(
                            f"ppo_clipping/rep_{rep_idx}/ratio_within_bounds_frac",
                            1.0 - clamp_active_frac,
                            batch_i,
                        )

                # Also log aggregated stats across all repetitions
                total_samples_all = sum(s["total_samples"] for s in per_rep_stats)
                if total_samples_all > 0:
                    total_above = sum(s["ratio_above_eps"] for s in per_rep_stats)
                    total_below = sum(s["ratio_below_eps"] for s in per_rep_stats)
                    total_clamp = sum(s["clamp_active"] for s in per_rep_stats)

                    writer.add_scalar(
                        "ppo_clipping/total/ratio_above_eps_frac",
                        total_above / total_samples_all,
                        batch_i,
                    )
                    writer.add_scalar(
                        "ppo_clipping/total/ratio_below_eps_frac",
                        total_below / total_samples_all,
                        batch_i,
                    )
                    writer.add_scalar(
                        "ppo_clipping/total/clamp_active_frac",
                        total_clamp / total_samples_all,
                        batch_i,
                    )

                if per_rep_stats:
                    print("  PPO clipping stats per repetition:")
                    for rep_idx, stats in enumerate(per_rep_stats):
                        if stats["total_samples"] > 0:
                            print(
                                f"    rep_{rep_idx}: "
                                f"above_eps={stats['ratio_above_eps'] / stats['total_samples']:.4f}, "
                                f"below_eps={stats['ratio_below_eps'] / stats['total_samples']:.4f}, "
                                f"clamp_active={stats['clamp_active'] / stats['total_samples']:.4f}"
                            )

                # Log whether this batch was rolled back (1) or not (0)
                writer.add_scalar(
                    "training/batch_rolled_back",
                    1 if batch_rolled_back else 0,
                    batch_i,
                )

            if is_main and total_filtering_samples > 0:
                zero_variance_filtered_share = (
                    zero_variance_filtered_samples / total_filtering_samples
                )
                writer.add_scalar(
                    "reward_metrics/zero_variance_filtered_share",
                    zero_variance_filtered_share,
                    batch_i,
                )

            scheduler.step()

            # Save checkpoint after each batch (only on main process)
            if rl_config.checkpoint_save_path is not None and is_main:
                _save_checkpoint(
                    model=unwrap_model(model),
                    base_train_config=base_train_config,
                    rl_config=rl_config,
                    output_path=rl_config.checkpoint_save_path,
                    batch_i=batch_i,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                )

        # Save final checkpoint only on main process (legacy path for backwards compatibility)

        return {
            "losses": batch_losses,
            "model": unwrap_model(model),
            "base_train_config": base_train_config,
        }

    finally:
        if writer is not None:
            writer.close()


def _save_checkpoint(
    model: torch.nn.Module,
    base_train_config: base_config.BaseConfig,
    rl_config: RLConfig,
    output_path: str,
    batch_i: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
):
    """Save a checkpoint to the specified path.

    Args:
        model: The model to save
        base_train_config: The base training config
        rl_config: The RL config
        output_path: Path to save the checkpoint
        batch_i: Current batch index (optional, for resumable checkpoints)
        optimizer: Optimizer state (optional, for resumable checkpoints)
        scaler: GradScaler state (optional, for resumable checkpoints)
        scheduler: LR scheduler state (optional, for resumable checkpoints)
    """
    checkpoint = {
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "base_train_config": base_train_config.to_dict(),
        "config": rl_config.to_dict(),
    }

    if batch_i is not None:
        checkpoint["batch_i"] = batch_i
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    local_save(checkpoint, output_path)


def should_load_rl_checkpoint(
    rl_config: RLConfig,
    check_path_exists_function: tp.Callable[[str], bool] | None = None,
) -> bool:
    """Check if we should load a checkpoint.

    Returns True if:
    - checkpoint_load_path is set AND
    - Either load_path != save_path, OR the file exists
    """
    if rl_config.checkpoint_load_path is None:
        return False

    if check_path_exists_function is None:
        check_path_exists_function = local_exists

    return (rl_config.checkpoint_save_path != rl_config.checkpoint_load_path) or (
        (rl_config.checkpoint_save_path == rl_config.checkpoint_load_path)
        and check_path_exists_function(rl_config.checkpoint_load_path)
    )


def load_rl_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    checkpoint_load_path: str,
    device: str,
    load_function: tp.Callable | None = None,
) -> int:
    """Load a checkpoint and restore training state.

    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into (or fast-forward)
        scaler: The GradScaler to load state into
        checkpoint_load_path: Path to load checkpoint from
        device: Device to map tensors to
        load_function: Custom load function (defaults to local_load)

    Returns:
        The batch index to resume from (checkpoint batch_i + 1)
    """
    print(f"Loading checkpoint from {checkpoint_load_path}")

    if load_function is None:
        load_function = local_load

    try:
        checkpoint = load_function(checkpoint_load_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            if "batch_i" in checkpoint:
                start_batch = checkpoint["batch_i"] + 1
                print(f"Resuming from batch {start_batch}")

                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                else:
                    for _ in range(start_batch):
                        scheduler.step()

                return start_batch
            else:
                print("Checkpoint does not contain batch index, starting from 0")
                return 0
        else:
            raise ValueError(
                f"Checkpoint does not contain 'model_state_dict'. Keys: {checkpoint.keys() if isinstance(checkpoint, dict) else type(checkpoint)}"
            )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise e
