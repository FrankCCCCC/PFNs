# pyre-strict

from dataclasses import fields

import torch
from pfns.priors.prior import Batch


def get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    single_eval_pos: int,
    get_batch: callable,
    hyperparameters: dict | None = None,
    n_targets_per_input: int = 1,
    **kwargs,
) -> Batch:
    """
    Wrapper function that converts traditional batch format to x-only format.

    This function takes a traditional get_batch function and converts its output
    from the format with separate x, y, target_y to the x-only format with
    x, test_x, target (and y=None, target_y=None).

    Args:
        batch_size: Number of sequences in the batch
        seq_len: Total sequence length (train + test)
        num_features: Number of input features
        single_eval_pos: Position where training ends and testing begins
        hyperparameters: Hyperparameter dictionary
        get_batch: The traditional get_batch function to wrap
        n_targets_per_input: Number of targets per input
        **kwargs: Additional arguments to pass to the wrapped get_batch function

    Returns:
        Batch in x-only format with x, test_x, target fields
    """
    assert n_targets_per_input == 1, "Only single target per input supported"
    # Call the traditional get_batch function
    traditional_batch = get_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        num_features=num_features,
        single_eval_pos=single_eval_pos,
        hyperparameters=hyperparameters,
        n_targets_per_input=n_targets_per_input,
        **kwargs,
    )

    # Extract traditional format components
    x_traditional = traditional_batch.x  # shape: (batch_size, seq_len, num_features)
    y_traditional = (
        traditional_batch.y
    )  # shape: (batch_size, seq_len, 1) or (batch_size, seq_len,)
    if len(y_traditional.shape) == 2:
        y_traditional = y_traditional.unsqueeze(-1)
    target_y_traditional = (
        traditional_batch.target_y
    )  # shape: (batch_size, seq_len, n_targets_per_input) or (batch_size, seq_len,)
    if len(target_y_traditional.shape) == 2:
        target_y_traditional = target_y_traditional.unsqueeze(-1)

    # Split into train and test portions
    x_train = x_traditional[
        :, :single_eval_pos, :
    ]  # shape: (batch_size, single_eval_pos, num_features)
    x_test = x_traditional[
        :, single_eval_pos:, :
    ]  # shape: (batch_size, seq_len - single_eval_pos, num_features)
    y_train = y_traditional[
        :, :single_eval_pos, :
    ]  # shape: (batch_size, single_eval_pos, 1)
    y_test_targets = target_y_traditional[
        :, single_eval_pos:, :
    ]  # shape: (batch_size, seq_len - single_eval_pos, n_targets_per_input)

    # Convert to x-only format
    # x: concatenate training inputs with training outputs
    x_with_y = torch.cat(
        [x_train, y_train], dim=2
    )  # shape: (batch_size, single_eval_pos, num_features + 1)

    # test_x: test inputs with NaN for y values (to be predicted)
    test_y_nan = torch.full(
        (batch_size, seq_len - single_eval_pos, 1),
        torch.nan,
        dtype=x_test.dtype,
        device=x_test.device,
    )
    test_x_with_nan_y = torch.cat(
        [x_test, test_y_nan], dim=2
    )  # shape: (batch_size, seq_len - single_eval_pos, num_features + 1)

    # target: test inputs with target y values
    target_with_y = torch.cat(
        [torch.full_like(x_test, torch.nan), y_test_targets], dim=2
    )  # shape: (batch_size, seq_len - single_eval_pos, num_features + 1)

    # Create the x-only format batch, taking over all entries from the original batch
    # except for the ones we need to change
    batch_dict = {
        field.name: getattr(traditional_batch, field.name)
        for field in fields(traditional_batch)
    }

    # Override the fields that need to change for x-only format
    batch_dict.update(
        {
            "x": x_with_y,
            "test_x": test_x_with_nan_y,
            "target": target_with_y,
            "y": None,
            "target_y": None,
        }
    )

    x_only_batch = Batch(**batch_dict)

    return x_only_batch
