import torch
from botorch.models.fully_bayesian import SaasPyroModel

from .path_stgp import sample_around_train_point, sample_clustered_x
from .prior import Batch


@torch.no_grad()
def get_batch(
    batch_size,
    seq_len,
    num_features,
    single_eval_pos,
    hyperparameters=None,
    n_targets_per_input=1,
    **kwargs,
):
    if hyperparameters is None:
        hyperparameters = {}

    sample_clustered_x_hp = hyperparameters.get("sample_clustered_x", None)

    # Sample x based on the specified method (same logic as path_stgp.py)
    if sample_clustered_x_hp is True or sample_clustered_x_hp == "clustered":
        x = sample_clustered_x(
            batch_size,
            seq_len,
            num_features,
            num_cluster_max=hyperparameters.get("num_cluster_max", 1),
            max_std=hyperparameters.get("max_std", 0.25),
        )
    elif sample_clustered_x_hp and sample_clustered_x_hp.startswith(
        "around_train_point_binp_"
    ):
        binary_prob = float(sample_clustered_x_hp.split("_")[-1])
        x = sample_around_train_point(
            batch_size,
            seq_len,
            num_features,
            single_eval_pos,
            binary_feature_likelihood=binary_prob,
        )
    elif sample_clustered_x_hp == "around_train_point":
        x = sample_around_train_point(
            batch_size,
            seq_len,
            num_features,
            single_eval_pos,
        )
    else:
        assert sample_clustered_x_hp is None, sample_clustered_x_hp
        x = torch.rand(batch_size, seq_len, num_features)

    # Sample each batch item separately using SAAS prior
    y_list = []
    noisy_y_list = []

    for i in range(batch_size):
        m = SaasPyroModel()
        m._prior_mode = True
        m.set_inputs(
            x[i].to(dtype=torch.float64),
            torch.zeros(seq_len, 1, dtype=torch.float64),
        )
        m.sample()

        # Get noiseless and noisy predictions
        noiseless_y = m.f_prior_sample.to(dtype=torch.float32)
        noisy_y = m.Y_prior_sample.to(dtype=torch.float32)

        y_list.append(noiseless_y)
        noisy_y_list.append(noisy_y)

    y = torch.stack(y_list, dim=0).view(batch_size, seq_len, 1)
    noisy_y = torch.stack(noisy_y_list, dim=0).view(batch_size, seq_len, 1)

    # Handle n_targets_per_input
    if hyperparameters.get("noisy_predictions", False):
        target_y = noisy_y.expand(-1, -1, n_targets_per_input)
    else:
        target_y = y.expand(-1, -1, n_targets_per_input)

    # Handle predict_advantage
    if hyperparameters.get("predict_advantage", False):
        target_y = (
            target_y[:, :, :]
            - target_y[:, :single_eval_pos, :]
            .max(dim=-1, keepdim=True)
            .values.max(dim=-2, keepdim=True)
            .values
        )

    # Set ys to nan in training set
    number_of_y_hidden = torch.randint(
        0, hyperparameters.get("max_num_hidden_y", 0) + 1, tuple()
    )
    noisy_y[:, single_eval_pos - number_of_y_hidden : single_eval_pos] = torch.nan

    return Batch(x=x, y=noisy_y, target_y=target_y)
