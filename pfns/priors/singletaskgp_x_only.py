from math import log, sqrt

import torch

from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel
from gpytorch.priors import LogNormalPrior

from pfns.priors.prior import Batch


def sample_clustered_x(batch_size, seq_len, num_features, pad_factor: int = 10):
    """
    This function samples a batch of inputs from normal distributions.
    Its outputs are all in [0,1], which is ensured by over-sampling (pad_factor)
    and then rejecting outside samples. In addition, we clamp the values to [0,1].
    """
    mean = torch.rand(batch_size, num_features)
    std = torch.rand(batch_size, num_features) / 4.0

    x = (
        torch.randn(batch_size, num_features, seq_len * pad_factor) * std[:, :, None]
        + mean[:, :, None]
    )
    sorting_x = (x >= 0.0) & (x <= 1.0)
    order = torch.argsort(sorting_x, dim=-1, stable=True, descending=True)
    x = x.gather(dim=-1, index=order[:, :, :seq_len])
    x = x.transpose(1, 2)
    x = x.clamp(0, 1)
    return x


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
        hyperparameters = {
            "lengthscale_loc_constant_add": sqrt(2),  # same as in fully
            "lengthscale_loc_feature_mul": 0.5,  # same as in fully
            "lengthscale_scale": sqrt(3),  # same as in fully
            "noise_var_loc": -4.0,  # different in fully bayesian
            "noise_var_scale": 1.0,  # different in fully bayesian
            "noise_var_dist": "lognormal",  # different in fully bayesian, where it is "gamma"
            "noise_var_concentration": 0.9,
            "noise_var_rate": 10.0,
            "mean_width": 2.0,
            "mean_dist": "uniform",
            "attsink_tokens": 0,
            "noisy_predictions": False,
            "sample_clustered_x": False,
            "train_normalized_y": False,
            "style_for_max_on_border_likelihood": False,
            "dummy_dim_prob": 0.0,
        }

    assert not hyperparameters.get(
        "noisy_predictions", False
    ), "noisy_predictions is not supported for x_only mode"

    if hyperparameters["sample_clustered_x"]:
        x_train = sample_clustered_x(batch_size, single_eval_pos, num_features)
        x_test = torch.rand(batch_size, seq_len - single_eval_pos, num_features)
        x = torch.cat([x_train, x_test], dim=1)
    else:
        x = torch.rand(batch_size, seq_len, num_features)

    mean_width = hyperparameters["mean_width"]
    if mean_width == 0:
        mean = torch.zeros(batch_size)
    else:
        mean_dist = hyperparameters.get("mean_dist", "uniform")
        if mean_dist == "uniform":
            min_mean, max_mean = -mean_width / 2, mean_width / 2
            mean = torch.rand(batch_size) * (max_mean - min_mean) + min_mean
        elif mean_dist == "normal":
            mean = torch.randn(batch_size) * mean_width / 2
        else:
            raise ValueError(f"Unknown mean distribution {mean_dist}")

    if (dummy_dim_prob := hyperparameters.get("dummy_dim_prob", 0.0)) > 0.0:
        num_important_features = 0
        while num_important_features == 0:
            dummy_dims_mask = torch.bernoulli(
                torch.full((num_features,), dummy_dim_prob)
            ).bool()
            used_dims_mask = ~dummy_dims_mask
            num_important_features = used_dims_mask.sum()
    else:
        used_dims_mask = torch.ones(num_features, dtype=torch.bool)
        num_important_features = num_features

    length_scales = LogNormalPrior(
        loc=hyperparameters["lengthscale_loc_constant_add"]
        + log(num_important_features) * hyperparameters["lengthscale_loc_feature_mul"],
        scale=hyperparameters["lengthscale_scale"],
    ).sample((batch_size, num_important_features))

    kernel_name = hyperparameters.get("kernel", "rbf")

    def get_covar(length_scales, x):
        if kernel_name == "rbf":
            covar_module = RBFKernel(
                batch_shape=torch.Size([batch_size]),
                ard_num_dims=length_scales.shape[1],
            )
        elif kernel_name == "matern_1.5":
            covar_module = MaternKernel(
                batch_shape=torch.Size([batch_size]),
                ard_num_dims=length_scales.shape[1],
                nu=1.5,
            )
        elif kernel_name == "matern_2.5":
            covar_module = MaternKernel(
                batch_shape=torch.Size([batch_size]),
                ard_num_dims=length_scales.shape[1],
                nu=2.5,
            )
        elif kernel_name == "linear":
            covar_module = LinearKernel(
                batch_shape=torch.Size([batch_size]),
                ard_num_dims=length_scales.shape[1],
            )
        else:
            raise ValueError(f"Unknown kernel {kernel_name}")
        if covar_module.has_lengthscale:
            covar_module._set_lengthscale(length_scales)
        covar = covar_module(x[..., used_dims_mask], x[..., used_dims_mask])
        return covar

    covar = get_covar(length_scales, x)
    d = MultivariateNormal(torch.ones_like(x[:, :, 0]) * mean[:, None], covar)
    y: torch.Tensor = d.sample()

    noise_var_dist = hyperparameters.get("noise_var_dist", "lognormal")
    if noise_var_dist == "lognormal":
        noise_variance: torch.Tensor = LogNormalPrior(
            loc=hyperparameters["noise_var_loc"],
            scale=hyperparameters["noise_var_scale"],
        ).sample((batch_size,))
    elif noise_var_dist == "gamma":
        noise_variance: torch.Tensor = (
            torch.distributions.Gamma(
                hyperparameters["noise_var_concentration"],
                hyperparameters["noise_var_rate"],
            ).sample((batch_size,))
            + 1e-4
        )
    else:
        raise ValueError(f"Unknown noise variance distribution {noise_var_dist}")

    noisy_y = y + torch.randn_like(y) * noise_variance[:, None] ** (1 / 2)
    noisy_y = noisy_y.view(batch_size, seq_len, 1)

    assert n_targets_per_input == 1, "n_targets_per_input must be 1 for x_only mode"
    target_y = y.view(batch_size, seq_len, 1).repeat(1, 1, n_targets_per_input)

    if hyperparameters.get("train_normalized_y", False):
        if single_eval_pos <= 1:
            raise ValueError("train_normalized_y requires single_eval_pos > 1")

        train_mean = noisy_y[:, :single_eval_pos].mean(dim=1, keepdim=True)
        train_std = noisy_y[:, :single_eval_pos].std(dim=1, keepdim=True)
        noisy_y = (noisy_y - train_mean) / train_std
        target_y = (target_y - train_mean[..., None]) / train_std[..., None]

    # we should hide the y from time to time in training but still incorporate it to compute EI
    # that is exactly what we need for batch EI, I believe
    # then we simply do EI and then condition on the point without passing y again
    number_of_y_hidden = torch.randint(
        0, hyperparameters.get("max_num_hidden_y", 0) + 1, tuple()
    )
    noisy_y[:, single_eval_pos - number_of_y_hidden : single_eval_pos, :] = torch.nan
    full_train_x = torch.cat([x, noisy_y], dim=2)[:, :single_eval_pos, :]

    # LETS GET TO THE TEST PART

    # ei values
    # target_y shape: batch_size, seq_len, 1
    if hyperparameters.get("predict_ei", True):
        target_y = (
            target_y[:, single_eval_pos:].squeeze(-1)
            - target_y[:, :single_eval_pos, :]
            .squeeze(-1)
            .max(dim=-1, keepdim=True)
            .values
        )
    else:
        target_y = target_y[:, single_eval_pos:].squeeze(-1)

    # ei shape: batch_size, test size

    full_test_x = torch.cat([x, noisy_y], dim=2)[:, single_eval_pos:, :]
    full_target = torch.cat([x[:, single_eval_pos:], target_y.unsqueeze(-1)], dim=2)

    # we need three partitions of test
    # 1. y shown, predict subset of features (rest shown)
    # 2. y hidden, all features shown, predict y
    # 3. y hidden, predict subset of features that maximize EI

    batch_size, test_size, num_features_plus_1 = full_target.shape
    # let's do case 3 first
    if hyperparameters.get("predict_maximizer", False):
        case_3_test_size = num_features  # all empty to full - 1
        num_1_and_2_test_points = test_size - case_3_test_size

        max_target_index = target_y.argmax(dim=-1)
        max_ei_features = x[
            torch.arange(batch_size), max_target_index + single_eval_pos, :
        ]
        max_ei_features_and_nan_for_y = torch.cat(
            [max_ei_features, torch.full((batch_size, 1), torch.nan)], dim=1
        )

        case_3_test_x = max_ei_features_and_nan_for_y.unsqueeze(1).repeat(
            1, case_3_test_size, 1
        )  # shape: batch_size, case_3_test_size, num_features + 1
        case_3_target = case_3_test_x.clone()
        for i in range(case_3_test_size):
            case_3_test_x[:, i, : i + 1] = torch.nan
            case_3_target[:, i, i + 1 :] = torch.nan

        # add case 3 to the tensors
        full_test_x[:, num_1_and_2_test_points:, :] = case_3_test_x
        full_target[:, num_1_and_2_test_points:, :] = case_3_target
    else:
        num_1_and_2_test_points = test_size

    num_1_test_points = round(
        hyperparameters.get("y_conditioned_share", 0.5) * num_1_and_2_test_points
    )
    num_2_test_points = num_1_and_2_test_points - num_1_test_points

    if num_1_test_points > 0:
        target_mask = torch.ones_like(
            full_target[:, :num_1_test_points, :], dtype=torch.bool
        )
        # show targets for case 1
        target_mask[:, :, -1] = False  # not target but shown

        # for features, sample uniformly 0 to all
        assert full_target.shape[2] == num_features + 1
        num_shown_features = torch.randint(
            0, num_features, (batch_size, num_1_test_points)
        )

        for i in range(batch_size):
            for j in range(num_1_test_points):
                shown_features_for_example = num_shown_features[i, j]
                shown_features = torch.randperm(num_features + 1)[
                    :shown_features_for_example
                ]
                target_mask[i, j, shown_features] = (
                    False  # not in the target_mask anymore
                )

        assert not target_mask[:, :, -1].any()  # always predict the y target

        # set inputs that are hidden to nan
        full_test_x[:, :num_1_test_points, :][target_mask] = torch.nan
        full_target[:, :num_1_test_points, :][~target_mask] = torch.nan

    # finally let's do case 2
    if num_2_test_points > 0:
        # y hidden, all features shown, predict ei
        full_test_x[:, num_1_test_points:num_1_and_2_test_points, -1] = torch.nan
        full_target[:, num_1_test_points:num_1_and_2_test_points, :-1] = torch.nan

    return Batch(
        x=full_train_x, test_x=full_test_x, target=full_target, y=None, target_y=None
    )
