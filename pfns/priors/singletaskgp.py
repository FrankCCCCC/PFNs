from math import log, sqrt

import torch

from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel
from gpytorch.priors import LogNormalPrior

from pfns.priors.prior import Batch
from torch import Tensor


def sample_clustered_x(
    batch_size,
    seq_len,
    num_features,
    pad_factor: int = 10,
    num_cluster_max: int = 1,
    max_std: float = 0.25,
):
    """
    This function samples a batch of inputs from normal distributions.
    Its outputs are all in [0,1], which is ensured by over-sampling (pad_factor)
    and then rejecting outside samples. In addition, we clamp the values to [0,1].
    """
    num_clusters = torch.randint(1, num_cluster_max + 1, tuple()).item()

    mean = torch.rand(batch_size, num_clusters, num_features)
    std = torch.rand(batch_size, num_clusters, num_features) * max_std

    # define mean and std for each position
    # to do that we randomly pick values from the num_clusters dimension
    mean = (
        mean[:, torch.randint(0, num_clusters, (seq_len,)), :]
        .transpose(1, 2)
        .repeat(1, 1, pad_factor)
    )
    std = (
        std[:, torch.randint(0, num_clusters, (seq_len,)), :]
        .transpose(1, 2)
        .repeat(1, 1, pad_factor)
    )

    x = torch.randn(batch_size, num_features, seq_len * pad_factor) * std + mean
    x = x.transpose(1, 2)
    sorting_x = ((x >= 0.0) & (x <= 1.0)).sum(dim=-1)
    order = torch.argsort(sorting_x, dim=-1, stable=True, descending=True)
    x = x.gather(
        dim=1, index=order[:, :seq_len].unsqueeze(-1).expand(-1, -1, num_features)
    )
    x = x.clamp(0, 1)
    return x


def sample_around_train_point(
    batch_size,
    seq_len,
    num_features,
    single_eval_pos,
    surrounding_std: float = 0.01,
    surrouding_share: float = 0.5,
):
    train_x = torch.rand(batch_size, single_eval_pos, num_features)

    num_test_points = seq_len - single_eval_pos
    num_surrounding = int(num_test_points * surrouding_share)

    normal_test_x = torch.rand(
        batch_size, num_test_points - num_surrounding, num_features
    )
    if single_eval_pos > 0:
        centers = torch.multinomial(
            torch.ones(single_eval_pos), num_surrounding, replacement=True
        )
        surrounding_test_x = (
            torch.randn(batch_size, num_surrounding, num_features) * surrounding_std
            + train_x[:, centers]
        )
    else:
        surrounding_test_x = torch.rand(batch_size, num_surrounding, num_features)
    x = torch.cat([train_x, normal_test_x, surrounding_test_x], dim=1)
    return x


# adapted from botorch to support batching
def inv_kumaraswamy_warp(
    X: Tensor, c0: Tensor, c1: Tensor, eps: float = 1e-8
) -> Tensor:
    """Map warped inputs through an inverse Kumaraswamy CDF.

    This takes warped inputs (X) and transforms those via an inverse
    Kumaraswamy CDF. This then unnormalizes the inputs using bounds of
    [eps, 1-eps]^d and ensures that the values are within [0, 1]^d.

    Args:
        X: A `b x n x d`-dim tensor of inputs.
        c0: A `b x d`-dim tensor of the concentration0 parameter for the
            Kumaraswamy distribution.
        c1: A `b x d`-dim tensor of the concentration1 parameter for the
            Kumaraswamy distribution.
        eps: A small value that is used to ensure inputs are not 0 or 1.

    Returns:
        A `batch_shape x n x d`-dim tensor of untransformed inputs.
    """
    X_range = 1 - 2 * eps
    # unnormalize from [eps, 1-eps] to [0,1]
    untf_X = (1 - (1 - X).pow(1 / c0.unsqueeze(1))).pow(1 / c1.unsqueeze(1))
    return ((untf_X - eps) / X_range).clamp(0.0, 1.0)


@torch.no_grad()
def get_batch(
    batch_size,
    seq_len,
    num_features,
    single_eval_pos,
    hyperparameters=None,
    n_targets_per_input=1,
    return_infos=False,
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
            "sample_strategy": "uniform",
            "train_normalized_y": False,
            "style_for_max_on_border_likelihood": False,
            "dummy_dim_prob": 0.0,
            # Oversampling factor: build a super dataset of this factor times seq_len
            "oversample_factor": 1.0,
            # Proportion of final dataset sampled from the top-share (by non-noisy y)
            "top_sampling_share": 0.0,
            # The fraction of the super dataset considered as the top-share
            "top_share_of_super": 0.1,
            # input warping
            "input_warping_prob": 0.0,
            "input_warping_c0_std": 0.75**0.5,
            "input_warping_c1_std": 0.75**0.5,
        }

    # Build dataset, possibly oversampled if using top sampling
    top_sampling_share = hyperparameters.get("top_sampling_share", 0.0)

    if top_sampling_share > 0.0:
        oversample_factor = hyperparameters.get("oversample_factor", 1.0)
        assert (
            oversample_factor > 1.0
        ), "oversample_factor must be > 1.0 when top_sampling_share > 0"
        super_seq_len = round(seq_len * oversample_factor)
    else:
        assert (
            hyperparameters.get("oversample_factor", 1.0) == 1.0
        ), "oversample_factor must be 1.0 when top_sampling_share is 0"
        super_seq_len = seq_len

    sample_clustered_x_hp = hyperparameters.get("sample_clustered_x", False)

    if sample_clustered_x_hp is True or sample_clustered_x_hp == "clustered":
        x_super = sample_clustered_x(
            batch_size,
            super_seq_len,
            num_features,
            num_cluster_max=hyperparameters.get("num_cluster_max", 1),
            max_std=hyperparameters.get("max_std", 0.25),
        )
    elif sample_clustered_x_hp == "around_train_point":
        x_super = sample_around_train_point(
            batch_size,
            super_seq_len,
            num_features,
            single_eval_pos,
        )
    else:
        x_super = torch.rand(batch_size, super_seq_len, num_features)

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

    style = None

    if hyperparameters.get("additive", False):
        num_features_in_group1 = torch.randint(0, num_features, tuple()).item()
        perm = torch.randperm(num_features)
        features_in_group1 = perm[:num_features_in_group1]
        features_in_group0 = perm[num_features_in_group1:]

        covar0 = get_covar(
            length_scales[:, features_in_group0], x_super[:, :, features_in_group0]
        )
        covar1 = get_covar(
            length_scales[:, features_in_group1], x_super[:, :, features_in_group1]
        )

        d0 = MultivariateNormal(
            torch.ones_like(x_super[:, :, 0]) * mean[:, None], covar0
        )
        d1 = MultivariateNormal(torch.zeros_like(x_super[:, :, 0]), covar1)
        y_super: torch.Tensor = d0.sample() + d1.sample()
        style = torch.zeros(batch_size, num_features, 1)
        style[:, features_in_group1, :] = 1.0
        style = (style * 2.0) - 1.0
    else:
        covar = get_covar(length_scales, x_super)
        d = MultivariateNormal(torch.ones_like(x_super[:, :, 0]) * mean[:, None], covar)
        y_super: torch.Tensor = d.sample()

    # Select final dataset of length seq_len, possibly from top share of larger super dataset
    device = x_super.device
    x = torch.empty(batch_size, seq_len, num_features, device=device)
    y = torch.empty(batch_size, seq_len, device=device)

    if top_sampling_share > 0.0:
        top_share_of_super = hyperparameters.get("top_share_of_super", 0.1)
        # Calculate sizes with bounds checking using max/min
        top_k_count = min(
            max(0, round(top_share_of_super * super_seq_len)), super_seq_len
        )
        n_top = min(
            max(0, round(top_sampling_share * seq_len)), min(top_k_count, seq_len)
        )
        n_rest = seq_len - n_top

        if top_k_count > 0:
            top_inds = torch.topk(y_super, k=top_k_count, largest=True).indices
        else:
            top_inds = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        for b in range(batch_size):
            # Get top indices by non-noisy y value
            top_idx_b = top_inds[b]

            chosen_indices = []
            if n_top > 0 and top_idx_b.numel() > 0:
                perm_top = torch.randperm(top_idx_b.numel(), device=device)
                chosen_top = top_idx_b[perm_top[:n_top]]
                chosen_indices.append(chosen_top)

            if n_rest > 0:
                all_idx = torch.arange(super_seq_len, device=device)
                if chosen_indices:
                    chosen_cat = torch.cat(chosen_indices)
                    mask = torch.ones(super_seq_len, dtype=torch.bool, device=device)
                    mask[chosen_cat] = False
                    remaining_idx = all_idx[mask]
                else:
                    remaining_idx = all_idx
                perm_rem = torch.randperm(remaining_idx.numel(), device=device)
                chosen_rest = remaining_idx[perm_rem[:n_rest]]
                chosen_indices.append(chosen_rest)

            # Combine and shuffle indices
            final_idx = torch.cat(chosen_indices)
            final_idx = final_idx[torch.randperm(final_idx.numel(), device=device)]

            x[b] = x_super[b, final_idx]
            y[b] = y_super[b, final_idx]
    else:
        # No oversampling, just use the dataset as is
        x = x_super
        y = y_super

    if hyperparameters.get("style_for_max_on_border_likelihood", False):
        max_i = y.max(1).indices  # (B,)
        max_x = x[torch.arange(len(max_i)), max_i]  # (B,F)
        mins = x.min(1).values  # (B,F)
        maxs = x.max(1).values  # (B,F)
        is_max_or_min_on_border = (max_x == mins) | (max_x == maxs)  # (B,F)

        sureness = torch.rand(batch_size, num_features)  # (B,F)
        correct_hint = torch.bernoulli(sureness).bool()  # (B,F)
        border_style = (
            correct_hint * is_max_or_min_on_border * sureness
            + ~correct_hint * is_max_or_min_on_border * (1 - sureness)
            + correct_hint * ~is_max_or_min_on_border * (1 - sureness)
            + ~correct_hint * ~is_max_or_min_on_border * sureness
        )[:, :, None]  # (B,F,1)
        # for b in [0.1 * i for i in range(1, 10)]:
        #     mask = (border_style.flatten() < b) & (border_style.flatten() >= b - 0.1)
        #     print(b, is_max_or_min_on_border.flatten()[mask].float().mean(), mask.sum())
        # border_style = torch.zeros(batch_size, num_features, 1)
        if style is None:
            style = border_style
        else:
            style = torch.cat([style, border_style], dim=-1)

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

    if hyperparameters["noisy_predictions"]:
        target_y = y.view(batch_size, seq_len, 1) + torch.randn(
            batch_size, seq_len, n_targets_per_input
        ) * torch.sqrt(noise_variance[:, None, None])
    else:
        target_y = y.view(batch_size, seq_len, 1).repeat(1, 1, n_targets_per_input)

    train_normalized_y = hyperparameters.get("train_normalized_y", False)

    if hyperparameters.get("predict_advantage", False):
        assert not train_normalized_y
        target_y = (
            target_y[:, :, :]
            - target_y[:, :single_eval_pos, :]
            .max(dim=-1, keepdim=True)
            .values.max(dim=-2, keepdim=True)
            .values
        )

    if train_normalized_y:
        if single_eval_pos <= 1:
            raise ValueError("train_normalized_y requires single_eval_pos > 1")

        train_mean = noisy_y[:, :single_eval_pos].mean(dim=1, keepdim=True)
        train_std = noisy_y[:, :single_eval_pos].std(dim=1, keepdim=True)
        noisy_y = (noisy_y - train_mean) / train_std
        target_y = (target_y - train_mean[..., None]) / train_std[..., None]

    # Apply input warping if specified
    input_warping_prob = hyperparameters.get("input_warping_prob", 0.0)
    if input_warping_prob > 0.0:
        c0_std = hyperparameters.get("input_warping_c0_std", 0.75**0.5)
        c1_std = hyperparameters.get("input_warping_c1_std", 0.75**0.5)

        # Sample c0 and c1 parameters from LogNormal distributions for each batch
        c0 = LogNormalPrior(loc=0.0, scale=c0_std).sample((batch_size, num_features))
        c1 = LogNormalPrior(loc=0.0, scale=c1_std).sample((batch_size, num_features))

        no_warping_mask = torch.rand(batch_size, num_features) > input_warping_prob
        c0[no_warping_mask] = 1.0
        c1[no_warping_mask] = 1.0

        # Apply inverse Kumaraswamy warping to inputs with per-batch parameters
        x = inv_kumaraswamy_warp(x, c0, c1)

    # set ys to nan in training set
    number_of_y_hidden = torch.randint(
        0, hyperparameters.get("max_num_hidden_y", 0) + 1, tuple()
    )
    noisy_y[:, single_eval_pos - number_of_y_hidden : single_eval_pos] = torch.nan

    infos = {
        "lengthscales": length_scales,
        "noise_variances": noise_variance,
        "means": mean,
        "num_important_features": num_important_features,
        "kernel": kernel_name,
    }

    b = Batch(x=x, y=noisy_y, target_y=target_y, style=style)
    if return_infos:
        return b, infos
    else:
        return b
