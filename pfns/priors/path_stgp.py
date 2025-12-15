from functools import partial
from math import log, sqrt

import gpytorch

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel

from gpytorch.module import _pyro_sample_from_prior
from gpytorch.priors import LogNormalPrior, NormalPrior
from torch import Size

from .path_trace_sampling import generate_trace
from .prior import Batch


def to_random_module_no_copy(module) -> gpytorch.Module:
    random_module_cls = type(
        "_Random" + module.__class__.__name__,
        (gpytorch.module.RandomModuleMixin, module.__class__),
        {},
    )
    module.__class__ = random_module_cls  # hack

    for mname, child in module.named_children():
        if isinstance(child, gpytorch.Module):
            setattr(module, mname, to_random_module_no_copy(child))
    return module


def sample_paths(batch_size, num_features, hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {}
    base_class = (
        RBFKernel if hyperparameters.get("use_rbf_kernel", True) else MaternKernel
    )
    lengthscale_prior = LogNormalPrior(
        loc=hyperparameters.get("lengthscale_loc_constant_add", sqrt(2))
        + log(num_features) * hyperparameters.get("lengthscale_loc_feature_mul", 0.5),
        scale=hyperparameters.get("lengthscale_scale", sqrt(3)),
    )
    base_kernel = base_class(
        ard_num_dims=num_features,
        batch_shape=torch.Size([batch_size]),
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=GreaterThan(
            2.5e-2, transform=None, initial_value=lengthscale_prior.mode
        ),
        # pyre-ignore[6] GPyTorch type is unnecessarily restrictive.
        active_dims=None,
    )

    model = SingleTaskGP(
        torch.zeros(batch_size, 1, num_features),
        torch.zeros(batch_size, 1, 1),
        covar_module=base_kernel,
        mean_module=gpytorch.means.ConstantMean(
            constant_prior=NormalPrior(loc=0.0, scale=hyperparameters["mean_width"] / 2)
        ),
    )
    model = to_random_module_no_copy(model)
    _pyro_sample_from_prior(module=model, memo=None, prefix="")

    init_paths = draw_kernel_feature_paths(model=model, sample_shape=Size((1,)))

    if (
        additive_cosine_per_dim_prob := hyperparameters.get(
            "additive_cosine_per_dim_prob", 0.0
        )
    ) > 0.0:
        additive_cosine_per_dim = (
            torch.rand(batch_size, num_features) < additive_cosine_per_dim_prob
        )
        lengthscale = torch.rand(batch_size, num_features) * 0.2 + 0.08
        gp_lengthscale = model.covar_module.lengthscale.view(batch_size, num_features)
        magnitude = (
            torch.randn(batch_size, num_features) / 10 / gp_lengthscale
        )  # very rough...
        offset = torch.rand(batch_size, num_features)

        def paths(x):  # [batch_size, n, num_features]
            y = init_paths(x)
            mask = additive_cosine_per_dim.unsqueeze(1).expand(-1, x.shape[1], -1)
            return y + torch.where(
                mask,
                (
                    magnitude.unsqueeze(1)
                    * torch.cos(
                        2
                        * torch.pi
                        * (x / lengthscale.unsqueeze(1) + offset.unsqueeze(1))
                    )
                ),
                0.0,
            ).sum(-1).unsqueeze(0) / sqrt(num_features)
    else:
        paths = init_paths

    # paths takes a tensor of shape [batch_size, n, num_features] and returns a tensor of shape [1, batch_size, n]
    return paths


# paths = draw_matheron_paths(model=model, sample_shape=Size((128,)))


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
    print(f"{order.shape=}")
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
    surrounding_share: float = 0.5,
    binary_feature_likelihood: float = 0.0,
):
    binary_features = (
        (torch.rand(batch_size, num_features) < binary_feature_likelihood)
        .unsqueeze(1)
        .expand(-1, single_eval_pos, -1)
    )
    train_x = torch.rand(batch_size, single_eval_pos, num_features)
    train_x_cutoffs = torch.rand(batch_size, single_eval_pos, num_features)
    train_x[binary_features] = (
        train_x[binary_features] > train_x_cutoffs[binary_features]
    ).float()

    num_test_points = seq_len - single_eval_pos
    num_surrounding = int(num_test_points * surrounding_share)

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


def add_noise(y, hyperparameters, no_noise):
    batch_size = y.shape[0]
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

    noise_variance[no_noise] = 0.0

    noisy_y = y + torch.randn_like(y) * noise_variance[:, None, None] ** (1 / 2)
    return noisy_y


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
            "noise_var_loc": -4.0,
            "noise_var_scale": 1.0,
        }

    no_noise_prob = hyperparameters.get("no_noise_prob", 0.0)
    no_noise = torch.rand(batch_size) < no_noise_prob

    assert hyperparameters.get("mean_dist", "normal") == "normal"

    if (dummy_dim_prob := hyperparameters.get("dummy_dim_prob", 0.0)) > 0.0:
        dummy_dims = torch.rand(num_features) < dummy_dim_prob
        if (~dummy_dims).sum().item() == 0:
            non_dummy_dim = torch.randint(0, num_features, tuple())
            dummy_dims[non_dummy_dim] = False
        inner_paths = sample_paths(
            batch_size, (~dummy_dims).int().sum().item(), hyperparameters
        )  # paths maps [batch_size, n, real features] to [batch_size, n, 1

        def paths(x):
            return inner_paths(x[:, :, ~dummy_dims])
    else:
        paths = sample_paths(
            batch_size, num_features, hyperparameters
        )  # paths maps [batch_size, n, num_features] to [batch_size, n, 1]

    sample_clustered_x_hp = hyperparameters.get("sample_clustered_x", None)

    if sample_clustered_x_hp == "trace":
        x, y = generate_trace(
            batch_size,
            paths,
            partial(add_noise, hyperparameters=hyperparameters, no_noise=no_noise),
            seq_len,
            single_eval_pos,
            bounds=[(0, 1)] * num_features,
        )

        y = y.view(batch_size, seq_len, 1)
        noisy_y = y

    else:
        if sample_clustered_x_hp is True or sample_clustered_x_hp == "clustered":
            x = sample_clustered_x(
                batch_size,
                seq_len,
                num_features,
                num_cluster_max=hyperparameters.get("num_cluster_max", 1),
                max_std=hyperparameters.get("max_std", 0.25),
            )
        elif sample_clustered_x_hp.startswith("around_train_point_binp_"):
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

        y = paths(x).squeeze(0)  # shape: (batch_size, seq_len)

        y = y.view(batch_size, seq_len, 1)

        noisy_y = add_noise(y, hyperparameters, no_noise)

    if hyperparameters["noisy_predictions"]:
        target_y = add_noise(
            y.expand(-1, -1, n_targets_per_input), hyperparameters, no_noise
        )
    else:
        target_y = y.expand(-1, -1, n_targets_per_input)

    if hyperparameters.get("train_normalized_y", False):
        if single_eval_pos <= 1:
            raise ValueError("train_normalized_y requires single_eval_pos > 1")

        train_mean = noisy_y[:, :single_eval_pos].mean(dim=1, keepdim=True)
        train_std = noisy_y[:, :single_eval_pos].std(dim=1, keepdim=True)
        noisy_y = (noisy_y - train_mean) / train_std
        target_y = (target_y - train_mean) / train_std

    if hyperparameters.get("predict_advantage", False):
        target_y = (
            target_y[:, :, :]
            - target_y[:, :single_eval_pos, :]
            .max(dim=-1, keepdim=True)
            .values.max(dim=-2, keepdim=True)
            .values
        )

    # set ys to nan in training set
    number_of_y_hidden = torch.randint(
        0, hyperparameters.get("max_num_hidden_y", 0) + 1, tuple()
    )
    noisy_y[:, single_eval_pos - number_of_y_hidden : single_eval_pos] = torch.nan

    return Batch(x=x, y=noisy_y, target_y=target_y)
