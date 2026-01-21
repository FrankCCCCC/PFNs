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
from .utils import sample_x_around_points


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


def _sample_paths_inner(batch_size, num_features, hyperparameters=None):
    """Internal function that samples paths for the given number of non-dummy features.

    Returns a function that takes [batch_size, n, num_features] and returns [1, batch_size, n].
    """
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
        torch.zeros(batch_size, 1, num_features, dtype=torch.double),
        torch.zeros(batch_size, 1, 1, dtype=torch.double),
        covar_module=base_kernel,
        mean_module=gpytorch.means.ConstantMean(
            constant_prior=NormalPrior(loc=0.0, scale=hyperparameters["mean_width"] / 2)
        ),
    ).to(dtype=torch.double)
    model = to_random_module_no_copy(model)
    _pyro_sample_from_prior(module=model, memo=None, prefix="")

    init_paths = draw_kernel_feature_paths(model=model, sample_shape=Size((1,))).to(
        dtype=torch.double
    )

    if (
        additive_cosine_per_dim_prob := hyperparameters.get(
            "additive_cosine_per_dim_prob", 0.0
        )
    ) > 0.0:
        additive_cosine_per_dim = (
            torch.rand(batch_size, num_features) < additive_cosine_per_dim_prob
        )
        lengthscale = (
            torch.rand(batch_size, num_features, dtype=torch.double) * 0.2 + 0.08
        )
        gp_lengthscale = model.covar_module.lengthscale.view(batch_size, num_features)
        magnitude = (
            torch.randn(batch_size, num_features, dtype=torch.double)
            / 10
            / gp_lengthscale
        )  # very rough...
        offset = torch.rand(batch_size, num_features, dtype=torch.double)

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
                        * (x / lengthscale.unsqueeze(1) + offset.unsqueeze(1)),
                        dtype=torch.double,
                    )
                ),
                0.0,
            ).sum(-1).unsqueeze(0) / sqrt(num_features)
    else:
        paths = init_paths

    def paths_with_double(x):
        dtype = x.dtype
        y = paths(x.double())
        return y.to(dtype)

    # paths takes a tensor of shape [batch_size, n, num_features] and returns a tensor of shape [1, batch_size, n]
    return paths_with_double


def _get_dummy_dims(num_features, hyperparameters):
    """Determine which dimensions are dummy. Returns (dummy_dims mask, num_non_dummy).

    Hyperparameters:
        dummy_dim_sample_non_dummy_range: tuple (min, max) - Sample number of non-dummy
            dimensions uniformly from [min, max]. Remaining dimensions are dummy.
        dummy_dim_sample_non_dummy_range_prob: float - Probability of applying the
            dummy_dim_sample_non_dummy_range logic. If not applied, all dimensions
            are non-dummy. Default: 1.0 (always apply when range is set).
        dummy_dim_prob: float - Each dimension has this probability of being a dummy.
            At least one dimension will always be non-dummy.
    """
    if (
        non_dummy_range := hyperparameters.get("dummy_dim_sample_non_dummy_range", None)
    ) is not None:
        # Check if we should apply the non-dummy range logic for this dataset
        range_prob = hyperparameters.get("dummy_dim_sample_non_dummy_range_prob", 1.0)
        if torch.rand(()).item() >= range_prob:
            # Don't apply dummy dims - all dimensions are non-dummy
            return None, num_features

        min_non_dummy, max_non_dummy = non_dummy_range
        num_non_dummy = min(
            torch.randint(min_non_dummy, max_non_dummy + 1, ()).item(),
            num_features,
        )
        perm = torch.randperm(num_features)
        dummy_dims = torch.ones(num_features, dtype=torch.bool)
        dummy_dims[perm[:num_non_dummy]] = False
        return dummy_dims, num_non_dummy

    elif (dummy_dim_prob := hyperparameters.get("dummy_dim_prob", 0.0)) > 0.0:
        dummy_dims = torch.rand(num_features) < dummy_dim_prob
        if (~dummy_dims).sum().item() == 0:
            dummy_dims[torch.randint(0, num_features, ())] = False
        return dummy_dims, (~dummy_dims).int().sum().item()

    return None, num_features


def sample_paths(batch_size, num_features, hyperparameters=None):
    """Sample GP paths with optional dummy dimension handling and gap discontinuities.

    This function handles dummy dimensions by sampling a GP on only the non-dummy
    dimensions and wrapping it in a function that filters out dummy dimensions.
    When gaps are enabled, all region functions share the same dummy dimensions.

    Hyperparameters:
        dummy_dim_sample_non_dummy_range: tuple (min, max) - Sample number of non-dummy
            dimensions uniformly from [min, max]. Remaining dimensions are dummy.
        dummy_dim_prob: float - Each dimension has this probability of being a dummy.
            At least one dimension will always be non-dummy.
        gap_max_splits: int - Max axis-aligned splits for gaps (default: 0)
        gap_prob: float - Probability of applying gaps (default: 1.0)
        gap_lengthscale_add: float - Add to lengthscale_loc_constant_add when
            gaps are applied (default: 0.0, no change)
        gap_lengthscale_add_prob: float - Probability of applying the
            lengthscale adjustment when gaps are applied (default: 1.0)

    Returns a function that takes [batch_size, n, num_features] and returns [1, batch_size, n].
    """
    if hyperparameters is None:
        hyperparameters = {}

    # Determine if gaps will be applied
    max_splits = hyperparameters.get("gap_max_splits", 0)
    gap_prob = hyperparameters.get("gap_prob", 1.0)
    apply_gaps = max_splits > 0 and torch.rand(()).item() < gap_prob
    num_splits = torch.randint(1, max_splits + 1, (1,)).item() if apply_gaps else 0

    # Optionally adjust lengthscale when gaps are applied
    if num_splits > 0:
        lf = hyperparameters.get("gap_lengthscale_add", 0.0)
        lf_prob = hyperparameters.get("gap_lengthscale_add_prob", 1.0)
        if lf != 0.0 and torch.rand(()).item() < lf_prob:
            hyperparameters = hyperparameters.copy()
            loc = hyperparameters.get("lengthscale_loc_constant_add", sqrt(2))
            hyperparameters["lengthscale_loc_constant_add"] = loc + lf

    # Determine dummy dimensions once (shared across all region functions)
    dummy_dims, num_non_dummy = _get_dummy_dims(num_features, hyperparameters)

    if num_splits == 0:
        inner = _sample_paths_inner(batch_size, num_non_dummy, hyperparameters)
        if dummy_dims is None:
            return inner
        return lambda x: inner(x[:, :, ~dummy_dims])

    # Sample independent GP paths for each region (2^num_splits regions)
    num_regions = 2**num_splits
    region_paths = [
        _sample_paths_inner(batch_size, num_non_dummy, hyperparameters)
        for _ in range(num_regions)
    ]

    # Sample split parameters (on original feature space, not reduced)
    feature_indices = torch.randint(0, num_features, (batch_size, num_splits))
    thresholds = torch.rand(batch_size, num_splits)

    def paths_with_gaps(x):
        # Compute region index for each point (using full x with all features)
        fi_exp = feature_indices.unsqueeze(1).expand(-1, x.shape[1], -1)
        x_at_splits = torch.gather(x, dim=2, index=fi_exp)
        above = (x_at_splits > thresholds.unsqueeze(1)).long()
        powers = (2 ** torch.arange(num_splits)).view(1, 1, -1)
        region_idx = (above * powers).sum(dim=2)  # [batch_size, n]

        # Filter to non-dummy dims for GP evaluation
        x_filtered = x if dummy_dims is None else x[:, :, ~dummy_dims]

        # Evaluate all region paths and select based on region_idx
        all_ys = torch.stack(
            [p(x_filtered).squeeze(0) for p in region_paths], dim=2
        )  # [batch_size, n, num_regions]
        y = torch.gather(all_ys, dim=2, index=region_idx.unsqueeze(2)).squeeze(2)
        return y.unsqueeze(0)  # [1, batch_size, n]

    return paths_with_gaps


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

    # Use shared utility for sampling around training points
    surrounding_test_x = sample_x_around_points(
        batch_size=batch_size,
        num_samples=num_surrounding,
        num_features=num_features,
        centers=train_x,
        std=surrounding_std,
        device=train_x.device,
    )

    x = torch.cat([train_x, normal_test_x, surrounding_test_x], dim=1)
    return x


def add_noise(y, hyperparameters, no_noise=None):
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

    if no_noise is not None:
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

    paths = sample_paths(
        batch_size, num_features, hyperparameters
    )  # paths maps [batch_size, n, num_features] to [batch_size, n, 1]

    sample_clustered_x_hp = hyperparameters.get("sample_clustered_x", None)

    if sample_clustered_x_hp == "trace":
        assert no_noise_prob == 0.0
        x, y = generate_trace(
            batch_size,
            paths,
            partial(add_noise, hyperparameters=hyperparameters),
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
            assert (sample_clustered_x_hp is None) or (
                sample_clustered_x_hp == "none"
            ), sample_clustered_x_hp
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

    predict_advantage = hyperparameters.get("predict_advantage", False)
    if predict_advantage is True:
        target_y = (
            target_y[:, :, :]
            - target_y[:, :single_eval_pos, :]
            .max(dim=-1, keepdim=True)
            .values.max(dim=-2, keepdim=True)
            .values
        )
    elif predict_advantage == "y":
        target_y = (
            target_y[:, :, :]
            - noisy_y[:, :single_eval_pos, :]
            .max(dim=-1, keepdim=True)
            .values.max(dim=-2, keepdim=True)
            .values
        )
    else:
        assert predict_advantage is False

    if hyperparameters.get("relu_target", False):
        target_y = target_y.clamp(min=0.0)

    # set ys to nan in training set
    number_of_y_hidden = torch.randint(
        0, hyperparameters.get("max_num_hidden_y", 0) + 1, tuple()
    )
    noisy_y[:, single_eval_pos - number_of_y_hidden : single_eval_pos] = torch.nan

    return Batch(x=x, y=noisy_y, target_y=target_y)
