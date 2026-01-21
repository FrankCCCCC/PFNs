# This is a wrapper prior that adds heteroscedastic noise to datasets.
# The noise variance varies spatially based on a random GP function, and can be
# either normally distributed or long-tailed (with outliers).

from copy import deepcopy

import torch
from gpytorch.priors import LogNormalPrior
from pfns.priors import Batch

from .path_stgp import sample_paths


@torch.no_grad()
def get_batch(
    batch_size: int,
    seq_len: int,
    num_features: int,
    *args,
    hyperparameters: dict | None = None,
    get_batch=None,
    **kwargs,
) -> Batch:
    """Generate a batch with heteroscedastic noise added to the base prior.

    This wrapper prior adds spatially-varying noise to datasets. The noise
    variance at each point is determined by a random GP function sampled
    using sample_paths from path_stgp.py. The noise can be either normal
    or long-tailed (Student-t distribution with outliers).

    Hyperparameters:
        hetero_noise_prob: Probability of making the noise heteroscedastic
            (spatially varying) vs homoscedastic (constant) (default: 0.5)
        hetero_noise_long_tailed_prob: Probability of using long-tailed noise
            vs normal noise (default: 0.5)
        hetero_noise_df_min: Minimum degrees of freedom for Student-t when
            long-tailed (default: 2.0)
        hetero_noise_df_max: Maximum degrees of freedom for Student-t when
            long-tailed (default: 5.0)
        hetero_noise_var_loc: Location parameter for log-normal base noise
            variance distribution (default: -4.0, same as path_stgp)
        hetero_noise_var_scale: Scale parameter for log-normal base noise
            variance distribution (default: 1.0, same as path_stgp)
        hetero_noise_range_scale: Scale factor for heteroscedastic variation
            on top of base noise (default: 1.0). The heteroscedastic
            component adds variation in [0, base_std * range_scale].

        Additional hyperparameters are passed to sample_paths for the
        variance function GP:
        - use_rbf_kernel, lengthscale_loc_constant_add, lengthscale_loc_feature_mul,
          lengthscale_scale, mean_width, additive_cosine_per_dim_prob

    Args:
        batch_size: Number of samples in the batch
        seq_len: Sequence length (number of points per sample)
        num_features: Number of input features
        hyperparameters: Dictionary of hyperparameters
        get_batch: The underlying prior's get_batch function to wrap
        **kwargs: Additional arguments passed to the underlying prior

    Returns:
        Batch with heteroscedastic noise added
    """
    if hyperparameters is None:
        hyperparameters = {}

    hyperparameters = deepcopy(hyperparameters)

    # Extract heteroscedastic noise hyperparameters
    hetero_prob: float = hyperparameters.pop("hetero_noise_prob", 0.5)
    long_tailed_prob: float = hyperparameters.pop("hetero_noise_long_tailed_prob", 0.5)
    df_min: float = hyperparameters.pop("hetero_noise_df_min", 2.0)
    df_max: float = hyperparameters.pop("hetero_noise_df_max", 5.0)
    noise_var_loc: float = hyperparameters.pop("hetero_noise_var_loc", -4.0)
    noise_var_scale: float = hyperparameters.pop("hetero_noise_var_scale", 1.0)

    # Extract hyperparameters for the variance GP (with prefixed names)
    variance_gp_hyperparameters = {}
    variance_gp_keys = [
        "hetero_noise_use_rbf_kernel",
        "hetero_noise_lengthscale_loc_constant_add",
        "hetero_noise_lengthscale_loc_feature_mul",
        "hetero_noise_lengthscale_scale",
        "hetero_noise_mean_width",
        "hetero_noise_additive_cosine_per_dim_prob",
    ]

    for key in variance_gp_keys:
        if key in hyperparameters:
            # Remove prefix and add to variance GP hyperparameters
            gp_key = key.replace("hetero_noise_", "")
            variance_gp_hyperparameters[gp_key] = hyperparameters.pop(key)

    # Set defaults for variance GP if not provided
    variance_gp_hyperparameters.setdefault("use_rbf_kernel", True)
    variance_gp_hyperparameters.setdefault("mean_width", 1.0)

    if get_batch is None:
        raise ValueError(
            "heteroscedastic_noise_prior requires a base get_batch function to wrap"
        )

    # Get batch from base prior
    base_batch = get_batch(
        batch_size,
        seq_len,
        num_features,
        *args,
        hyperparameters=hyperparameters,
        **kwargs,
    )

    device = base_batch.x.device
    dtype = base_batch.x.dtype
    x = base_batch.x  # (batch_size, seq_len, num_features)

    # Decide per batch element whether to use heteroscedastic or homoscedastic noise
    use_hetero = torch.rand(batch_size, device=device) < hetero_prob

    # Sample base noise variance from log-normal distribution (like path_stgp.py)
    base_noise_variance: torch.Tensor = LogNormalPrior(
        loc=noise_var_loc,
        scale=noise_var_scale,
    ).sample((batch_size,))
    base_noise_std = base_noise_variance.sqrt().to(device=device, dtype=dtype)

    # Sample the variance function using sample_paths from path_stgp
    # This returns a function that maps x -> y where y varies smoothly
    variance_paths = sample_paths(batch_size, num_features, variance_gp_hyperparameters)

    # Evaluate the variance function at all x points
    # variance_paths expects (batch_size, n, num_features) and returns (1, batch_size, n)
    variance_func_values = variance_paths(x).squeeze(0)  # (batch_size, seq_len)

    # Normalize variance function values to [0, 1] per batch
    var_min = variance_func_values.min(dim=1, keepdim=True).values
    var_max = variance_func_values.max(dim=1, keepdim=True).values
    var_range = (var_max - var_min).clamp(min=1e-8)
    normalized_variance = (variance_func_values - var_min) / var_range  # [0, 1]

    # For heteroscedastic noise:
    # std = base_std + normalized_variance * (base_std * range_scale)
    # This adds variation in [0, base_std * range_scale] on top of base_std
    base_std_expanded = base_noise_std.unsqueeze(1)  # (batch_size, 1)
    range_size = torch.rand(batch_size, device=device) * 2  # [0, 2]
    hetero_std = base_std_expanded * (
        1 - range_size.clamp(max=1.0).unsqueeze(1)
    ) + normalized_variance * (
        base_std_expanded * range_size.unsqueeze(1)
    )  # (batch_size, seq_len)

    # For homoscedastic noise: just use the base std
    homo_std = base_std_expanded.expand(-1, seq_len)  # (batch_size, seq_len)

    # Select heteroscedastic or homoscedastic std per batch element
    point_std = torch.where(
        use_hetero.unsqueeze(1).expand(-1, seq_len),
        hetero_std,
        homo_std,
    )  # (batch_size, seq_len)

    # Decide per batch element whether to use long-tailed or normal noise
    use_long_tailed = torch.rand(batch_size, device=device) < long_tailed_prob

    # Sample noise
    # For normal noise: N(0, std^2)
    # For long-tailed noise: Student-t with df degrees of freedom, scaled by std
    normal_noise = torch.randn(batch_size, seq_len, device=device, dtype=dtype)

    # Sample degrees of freedom for Student-t (per batch)
    df = torch.rand(batch_size, device=device) * (df_max - df_min) + df_min

    # Sample Student-t noise using the relationship: t = Z / sqrt(V/df)
    # where Z ~ N(0,1) and V ~ Chi-squared(df)
    chi2_samples = torch.distributions.Chi2(
        df.unsqueeze(1).expand(-1, seq_len)
    ).sample()
    student_t_noise = normal_noise / torch.sqrt(
        chi2_samples / df.unsqueeze(1)
    )  # (batch_size, seq_len)

    # Select noise type based on use_long_tailed
    noise = torch.where(
        use_long_tailed.unsqueeze(1).expand(-1, seq_len),
        student_t_noise,
        normal_noise,
    )

    # Scale noise by point-wise std
    scaled_noise = noise * point_std  # (batch_size, seq_len)

    # Add noise to y
    y_with_noise = base_batch.y + scaled_noise.unsqueeze(2)
    target_y = base_batch.target_y
    if hyperparameters.get("noisy_predictions", False):
        target_y += scaled_noise.unsqueeze(2)

    # # Plot noises on linspace [0,1] if num_features == 1 (for prototyping)
    # import matplotlib.pyplot as plt
    # if num_features == 1:
    #     print("hiii")
    #     x_plot = torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(-1)  # (1, 100, 1)
    #     x_plot = x_plot.expand(batch_size, -1, -1).to(device=device, dtype=dtype)

    #     # Evaluate variance function on linspace
    #     variance_func_plot = variance_paths(x_plot).squeeze(0)  # (batch_size, 100)

    #     # Normalize to [0, 1]
    #     var_min_plot = variance_func_plot.min(dim=1, keepdim=True).values
    #     var_max_plot = variance_func_plot.max(dim=1, keepdim=True).values
    #     var_range_plot = (var_max_plot - var_min_plot).clamp(min=1e-8)
    #     normalized_variance_plot = (variance_func_plot - var_min_plot) / var_range_plot

    #     # Compute std on linspace
    #     hetero_std_plot = base_std_expanded * (
    #         1 - range_size.clamp(max=1.0).unsqueeze(1)
    #     ) + normalized_variance_plot * (base_std_expanded * range_size.unsqueeze(1))

    #     # Plot a few samples
    #     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    #     _ = fig  # suppress unused variable warning
    #     x_np = torch.linspace(0, 1, 100).numpy()
    #     for i, ax in enumerate(axes.flat):
    #         if i < batch_size:
    #             ax.plot(x_np, hetero_std_plot[i].cpu().numpy(), label="std(x)")
    #             ax.set_xlabel("x")
    #             ax.set_ylabel("noise std")
    #             ax.set_title(f"Sample {i} (hetero={use_hetero[i].item()})")
    #             ax.legend()
    #     plt.suptitle("Heteroscedastic Noise Std on [0,1]")
    #     plt.tight_layout()
    #     plt.show()

    return Batch(
        x=base_batch.x,
        y=y_with_noise,
        target_y=target_y,
        style=base_batch.style,
    )
