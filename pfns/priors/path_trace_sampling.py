import numpy as np
import torch


# Vectorized corner check for batch of points
def corner_check(x, corners):
    # x can be a single point or batch of points
    batching = True
    if x.ndim == 1:
        batching = False
        x = x[np.newaxis, :]

    # Check which points are corners (all coords <= 0 or >= 1)
    is_corner = np.all((x <= 0) | (x >= 1), axis=1)
    new_corners = np.zeros(len(x), dtype=bool)

    # For non-corner points, return True
    results = np.ones(len(x), dtype=bool)

    if np.any(is_corner):
        # For corner points, compute their IDs
        corner_powers = np.array([2**i for i in range(x.shape[1])])
        corner_ids = (x[is_corner] @ corner_powers).round().astype(int)
        corner_ids = np.array(
            [cid * x.shape[0] + i for i, cid in enumerate(corner_ids)]
        )

        # Check which corner IDs are new
        new_corners = np.array([cid not in corners for cid in corner_ids])

        # Add new corner IDs to the set
        corners.update(corner_ids[new_corners])

        # For corner points, return True if new corner, False if already seen
        results[is_corner] = new_corners

    return results if batching else results[0], corners


def sample_until_all_success(sampling_function, corners):
    sample = sampling_function()  # shape: [batch_size, d]
    success, corners = corner_check(sample, corners)
    while not np.all(success):
        # Keep successful samples, only resample failed ones
        failed_mask = ~success
        new_sample = sampling_function()
        new_success, corners = corner_check(new_sample, corners)
        # Update only failed positions with new successful samples
        sample[failed_mask] = new_sample[failed_mask]
        success[failed_mask] = new_success[failed_mask]
    return sample


@torch.no_grad()
def generate_trace(
    batch_size,
    paths,
    add_noise,
    L,
    cutoff,
    bounds,
    best=None,
    never_local=False,
    dtype=torch.float,
):
    """
    Generate optimization traces blending exploration and exploitation for batched Gaussian Processes.

    Parameters:
    - L: int, length of the trace.
    - cutoff: int, position in the trace after which we may sample around the global optimum.
    - bounds: list of tuples [(min1, max1), (min2, max2), ...], search space bounds.
    - t_random_weights: torch.Tensor, random weights for GP Fourier features [batch_size, d, num_features].
    - t_random_offset: torch.Tensor, random offsets for GP Fourier features [batch_size, num_features].
    - t_W_GP: torch.Tensor, weights defining the GP in RFF space [batch_size, num_features, 1].
    - sigma_output: float, output scale of the GP.
    - sigma_noise: float, observation noise level.
    - mean_function: float, mean function of the GP.
    - best: ndarray, optional, the location of the global optimum [batch_size, d].

    Returns:
    - trace: ndarray of shape [batch_size, L, d], the optimization traces.
    - y: ndarray of shape [batch_size, L], the function values at each point in the traces.
    """
    d = len(bounds)
    trace = np.zeros((batch_size, L, d))
    y = np.zeros((batch_size, L))

    corners = set()

    # Initialize
    eps = (1 - np.random.rand(batch_size) ** (d / 6)) / 2
    sigma = np.exp(np.random.normal(-3, 0.5, size=(batch_size,)))
    u = np.random.uniform(size=(batch_size, 3))
    initial_alpha = u.min(axis=1)
    final_alpha = u.max(axis=1)
    trace[:, 0] = np.clip(
        np.random.uniform(-eps[:, None], 1 + eps[:, None], (batch_size, d)), 0, 1
    )
    best_point = trace[:, 0].copy()

    # Get initial values using vectorized GP evaluation
    y[:, :1] = add_noise(paths(torch.tensor(trace[:, :1], dtype=dtype))).numpy()
    y_best = y[:, 0].copy()

    for i in range(1, L):
        alpha = initial_alpha + (final_alpha - initial_alpha) * (i / L)
        local = np.random.rand(batch_size) < alpha
        if never_local:
            local[:] = False

        # could speed up by factor of 2

        def sample_local():
            if i < cutoff:  # noqa: B023
                inc = best_point
            else:
                inc = np.zeros_like(best_point)
                if best is not None:
                    use_best = (L - cutoff) * np.random.rand(batch_size) < 5 * d
                    inc[use_best] = best[use_best]

                if cutoff > 0:
                    use_cutoff = (
                        ~use_best
                        if best is not None
                        else np.ones(batch_size, dtype=bool)
                    )
                    random_cutoff_indices = np.random.choice(cutoff, size=batch_size)
                    inc[use_cutoff] = trace[
                        np.arange(batch_size)[use_cutoff],
                        random_cutoff_indices[use_cutoff],
                    ]
                else:
                    # No point to sample locally around, just sample globally
                    use_global = (
                        ~use_best
                        if best is not None
                        else np.ones(batch_size, dtype=bool)
                    )
                    inc[use_global] = np.clip(
                        np.random.uniform(
                            -eps[use_global, None],
                            1 + eps[use_global, None],
                            (np.sum(use_global), d),
                        ),
                        0,
                        1,
                    )

            ret = np.random.normal(inc, sigma[:, None], size=(batch_size, d))
            return np.clip(
                ret, [low for low, _ in bounds], [high for _, high in bounds]
            )

        def sample_global():
            return np.clip(
                np.random.uniform(-eps[:, None], 1 + eps[:, None], (batch_size, d)),
                0,
                1,
            )

        # Sample points based on local/global strategy
        trace[:, i] = np.where(
            local[:, None],
            sample_until_all_success(sample_local, corners),
            sample_until_all_success(sample_global, corners),
        )

        # Update the current best if before cutoff
        if i < cutoff:
            y[:, i : i + 1] = add_noise(
                paths(torch.tensor(trace[:, i : i + 1], dtype=dtype))
            ).numpy()
            better_mask = y[:, i] > y_best
            best_point[better_mask] = trace[better_mask, i]
            y_best[better_mask] = y[:, i][better_mask]

    # Get noiseless values after cutoff using vectorized evaluation
    if cutoff < L:
        y[:, cutoff:] = paths(torch.tensor(trace[:, cutoff:], dtype=dtype)).numpy()

    return torch.tensor(trace, dtype=dtype), torch.tensor(y, dtype=dtype)
