from dataclasses import dataclass
from math import sqrt
from typing import Literal, Optional

import torch
from gpytorch.priors import LogNormalPrior
from pfns.priors.path_stgp import sample_paths

from .function_sampler import FunctionSamplerConfig


@dataclass(frozen=True)
class Config(FunctionSamplerConfig):
    # Noise variance distribution type: "gamma" or "lognormal"
    noise_var_dist: Literal["gamma", "lognormal"] = "lognormal"

    # Gamma distribution parameters (used when noise_var_dist="gamma")
    noise_variance_gamma_concentration: float = 0.9
    noise_variance_gamma_rate: float = 10.0

    # LogNormal distribution parameters (used when noise_var_dist="lognormal")
    noise_var_loc: Optional[float] = -4.0
    noise_var_scale: Optional[float] = 1.0

    # sample_paths hyperparameters
    use_rbf_kernel: bool = True
    lengthscale_loc_constant_add: float = sqrt(2)
    lengthscale_loc_feature_mul: float = 0.5
    lengthscale_scale: float = sqrt(3)
    mean_width: float = 2.0

    # Dummy dimension configuration
    # If set, sample number of non-dummy dimensions from [min, max] range
    # E.g., (1, 3) means 1-3 dimensions are non-dummy, rest are ignored
    dummy_dim_sample_non_dummy_range: Optional[tuple[int, int]] = None
    # Probability of applying dummy_dim_sample_non_dummy_range logic.
    # If not applied, all dimensions are non-dummy.
    dummy_dim_sample_non_dummy_range_prob: float = 1.0

    # Gap discontinuity configuration
    gap_max_splits: int = 0  # Max axis-aligned splits (0 = disabled)
    gap_prob: float = 1.0  # Probability of applying gaps
    # Add to lengthscale_loc_constant_add when gaps applied
    gap_lengthscale_add: float = 0.0
    # Probability of applying lengthscale adjustment
    gap_lengthscale_add_prob: float = 1.0

    def _sample_noise_variance(self, batch_size: int) -> torch.Tensor:
        if self.noise_var_dist == "lognormal":
            return LogNormalPrior(
                loc=self.noise_var_loc,
                scale=self.noise_var_scale,
            ).sample((batch_size,))
        elif self.noise_var_dist == "gamma":
            return (
                torch.distributions.Gamma(
                    self.noise_variance_gamma_concentration,
                    self.noise_variance_gamma_rate,
                ).sample((batch_size,))
                + 1e-4
            )
        else:
            raise ValueError(
                f"Unknown noise variance distribution {self.noise_var_dist}"
            )

    @torch.no_grad()
    def function_sampler(self, batch_size, num_features=1, device="cpu", seed=None):
        hyperparameters = {
            "use_rbf_kernel": self.use_rbf_kernel,
            "lengthscale_loc_constant_add": self.lengthscale_loc_constant_add,
            "lengthscale_loc_feature_mul": self.lengthscale_loc_feature_mul,
            "lengthscale_scale": self.lengthscale_scale,
            "mean_width": self.mean_width,
            "dummy_dim_sample_non_dummy_range": self.dummy_dim_sample_non_dummy_range,
            "dummy_dim_sample_non_dummy_range_prob": self.dummy_dim_sample_non_dummy_range_prob,
            "gap_max_splits": self.gap_max_splits,
            "gap_prob": self.gap_prob,
            "gap_lengthscale_add": self.gap_lengthscale_add,
            "gap_lengthscale_add_prob": self.gap_lengthscale_add_prob,
        }
        paths = sample_paths(batch_size, num_features, hyperparameters)

        noise_variance: torch.Tensor = self._sample_noise_variance(batch_size)

        @torch.no_grad()
        def noisy_eval(batch_inputs, independent_noise=False):
            # Calculate a spike function: resembles a triangle with peak at 1
            noiseless_outputs = paths(batch_inputs.cpu())[0]
            if independent_noise:
                noise = (
                    torch.randn(batch_inputs.shape[:-1])
                    * noise_variance[:, None] ** (1 / 2)
                ).squeeze(-1)
            else:
                noise = (torch.randn(batch_size) * noise_variance ** (1 / 2))[:, None]
            outputs = noiseless_outputs + noise

            return noiseless_outputs.to(device), outputs.to(device)

        return noisy_eval
