from abc import ABCMeta, abstractmethod
from typing import Callable

from pfns.base_config import BaseConfig
from torch import Tensor


class FunctionSamplerConfig(BaseConfig, metaclass=ABCMeta):
    @property
    def restricts_sampling_points(self) -> bool:
        """Indicates whether this sampler restricts which points can be sampled.

        If True, the sampler's callable will have a `get_candidate_points` method
        that returns the available candidate points for each batch element.
        """
        return False

    @abstractmethod
    def function_sampler(
        self, batch_size: int, num_features: int = 1, device: str = "cpu"
    ) -> Callable[
        [Tensor], Tensor
    ]:  # going from tensor of shape (batch_size, n) to (batch_size, n)
        pass
