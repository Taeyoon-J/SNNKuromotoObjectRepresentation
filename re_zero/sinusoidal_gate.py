from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SinusoidalGate(nn.Module):
    """
    Bridge from oscillator states to SNN input modulation.

    The gate reads a delayed scalar theta state and returns a value in [0, 1].
    """

    def __init__(self, delay: int = 2) -> None:
        super().__init__()
        self.delay = delay

    def sinusoidal_gating(
        self,
        theta_history: List[torch.Tensor],
        theta_current: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build the sinusoidal gate from delayed theta history.

        `gamma` is accepted so this module can later use both theta and gamma
        without changing the outer model API.
        """
        if theta_history:
            delayed_idx = max(0, len(theta_history) - int(self.delay))
            theta_delayed = theta_history[delayed_idx]
        else:
            theta_delayed = theta_current
        return 0.5 * (1.0 + torch.sin(theta_delayed))

