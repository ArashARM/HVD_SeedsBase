import math
import torch
import torch.nn as nn

from .utils import check_finite


class TauPredictor(nn.Module):
    """Optional global tau head scaled into [tau_pred_min, tau_pred_max]."""

    def __init__(
        self,
        hidden,
        predict_tau=False,
        tau_pred_start=0.02,
        tau_pred_min=1e-4,
        tau_pred_max=0.2,
        enable_checks=True,
    ):
        super().__init__()
        self.predict_tau = predict_tau
        self.tau_pred_start = float(tau_pred_start)
        self.tau_pred_min = float(tau_pred_min)
        self.tau_pred_max = float(tau_pred_max)
        self.enable_checks = enable_checks

        if self.predict_tau:
            if not (self.tau_pred_min > 0.0):
                raise ValueError(f"tau_pred_min must be > 0, got {self.tau_pred_min}")
            if not (self.tau_pred_max > self.tau_pred_min):
                raise ValueError(
                    f"tau_pred_max must be > tau_pred_min, got min={self.tau_pred_min}, max={self.tau_pred_max}"
                )
            if not (self.tau_pred_min <= self.tau_pred_start <= self.tau_pred_max):
                raise ValueError(
                    "tau_pred_start must lie within [tau_pred_min, tau_pred_max], "
                    f"got start={self.tau_pred_start}, min={self.tau_pred_min}, max={self.tau_pred_max}"
                )

            self.tau_head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.tau_head.weight)
            tau_range = self.tau_pred_max - self.tau_pred_min
            tau_frac = (self.tau_pred_start - self.tau_pred_min) / tau_range
            tau_frac = min(max(tau_frac, 1e-6), 1.0 - 1e-6)
            nn.init.constant_(self.tau_head.bias, math.log(tau_frac / (1.0 - tau_frac)))
        else:
            self.tau_head = None

    def forward(self, z):
        if not self.predict_tau:
            return None

        tau_logits = self.tau_head(z).reshape(())
        tau = self.tau_pred_min + (self.tau_pred_max - self.tau_pred_min) * torch.sigmoid(tau_logits)
        check_finite(tau, "tau", self.enable_checks)
        return tau
