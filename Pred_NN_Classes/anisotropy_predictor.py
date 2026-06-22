import torch.nn as nn

from .utils import check_finite


class AnisotropyPredictor(nn.Module):
    """Optional per-seed anisotropy heads."""

    def __init__(self, hidden, use_Metric_anisotropy=False, enable_checks=True):
        super().__init__()
        self.use_Metric_anisotropy = use_Metric_anisotropy
        self.enable_checks = enable_checks
        if self.use_Metric_anisotropy:
            self.theta_head = nn.Linear(hidden, 1)
            self.a_head = nn.Linear(hidden, 1)
        else:
            self.theta_head = None
            self.a_head = None

    def forward(self, h):
        if not self.use_Metric_anisotropy:
            return None, None

        theta = self.theta_head(h).squeeze(-1)
        a_raw = self.a_head(h).squeeze(-1)
        check_finite(theta, "theta", self.enable_checks)
        check_finite(a_raw, "a_raw", self.enable_checks)
        return theta, a_raw
