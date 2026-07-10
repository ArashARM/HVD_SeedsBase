import torch.nn as nn

from .utils import check_finite


class BoundaryPredictor(nn.Module):
    """Optional global boundary-attachment parameter heads."""

    def __init__(self, hidden, predict_boundary_params=False, enable_checks=True):
        super().__init__()
        self.predict_boundary_params = predict_boundary_params
        self.enable_checks = enable_checks
        if self.predict_boundary_params:
            self.boundary_width_head = nn.Linear(hidden, 1)
            self.boundary_alpha_head = nn.Linear(hidden, 1)
            self.boundary_beta_head = nn.Linear(hidden, 1)
        else:
            self.boundary_width_head = None
            self.boundary_alpha_head = None
            self.boundary_beta_head = None

    def forward(self, z):
        if not self.predict_boundary_params:
            return None, None, None

        boundary_width_raw = self.boundary_width_head(z).reshape(())
        boundary_alpha_raw = self.boundary_alpha_head(z).reshape(())
        boundary_beta_raw = self.boundary_beta_head(z).reshape(())

        check_finite(boundary_width_raw, "boundary_width_raw", self.enable_checks)
        check_finite(boundary_alpha_raw, "boundary_alpha_raw", self.enable_checks)
        check_finite(boundary_beta_raw, "boundary_beta_raw", self.enable_checks)
        return boundary_width_raw, boundary_alpha_raw, boundary_beta_raw
