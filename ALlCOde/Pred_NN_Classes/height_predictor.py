import torch.nn as nn

from .utils import check_finite


class HeightPredictor(nn.Module):
    """Optional global height head."""

    def __init__(self, hidden, predict_height=False, enable_checks=True):
        super().__init__()
        self.predict_height = predict_height
        self.enable_checks = enable_checks
        if self.predict_height:
            self.h_head = nn.Linear(hidden, 1)
        else:
            self.h_head = None

    def forward(self, z):
        if not self.predict_height:
            return None

        h_raw = self.h_head(z).reshape(())
        check_finite(h_raw, "h_raw", self.enable_checks)
        return h_raw
