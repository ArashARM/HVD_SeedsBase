import torch
import torch.nn as nn

from .utils import check_finite


class WidthPredictor(nn.Module):
    """Expand one global raw strut-width parameter to the decoder matrix shape."""

    def __init__(
        self,
        hidden,
        freeze_w=False,
        w_const=0.25,
        w_head_bias_init=0.0,
        enable_checks=True,
    ):
        super().__init__()
        self.freeze_w = freeze_w
        self.w_const = w_const
        self.enable_checks = enable_checks

        self.w_raw = nn.Parameter(torch.tensor(float(w_head_bias_init)))

    def forward(self, h, n_seeds, z):
        if self.freeze_w:
            raw = torch.as_tensor(self.w_const, device=z.device, dtype=z.dtype)
        else:
            raw = self.w_raw.to(device=z.device, dtype=z.dtype)

        w_raw = raw.expand(n_seeds, n_seeds)

        check_finite(w_raw, "w_raw", self.enable_checks)
        return w_raw
