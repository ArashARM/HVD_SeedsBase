import torch


def check_finite(tensor, name, enable_checks):
    """Raise if a tensor contains NaN or Inf values."""
    if enable_checks and not torch.isfinite(tensor).all():
        raise RuntimeError(f"PPNet produced non-finite {name}")
