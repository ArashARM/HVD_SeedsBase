import torch


class Loss_Boundary:
    def __call__(
        self,
        seeds: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        seed_active_weights: torch.Tensor | None = None,
        margin: float = 0.05,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if boundary_uv is None or boundary_uv.numel() == 0:
            return torch.zeros((), dtype=seeds.dtype, device=seeds.device)

        dmin = torch.cdist(seeds, boundary_uv).amin(dim=1)
        penalty = torch.exp(-dmin / (margin + eps))

        if seed_active_weights is None:
            return penalty.mean()

        g = seed_active_weights.view(-1)
        penalty = (g * penalty).sum() / (g.sum() + eps)
        penalty = torch.zeros((), dtype=seeds.dtype, device=seeds.device)
        return penalty
