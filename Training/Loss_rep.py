import torch


class Loss_rep:
    def __call__(
        self,
        seeds: torch.Tensor,
        seed_active_weights: torch.Tensor | None = None,
        sigma: float = 0.08,
        min_dist: float | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        s = seeds.shape[0]
        if s < 2:
            return torch.zeros((), dtype=seeds.dtype, device=seeds.device)

        d = torch.cdist(seeds, seeds)
        pair_mask = torch.triu(
            torch.ones((s, s), dtype=torch.bool, device=seeds.device),
            diagonal=1,
        )

        if min_dist is not None and min_dist > 0.0:
            target = torch.as_tensor(min_dist, dtype=seeds.dtype, device=seeds.device)
            penalty = torch.relu(target - d).pow(2) / (target.pow(2) + eps)
        else:
            penalty = torch.exp(-d.pow(2) / (sigma**2 + eps))

        if seed_active_weights is None:
            return penalty[pair_mask].max()

        g = seed_active_weights.view(-1).clamp(0.0, 1.0)
        pair_weight = g[:, None] * g[None, :]
        active_pair = pair_mask & (pair_weight > eps)
        if not bool(active_pair.any()):
            return torch.zeros((), dtype=seeds.dtype, device=seeds.device)
        return (pair_weight * penalty)[active_pair].max()
