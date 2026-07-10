
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
        num_seeds = seeds.shape[0]

        if num_seeds < 2:
            return seeds.new_zeros(())

        distances = torch.cdist(seeds, seeds)

        pair_mask = torch.triu(
            torch.ones(
                (num_seeds, num_seeds),
                dtype=torch.bool,
                device=seeds.device,
            ),
            diagonal=1,
        )

        if min_dist is not None and min_dist > 0.0:
            target = seeds.new_tensor(min_dist)

            pair_penalty = (
                torch.relu(target - distances).square()
                / target.square().clamp_min(eps)
            )
        else:
            sigma_tensor = seeds.new_tensor(sigma)

            pair_penalty = torch.exp(
                -distances.square()
                / sigma_tensor.square().clamp_min(eps)
            )

        pair_penalty = pair_penalty[pair_mask]

        if seed_active_weights is None:
            return pair_penalty.mean()

        weights = seed_active_weights.reshape(-1).to(
            dtype=seeds.dtype,
            device=seeds.device,
        ).clamp(0.0, 1.0)

        if weights.numel() != num_seeds:
            raise ValueError(
                "seed_active_weights must contain one value per seed."
            )

        pair_weights = (
            weights[:, None] * weights[None, :]
        )[pair_mask]

        weight_sum = pair_weights.sum()

        if not bool(weight_sum > eps):
            return seeds.new_zeros(())

        return (
            pair_weights * pair_penalty
        ).sum() / weight_sum.clamp_min(eps)
