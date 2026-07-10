
import torch

class Loss_rep:
    def __call__(
        self,
        seeds: torch.Tensor,
        seed_active_weights: torch.Tensor | None = None,
        sigma: float = 0.08,
        min_dist: float | None = None,
        temperature: float = 0.01,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if seeds.ndim != 2:
            raise ValueError(
                f"Expected seeds with shape (num_seeds, dimension), "
                f"but received {tuple(seeds.shape)}."
            )

        num_seeds = seeds.shape[0]

        if num_seeds < 2:
            return seeds.new_zeros(())

        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, received {sigma}.")

        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be positive, received {temperature}."
            )

        distances = torch.cdist(seeds, seeds)

        pair_mask = torch.triu(
            torch.ones(
                (num_seeds, num_seeds),
                dtype=torch.bool,
                device=seeds.device,
            ),
            diagonal=1,
        )

        if min_dist is not None:
            if min_dist <= 0.0:
                raise ValueError(
                    f"min_dist must be positive when provided, "
                    f"received {min_dist}."
                )

            target = seeds.new_tensor(min_dist)

            penalty = (
                torch.relu(target - distances).square()
                / target.square().clamp_min(eps)
            )
        else:
            sigma_tensor = seeds.new_tensor(sigma)
            penalty = torch.exp(
                -distances.square()
                / sigma_tensor.square().clamp_min(eps)
            )

        pair_penalty = penalty[pair_mask]

        if seed_active_weights is not None:
            weights = seed_active_weights.reshape(-1)

            if weights.numel() != num_seeds:
                raise ValueError(
                    "seed_active_weights must contain one value per seed: "
                    f"expected {num_seeds}, received {weights.numel()}."
                )

            weights = weights.to(
                dtype=seeds.dtype,
                device=seeds.device,
            ).clamp(0.0, 1.0)

            pair_weights = (
                weights[:, None] * weights[None, :]
            )[pair_mask]

            active_mask = pair_weights > eps

            if not torch.any(active_mask):
                return seeds.new_zeros(())

            pair_penalty = pair_penalty[active_mask]
            pair_weights = pair_weights[active_mask]

            # Include activity weights in the smooth maximum.
            scores = (
                pair_penalty / temperature
                + torch.log(pair_weights.clamp_min(eps))
            )

            loss = temperature * (
                torch.logsumexp(scores, dim=0)
                - torch.log(pair_weights.sum().clamp_min(eps))
            )
        else:
            # Normalization makes the smooth maximum zero when every
            # pair penalty is zero.
            normalizer = torch.log(
                pair_penalty.new_tensor(pair_penalty.numel())
            )

            loss = temperature * (
                torch.logsumexp(pair_penalty / temperature, dim=0)
                - normalizer
            )

        return loss.clamp_min(0.0)
