import torch


class Loss_SeedActive:
    def __call__(
        self,
        seed_active_weights: torch.Tensor,
        target_active: float,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        active_mass = seed_active_weights.clamp(0.0, 1.0).sum()
        target_active_t = torch.as_tensor(
            float(target_active),
            device=seed_active_weights.device,
            dtype=seed_active_weights.dtype,
        )
        return torch.relu(target_active_t - active_mass).pow(2)
