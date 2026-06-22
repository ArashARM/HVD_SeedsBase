import torch


class Loss_Wactive:
    def __call__(
        self,
        w_raw: torch.Tensor,
        seeds: torch.Tensor,
        seed_active_weights: torch.Tensor | None,
        width_target_frac: float,
        width_target_sparse_boost: float,
        width_target_frac_max: float,
        active_threshold: float,
        raw_temp: float,
        w_min: float,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if w_raw.ndim != 2 or w_raw.shape[0] != w_raw.shape[1]:
            raise ValueError(f"w_raw must be square, got {tuple(w_raw.shape)}")
        if seeds.ndim != 2 or seeds.shape[1] != 2:
            raise ValueError(f"seeds must be (S,2), got {tuple(seeds.shape)}")

        s = w_raw.shape[0]
        if s < 2:
            return torch.zeros((), dtype=w_raw.dtype, device=w_raw.device)

        tri_mask = torch.triu(torch.ones((s, s), dtype=torch.bool, device=w_raw.device), diagonal=1)
        width_frac = 0.5 * (torch.tanh(w_raw / max(float(raw_temp), eps)) + 1.0)

        if seed_active_weights is None:
            target_frac_eff = float(width_target_frac)
            pair_weight = tri_mask.to(dtype=w_raw.dtype)
        else:
            g = seed_active_weights.to(device=w_raw.device, dtype=w_raw.dtype).reshape(-1)
            g = g.clamp(0.0, 1.0)
            active_mass = g.sum().clamp_min(eps)
            sparse_ratio = float(s) / float(active_mass.detach().item())
            target_frac_eff = float(width_target_frac) * (sparse_ratio ** float(width_target_sparse_boost))
            target_frac_eff = min(max(target_frac_eff, 0.0), float(width_target_frac_max))
            pair_weight = torch.sqrt((g[:, None] * g[None, :]).clamp_min(0.0)) * tri_mask.to(dtype=w_raw.dtype)

            if float(pair_weight.sum().detach().item()) <= eps:
                strongest = int(torch.argmax(g).detach().item())
                pair_weight = torch.zeros_like(w_raw)
                if s > 1:
                    partner_scores = g.clone()
                    partner_scores[strongest] = -1.0
                    partner = int(torch.argmax(partner_scores).detach().item())
                    i, j = sorted((strongest, partner))
                    pair_weight[i, j] = 1.0

        pair_weight_sum = pair_weight.sum()
        if float(pair_weight_sum.detach().item()) <= eps:
            return torch.zeros((), dtype=w_raw.dtype, device=w_raw.device)

        target_frac_eff = min(max(float(target_frac_eff), 1e-4), 1.0 - 1e-4)
        active_width = width_frac * pair_weight
        target = torch.as_tensor(target_frac_eff, dtype=w_raw.dtype, device=w_raw.device)
        shortfall = torch.relu(target - width_frac)
        width_floor_loss = (pair_weight * shortfall.square()).sum() / (pair_weight_sum + eps)

        mean_width = active_width.sum() / (pair_weight_sum + eps)
        clarity_loss = (pair_weight * (width_frac - mean_width).square()).sum() / (pair_weight_sum + eps)
        return width_floor_loss + 0.25 * clarity_loss
