import torch
import torch.nn as nn

from .utils import check_finite


class SeedRefiner(nn.Module):
    """Refine initial UV seed positions with bounded residual updates."""

    def __init__(
        self,
        hidden,
        seed_id_dim,
        eps_uv=1e-4,
        max_step_uv=0.08,
        allow_seed_outside_domain=False,
        seed_domain_margin=0.25,
        enable_checks=True,
    ):
        super().__init__()
        self.eps_uv = eps_uv
        self.max_step_uv = max_step_uv
        self.allow_seed_outside_domain = bool(allow_seed_outside_domain)
        self.seed_domain_margin = float(seed_domain_margin)
        self.enable_checks = enable_checks

        self.seed_refine = nn.Sequential(
            nn.Linear(hidden + 2 + max(int(seed_id_dim), 0), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden, 2)

    def forward(self, z, uv_base, seed_id_features=None, offset_scale=1.0):
        # Per-seed input combines the shared latent, seed UV, and optional seed ID.
        z_rep = z.unsqueeze(0).expand(uv_base.shape[0], -1)
        if seed_id_features is not None:
            seed_in = torch.cat([z_rep, uv_base, seed_id_features], dim=-1)
        else:
            seed_in = torch.cat([z_rep, uv_base], dim=-1)
        check_finite(seed_in, "seed_in", self.enable_checks)

        h = self.seed_refine(seed_in)
        check_finite(h, "h", self.enable_checks)

        delta_raw = self.delta_head(h)
        check_finite(delta_raw, "delta_raw", self.enable_checks)

        # Use UV-space residuals so boundary seeds can still move inward.
        delta_dir = torch.tanh(delta_raw)
        check_finite(delta_dir, "delta_dir", self.enable_checks)

        step_cap = torch.as_tensor(
            self.max_step_uv * offset_scale,
            device=uv_base.device,
            dtype=uv_base.dtype,
        )
        if self.allow_seed_outside_domain:
            delta_uv = delta_dir * step_cap
        else:
            room_lo = (uv_base - self.eps_uv).clamp_min(0.0)
            room_hi = (1.0 - self.eps_uv - uv_base).clamp_min(0.0)
            step_lo = torch.minimum(room_lo, step_cap)
            step_hi = torch.minimum(room_hi, step_cap)
            delta_uv = torch.where(
                delta_dir >= 0.0,
                delta_dir * step_hi,
                delta_dir * step_lo,
            )
        check_finite(delta_uv, "delta_uv", self.enable_checks)

        seeds_uv = uv_base + delta_uv
        if self.allow_seed_outside_domain:
            margin = max(float(self.seed_domain_margin), 0.0)
            seeds_uv = seeds_uv.clamp(-margin, 1.0 + margin)
        else:
            seeds_uv = seeds_uv.clamp(self.eps_uv, 1.0 - self.eps_uv)
        check_finite(seeds_uv, "seeds_uv_final", self.enable_checks)

        return h, seeds_uv
