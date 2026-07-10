import torch
import torch.nn as nn

from .seed_identity import SeedIdentityEmbedding
from .seed_refiner import SeedRefiner
from .utils import check_finite
from .width_predictor import WidthPredictor


class PPNet(nn.Module):
    """
    PPNet predicts only the parameters used by the current centerline decoder.

    It refines seed UV locations and exposes one trainable global raw strut
    width expanded to the legacy ``w_raw`` matrix shape.
    """

    def __init__(
        self,
        n_seeds,
        hidden=256,
        freeze_w=False,
        w_const=0.25,
        w_head_bias_init=0.0,
        eps_uv=1e-4,
        max_delta_logit=0.30,
        max_step_uv=0.08,
        seed_id_dim=16,
        use_independent_seed_offsets=True,
        independent_seed_offset_max=0.05,
        allow_seed_outside_domain=False,
        seed_domain_margin=0.25,
        enable_checks=True,
        **unused_kwargs,
    ):
        super().__init__()

        self.n_seeds = n_seeds
        self.freeze_w = freeze_w
        self.w_const = w_const

        self.eps_uv = eps_uv
        self.max_delta_logit = max_delta_logit
        self.max_step_uv = max_step_uv
        self.seed_id_dim = int(seed_id_dim)
        self.use_independent_seed_offsets = bool(use_independent_seed_offsets)
        self.independent_seed_offset_max = float(independent_seed_offset_max)
        self.allow_seed_outside_domain = bool(allow_seed_outside_domain)
        self.seed_domain_margin = float(seed_domain_margin)

        self.enable_checks = enable_checks

        # This model is used as an optimization parameterization for a single problem instance, so no external context conditioning is required.
        self.global_latent = nn.Parameter(torch.zeros(hidden))
        # Each seed gets a learnable identity embedding, which can help the network learn seed-specific behavior.
        self.seed_identity = SeedIdentityEmbedding(self.n_seeds, self.seed_id_dim)
        # The seed refiner takes the global latent, seed UVs, and optional seed ID features to produce refined seed positions and a hidden representation for each seed.
        self.seed_refiner = SeedRefiner(
            hidden=hidden,
            seed_id_dim=self.seed_id_dim,
            eps_uv=self.eps_uv,
            max_step_uv=self.max_step_uv,
            allow_seed_outside_domain=self.allow_seed_outside_domain,
            seed_domain_margin=self.seed_domain_margin,
            enable_checks=self.enable_checks,
        )
        if self.use_independent_seed_offsets:
            self.seed_free_offset_raw = nn.Parameter(torch.zeros(self.n_seeds, 2))
        else:
            self.seed_free_offset_raw = None
        self.width_predictor = WidthPredictor(
            hidden=hidden,
            freeze_w=self.freeze_w,
            w_const=self.w_const,
            w_head_bias_init=w_head_bias_init,
            enable_checks=self.enable_checks,
        )

    # Compatibility properties for existing training code.
    @property
    def seed_id_embed(self):
        return self.seed_identity.embedding

    @property
    def seed_refine(self):
        return self.seed_refiner.seed_refine

    @property
    def delta_head(self):
        return self.seed_refiner.delta_head

    @property
    def independent_seed_offsets(self):
        return self.seed_free_offset_raw

    @property
    def w_head(self):
        # The trainer still groups width parameters through this legacy name.
        return self.width_predictor

    @property
    def h_head(self):
        return None

    @property
    def theta_head(self):
        return None

    @property
    def a_head(self):
        return None

    @property
    def boundary_width_head(self):
        return None

    @property
    def boundary_alpha_head(self):
        return None

    @property
    def boundary_beta_head(self):
        return None

    @property
    def tau_head(self):
        return None

    def _check(self, tensor, name):
        check_finite(tensor, name, self.enable_checks)

    def _clamp_seeds_to_domain(self, seeds_uv):
        if self.allow_seed_outside_domain:
            margin = max(float(self.seed_domain_margin), 0.0)
            return seeds_uv.clamp(-margin, 1.0 + margin)
        return seeds_uv.clamp(self.eps_uv, 1.0 - self.eps_uv)

    def _apply_independent_seed_offsets(self, seeds_uv):
        if self.seed_free_offset_raw is None:
            return seeds_uv
        if self.seed_free_offset_raw.shape != seeds_uv.shape:
            raise ValueError(
                "seed_free_offset_raw must match seeds_uv shape, "
                f"got {tuple(self.seed_free_offset_raw.shape)} vs {tuple(seeds_uv.shape)}"
            )

        offset_cap = torch.as_tensor(
            max(float(self.independent_seed_offset_max), 0.0),
            device=seeds_uv.device,
            dtype=seeds_uv.dtype,
        )
        free_delta = torch.tanh(self.seed_free_offset_raw.to(dtype=seeds_uv.dtype)) * offset_cap
        check_finite(free_delta, "seed_free_delta", self.enable_checks)
        return self._clamp_seeds_to_domain(seeds_uv + free_delta)

    def forward(self, uv_init, offset_scale=1.0):
        n_seeds = self.n_seeds
        eps_uv = self.eps_uv

        if uv_init.dim() != 2 or uv_init.shape[-1] != 2:
            raise ValueError("uv_init must be (S,2)")

        if self.allow_seed_outside_domain:
            uv_base = uv_init
        else:
            uv_base = uv_init.clamp(eps_uv, 1.0 - eps_uv)
        self._check(uv_base, "uv_base")

        z = self.global_latent
        self._check(z, "z")

        # z + uv + seed_id_features -> SeedRefiner -> h, seeds_uv
        seed_id_features = self.seed_identity(n_seeds, uv_base.device)
        self.seed_refiner.allow_seed_outside_domain = bool(self.allow_seed_outside_domain)
        self.seed_refiner.seed_domain_margin = float(self.seed_domain_margin)
        h, seeds_uv = self.seed_refiner(
            z,
            uv_base,
            seed_id_features=seed_id_features,
            offset_scale=offset_scale,
        )
        seeds_uv = self._apply_independent_seed_offsets(seeds_uv)

        self.width_predictor.freeze_w = self.freeze_w
        self.width_predictor.w_const = self.w_const
        w_raw = self.width_predictor(h, n_seeds, z)

        return {
            "seeds_raw": seeds_uv,
            "w_raw": w_raw,
        }
