from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from dataclasses import dataclass
from typing import Any, Callable
from scipy.spatial import Voronoi, voronoi_plot_2d



class VoronoiDecoder(nn.Module):
    """
    Fully functional Voronoi decoder.

    Fiber directions are treated as an axial line field:
    t and -t are equivalent. To avoid sign-cancellation artifacts,
    pairwise directions are blended through orientation tensors t t^T.
    """

    def __init__(
        self,
        n_seeds: int,
        eps: float = 1e-8,
        use_Metric_anisotropy: bool = True,
        use_surface_metric: bool = True,
        use_metric_voronoi_distance: bool = False,
        normalize_metric_voronoi_distance: bool = True,
        use_band_weighted_fiber_pairs: bool = True,
        use_boundary_tangent_fibers: bool = True,
        fiber_band_prior_power: float = 2.0,
        fiber_band_prior_floor: float = 0.05,
        pair_boost_strength: float = 0.05,
        pair_boost_enabled: bool = False,
        skeleton_sigma: float = 0.1,
        skeleton_sigma_metric: float | None = None,

        # geometric strut half-width lower bound
        w_min: float = 0.05,
        w_min_metric: float | None = None,
        w_max_ratio: float = 0.8,
        width_cap_pair_quantile: float = 0.25,


        # density transition sharpness
        beta: float = 0.01,
        junction_beta_scale: float = 1.0,
        junction_width_bonus: float = 0.15,

        # effective-number boost for multi-seed zones
        junction_keff_lambda: float = 0.050,
        junction_keff_k0: float = 3.0,
        junction_keff_s: float = 0.35,

        # explicit triple-overlap junction term
        junction_triple_lambda: float = 0,
        junction_triple_power: float = 1.5,

        # raw parameter temperature for bounded maps
        raw_temp: float = 1.25,

        # union sharpness for combining pair bands
        alpha_union: float = 16.0,

        # optional smooth density projection. This keeps gradients continuous
        # while making the density used by FEM closer to a visible strut field.
        density_projection_strength: float = 0.0,
        density_projection_threshold: float = 0.5,
        density_projection_gamma: float = 0.05,

        # optional 3D distance-to-skeleton thickness field. When enabled,
        # Voronoi equations define only the centerline/core topology and this
        # physical radius controls the visible strut width.
        use_centerline_thickness: bool = False,
        centerline_threshold: float = 0.5,
        centerline_softmin_tau: float = 0.01,
        centerline_radius_min: float = 0.005,
        centerline_radius_max: float = 0.05,
        centerline_radius_fixed: float | None = None,
        centerline_beta: float = 0.002,
        centerline_threshold_softness: float = 0.05,

        # duplicate-seed activation. Seeds closer than this radius compete,
        # and one survivor remains effective in each connected duplicate cluster.
        duplicate_merge_sigma: float = 0.05,
        duplicate_merge_sigma_metric: float | None = None,
        duplicate_effect_temp_ratio: float = 0.20,
        duplicate_effect_strength: float = 6.0,
        duplicate_effect_floor: float = 5e-2,
        seed_activity_sharpness: float = 1.0,
        seed_activity_threshold: float = 0.5,
        domain_effect_floor: float = 1e-8,
        domain_pair_power: float = 2.0,
        duplicate_pair_power: float = 1.0,
        pair_activity_power: float | None = None,
        global_activity_power: float = 2.0,
        invalid_domain_assignment_threshold: float = 1e-6,
        point_domain_floor: float = 0.0,

        # height controls
        h_min: float = 0.50,
        h_max: float = 2.00,
        fixed_height: float | None = None,

        # boundary & periodicity
        boundary_solid_idx: torch.Tensor | None = None,
        face_u_periodic: torch.Tensor | None = None,
        face_v_periodic: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,

        # boundary attachment field
        use_boundary_attachment: bool = False,

        # keep these on comparable scales
        boundary_attach_width: float = 2e-5,
        boundary_attach_width_metric: float | None = None,
        boundary_attach_beta: float = 1e-5,
        boundary_attach_beta_metric: float | None = None,
        boundary_attach_alpha: float = 0.35,

        boundary_attach_width_min: float = 5e-6,
        boundary_attach_width_max: float = 5e-5,
        boundary_attach_width_min_metric: float | None = None,
        boundary_attach_width_max_metric: float | None = None,

        boundary_attach_alpha_min: float = 0.05,
        boundary_attach_alpha_max: float = 1.00,

        boundary_attach_beta_min: float = 1e-6,
        boundary_attach_beta_max: float = 1e-4,
        boundary_attach_beta_min_metric: float | None = None,
        boundary_attach_beta_max_metric: float | None = None,

        # robust boundary-distance evaluation
        boundary_knn_k: int = 8,
        boundary_softmin_tau: float = 2e-3,
        boundary_spacing_blend: float = 0.5,
        debug_metric_distances: bool = False,
    ):
        super().__init__()

        self.n_seeds = int(n_seeds)
        self.eps = float(eps)
        self.use_Metric_anisotropy = bool(use_Metric_anisotropy)
        self.use_surface_metric = bool(use_surface_metric)
        self.use_metric_voronoi_distance = bool(use_metric_voronoi_distance)
        self.normalize_metric_voronoi_distance = bool(normalize_metric_voronoi_distance)
        self.use_band_weighted_fiber_pairs = bool(use_band_weighted_fiber_pairs)
        self.use_boundary_tangent_fibers = bool(use_boundary_tangent_fibers)
        self.fiber_band_prior_power = float(fiber_band_prior_power)
        self.fiber_band_prior_floor = float(fiber_band_prior_floor)

        self.w_min = float(w_min)
        self.w_min_metric = float(w_min if w_min_metric is None else w_min_metric)
        self.w_max_ratio = float(w_max_ratio)
        self.width_cap_pair_quantile = float(width_cap_pair_quantile)
        self.beta = float(beta)
        self.skeleton_sigma = float(skeleton_sigma)
        self.skeleton_sigma_metric = float(
            skeleton_sigma if skeleton_sigma_metric is None else skeleton_sigma_metric
        )
        self.junction_beta_scale = float(junction_beta_scale)
        self.junction_width_bonus = float(junction_width_bonus)

        self.junction_keff_lambda = float(junction_keff_lambda)
        self.junction_keff_k0 = float(junction_keff_k0)
        self.junction_keff_s = float(junction_keff_s)

        self.junction_triple_lambda = float(junction_triple_lambda)
        self.junction_triple_power = float(junction_triple_power)

        self.raw_temp = float(raw_temp)
        self.alpha_union = float(alpha_union)
        self.density_projection_strength = float(density_projection_strength)
        self.density_projection_threshold = float(density_projection_threshold)
        self.density_projection_gamma = float(density_projection_gamma)
        self.use_centerline_thickness = bool(use_centerline_thickness)
        self.centerline_threshold = float(centerline_threshold)
        self.centerline_softmin_tau = float(centerline_softmin_tau)
        self.centerline_radius_min = float(centerline_radius_min)
        self.centerline_radius_max = float(centerline_radius_max)
        self.centerline_radius_fixed = (
            None if centerline_radius_fixed is None else float(centerline_radius_fixed)
        )
        self.centerline_beta = float(centerline_beta)
        self.centerline_threshold_softness = float(centerline_threshold_softness)
        self.duplicate_merge_sigma = float(duplicate_merge_sigma)
        self.duplicate_merge_sigma_metric = float(
            duplicate_merge_sigma
            if duplicate_merge_sigma_metric is None
            else duplicate_merge_sigma_metric
        )
        self.duplicate_effect_temp_ratio = float(duplicate_effect_temp_ratio)
        self.duplicate_effect_strength = float(duplicate_effect_strength)
        self.duplicate_effect_floor = float(duplicate_effect_floor)
        self.seed_activity_sharpness = float(seed_activity_sharpness)
        self.seed_activity_threshold = float(seed_activity_threshold)
        self.domain_effect_floor = float(domain_effect_floor)
        self.domain_pair_power = float(
            domain_pair_power if pair_activity_power is None else pair_activity_power
        )
        self.duplicate_pair_power = float(duplicate_pair_power)
        self.global_activity_power = float(global_activity_power)
        self.invalid_domain_assignment_threshold = float(invalid_domain_assignment_threshold)
        self.point_domain_floor = float(point_domain_floor)

        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.fixed_height = float(fixed_height) if fixed_height is not None else None

        self.use_boundary_attachment = bool(use_boundary_attachment)

        self.boundary_attach_width_min = float(boundary_attach_width_min)
        self.boundary_attach_width_max = float(boundary_attach_width_max)
        self.boundary_attach_width_min_metric = float(
            boundary_attach_width_min
            if boundary_attach_width_min_metric is None
            else boundary_attach_width_min_metric
        )
        self.boundary_attach_width_max_metric = float(
            boundary_attach_width_max
            if boundary_attach_width_max_metric is None
            else boundary_attach_width_max_metric
        )
        self.boundary_attach_alpha_min = float(boundary_attach_alpha_min)
        self.boundary_attach_alpha_max = float(boundary_attach_alpha_max)
        self.boundary_attach_beta_min = float(boundary_attach_beta_min)
        self.boundary_attach_beta_max = float(boundary_attach_beta_max)
        self.boundary_attach_beta_min_metric = float(
            boundary_attach_beta_min
            if boundary_attach_beta_min_metric is None
            else boundary_attach_beta_min_metric
        )
        self.boundary_attach_beta_max_metric = float(
            boundary_attach_beta_max
            if boundary_attach_beta_max_metric is None
            else boundary_attach_beta_max_metric
        )
        self.boundary_knn_k = int(boundary_knn_k)
        self.boundary_softmin_tau = float(boundary_softmin_tau)
        self.boundary_spacing_blend = float(boundary_spacing_blend)
        self.debug_metric_distances = bool(debug_metric_distances)

        self.pair_boost_enabled = bool(pair_boost_enabled)
        self.pair_boost_strength = float(pair_boost_strength)

        if not (self.boundary_attach_width_min < self.boundary_attach_width_max):
            raise ValueError(
                f"boundary_attach_width_min must be < boundary_attach_width_max, got "
                f"{self.boundary_attach_width_min} and {self.boundary_attach_width_max}"
            )
        if not (self.boundary_attach_width_min_metric < self.boundary_attach_width_max_metric):
            raise ValueError(
                f"boundary_attach_width_min_metric must be < boundary_attach_width_max_metric, got "
                f"{self.boundary_attach_width_min_metric} and {self.boundary_attach_width_max_metric}"
            )
        if not (self.boundary_attach_alpha_min < self.boundary_attach_alpha_max):
            raise ValueError(
                f"boundary_attach_alpha_min must be < boundary_attach_alpha_max, got "
                f"{self.boundary_attach_alpha_min} and {self.boundary_attach_alpha_max}"
            )
        if not (self.boundary_attach_beta_min < self.boundary_attach_beta_max):
            raise ValueError(
                f"boundary_attach_beta_min must be < boundary_attach_beta_max, got "
                f"{self.boundary_attach_beta_min} and {self.boundary_attach_beta_max}"
            )
        if not (self.boundary_attach_beta_min_metric < self.boundary_attach_beta_max_metric):
            raise ValueError(
                f"boundary_attach_beta_min_metric must be < boundary_attach_beta_max_metric, got "
                f"{self.boundary_attach_beta_min_metric} and {self.boundary_attach_beta_max_metric}"
            )
        if self.boundary_knn_k < 1:
            raise ValueError(f"boundary_knn_k must be >= 1, got {self.boundary_knn_k}")
        if self.boundary_softmin_tau <= 0:
            raise ValueError(f"boundary_softmin_tau must be > 0, got {self.boundary_softmin_tau}")
        if self.boundary_spacing_blend < 0:
            raise ValueError(f"boundary_spacing_blend must be >= 0, got {self.boundary_spacing_blend}")
        if self.junction_triple_power <= 0:
            raise ValueError(f"junction_triple_power must be > 0, got {self.junction_triple_power}")
        if self.duplicate_merge_sigma <= 0:
            raise ValueError(f"duplicate_merge_sigma must be > 0, got {self.duplicate_merge_sigma}")
        if self.duplicate_effect_temp_ratio <= 0:
            raise ValueError(
                f"duplicate_effect_temp_ratio must be > 0, got {self.duplicate_effect_temp_ratio}"
            )
        if self.duplicate_effect_strength < 0:
            raise ValueError(
                f"duplicate_effect_strength must be >= 0, got {self.duplicate_effect_strength}"
            )
        if not (0.0 < self.duplicate_effect_floor <= 1.0):
            raise ValueError(
                f"duplicate_effect_floor must be in (0, 1], got {self.duplicate_effect_floor}"
            )
        if self.seed_activity_sharpness <= 0.0:
            raise ValueError(
                f"seed_activity_sharpness must be > 0, got {self.seed_activity_sharpness}"
            )
        if not (0.0 < self.seed_activity_threshold < 1.0):
            raise ValueError(
                "seed_activity_threshold must be in (0, 1), "
                f"got {self.seed_activity_threshold}"
            )
        if not (0.0 < self.domain_effect_floor <= 1.0):
            raise ValueError(
                f"domain_effect_floor must be in (0, 1], got {self.domain_effect_floor}"
            )
        if self.domain_pair_power <= 0.0:
            raise ValueError(f"domain_pair_power must be > 0, got {self.domain_pair_power}")
        if self.duplicate_pair_power <= 0.0:
            raise ValueError(
                f"duplicate_pair_power must be > 0, got {self.duplicate_pair_power}"
            )
        if self.global_activity_power <= 0.0:
            raise ValueError(
                f"global_activity_power must be > 0, got {self.global_activity_power}"
            )
        if self.invalid_domain_assignment_threshold < 0.0:
            raise ValueError(
                "invalid_domain_assignment_threshold must be >= 0, "
                f"got {self.invalid_domain_assignment_threshold}"
            )
        if not (0.0 <= self.point_domain_floor <= 1.0):
            raise ValueError(
                f"point_domain_floor must be in [0, 1], got {self.point_domain_floor}"
            )
        if self.fiber_band_prior_power <= 0.0:
            raise ValueError(f"fiber_band_prior_power must be > 0, got {self.fiber_band_prior_power}")
        if not (0.0 <= self.fiber_band_prior_floor <= 1.0):
            raise ValueError(
                f"fiber_band_prior_floor must be in [0, 1], got {self.fiber_band_prior_floor}"
            )
        if self.alpha_union <= 0.0:
            raise ValueError(f"alpha_union must be > 0, got {self.alpha_union}")
        if not (0.0 <= self.density_projection_strength <= 1.0):
            raise ValueError(
                "density_projection_strength must be in [0,1], "
                f"got {self.density_projection_strength}"
            )
        if self.density_projection_gamma <= 0.0:
            raise ValueError(
                f"density_projection_gamma must be > 0, got {self.density_projection_gamma}"
            )
        if self.centerline_softmin_tau <= 0.0:
            raise ValueError(
                f"centerline_softmin_tau must be > 0, got {self.centerline_softmin_tau}"
            )
        if self.centerline_beta <= 0.0:
            raise ValueError(f"centerline_beta must be > 0, got {self.centerline_beta}")
        if self.centerline_threshold_softness <= 0.0:
            raise ValueError(
                "centerline_threshold_softness must be > 0, "
                f"got {self.centerline_threshold_softness}"
            )
        if not (0.0 <= self.centerline_threshold <= 1.0):
            raise ValueError(
                "centerline_threshold must be in [0,1], "
                f"got {self.centerline_threshold}"
            )
        if self.centerline_radius_fixed is not None and self.centerline_radius_fixed < 0.0:
            raise ValueError(
                "centerline_radius_fixed must be >= 0 when provided, "
                f"got {self.centerline_radius_fixed}"
            )
        if not (self.centerline_radius_min < self.centerline_radius_max):
            raise ValueError(
                "centerline_radius_min must be < centerline_radius_max, got "
                f"{self.centerline_radius_min} and {self.centerline_radius_max}"
            )

        if boundary_solid_idx is None:
            boundary_solid_idx = torch.empty(0, dtype=torch.long)
        if face_u_periodic is None:
            face_u_periodic = torch.zeros(1, dtype=torch.bool)
        if face_v_periodic is None:
            face_v_periodic = torch.zeros(1, dtype=torch.bool)
        if seed_face_id is None:
            seed_face_id = torch.zeros(self.n_seeds, dtype=torch.long)

        self.register_buffer("boundary_solid_idx", boundary_solid_idx.to(torch.long))
        self.register_buffer("face_u_periodic", face_u_periodic.to(torch.bool))
        self.register_buffer("face_v_periodic", face_v_periodic.to(torch.bool))
        self.register_buffer("seed_face_id", seed_face_id.to(torch.long))

        self.register_buffer(
            "boundary_attach_width_fixed",
            torch.tensor(float(boundary_attach_width), dtype=torch.float32),
        )
        self.register_buffer(
            "boundary_attach_width_metric_fixed",
            torch.tensor(
                float(boundary_attach_width if boundary_attach_width_metric is None else boundary_attach_width_metric),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "boundary_attach_alpha_fixed",
            torch.tensor(float(boundary_attach_alpha), dtype=torch.float32),
        )
        self.register_buffer(
            "boundary_attach_beta_fixed",
            torch.tensor(float(boundary_attach_beta), dtype=torch.float32),
        )
        self.register_buffer(
            "boundary_attach_beta_metric_fixed",
            torch.tensor(
                float(boundary_attach_beta if boundary_attach_beta_metric is None else boundary_attach_beta_metric),
                dtype=torch.float32,
            ),
        )

    # -------------------- parameter maps --------------------

    def seeds_uv(self, seeds_raw: torch.Tensor) -> torch.Tensor:
        return seeds_raw

    def _seed_face_id_for(
        self,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seed_face_id is not None:
            return seed_face_id.to(device=seeds.device, dtype=torch.long)
        if self.seed_face_id.shape[0] == seeds.shape[0]:
            return self.seed_face_id.to(device=seeds.device, dtype=torch.long)
        return torch.zeros(seeds.shape[0], device=seeds.device, dtype=torch.long)

    def _use_surface_distance_geometry(self) -> bool:
        return bool(self.use_surface_metric or self.use_metric_voronoi_distance)

    def _metric_scale_value(self, uv_value: float, metric_value: float) -> float:
        return metric_value if self._use_surface_distance_geometry() else uv_value

    def _debug_distance_stats(self, name: str, value: torch.Tensor | None) -> None:
        if not self.debug_metric_distances or value is None or value.numel() == 0:
            return
        v = value.detach()
        finite = v[torch.isfinite(v)]
        if finite.numel() == 0:
            print(f"{name}: no finite values")
            return
        print(
            f"{name}: min={finite.min().item():.6g} "
            f"mean={finite.mean().item():.6g} max={finite.max().item():.6g}"
        )

    def _distance_stats_tensor(self, value: torch.Tensor | None) -> torch.Tensor:
        if value is None or value.numel() == 0:
            return torch.empty(0)
        v = value.detach()
        finite = v[torch.isfinite(v)]
        if finite.numel() == 0:
            return torch.full((3,), float("nan"), device=v.device, dtype=v.dtype)
        return torch.stack([finite.min(), finite.mean(), finite.max()])

    def _wrap_seed_seed_diff(
        self,
        diff: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        S = diff.shape[0]
        if seed_face_id is None:
            seed_face_id = self._seed_face_id_for(
                torch.empty(S, 2, device=diff.device, dtype=diff.dtype)
            )
        else:
            seed_face_id = seed_face_id.to(device=diff.device, dtype=torch.long)

        same_face = seed_face_id[:, None] == seed_face_id[None, :]
        uper_face = self.face_u_periodic.to(device=diff.device)[seed_face_id]
        vper_face = self.face_v_periodic.to(device=diff.device)[seed_face_id]

        uper_pair = uper_face[:, None] & uper_face[None, :] & same_face
        vper_pair = vper_face[:, None] & vper_face[None, :] & same_face

        wrapped = diff.clone()
        wrapped[..., 0] = wrapped[..., 0] - torch.round(wrapped[..., 0]) * uper_pair.to(diff.dtype)
        wrapped[..., 1] = wrapped[..., 1] - torch.round(wrapped[..., 1]) * vper_pair.to(diff.dtype)
        return wrapped

    def _wrap_point_seed_diff(
        self,
        diff: torch.Tensor,
        points_face_id: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if points_face_id is None:
            return diff

        points_face_id = points_face_id.to(device=diff.device, dtype=torch.long)
        if seed_face_id is None:
            same_face = torch.ones(diff.shape[:2], device=diff.device, dtype=torch.bool)
            uper = self.face_u_periodic.to(device=diff.device)[points_face_id][:, None]
            vper = self.face_v_periodic.to(device=diff.device)[points_face_id][:, None]
        else:
            seed_face_id = seed_face_id.to(device=diff.device, dtype=torch.long)
            same_face = points_face_id[:, None] == seed_face_id[None, :]
            uper = self.face_u_periodic.to(device=diff.device)[points_face_id][:, None] & same_face
            vper = self.face_v_periodic.to(device=diff.device)[points_face_id][:, None] & same_face

        wrapped = diff.clone()
        wrapped[..., 0] = wrapped[..., 0] - torch.round(wrapped[..., 0]) * uper.to(diff.dtype)
        wrapped[..., 1] = wrapped[..., 1] - torch.round(wrapped[..., 1]) * vper.to(diff.dtype)
        return wrapped

    def _pairwise_seed_dist(
        self,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
        seed_metric: torch.Tensor | None = None,
        metric_aware: bool | None = None,
    ) -> torch.Tensor:
        return self._seed_seed_distance_metric(
            seeds=seeds,
            seed_face_id=seed_face_id,
            seed_metric=seed_metric,
            metric_aware=metric_aware,
        )

    def _sample_seed_domain_values(
        self,
        seeds: torch.Tensor,
        domain: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
        *,
        name: str,
    ) -> torch.Tensor:
        domain_is_callable = callable(domain)
        if callable(domain):
            values = domain(seeds)
            if torch.is_tensor(values):
                values = values.to(device=seeds.device, dtype=seeds.dtype)
            else:
                values = torch.as_tensor(values, device=seeds.device, dtype=seeds.dtype)
        else:
            if torch.is_tensor(domain):
                values = domain.to(device=seeds.device, dtype=seeds.dtype)
            else:
                values = torch.as_tensor(domain, device=seeds.device, dtype=seeds.dtype)

        if values.ndim == 0:
            values = values.expand(seeds.shape[0])
        elif values.shape == (seeds.shape[0],):
            pass
        elif values.shape == (seeds.shape[0], 1):
            values = values.reshape(seeds.shape[0])
        elif values.ndim == 2 and values.shape[-1] == 1 and values.shape[0] == seeds.shape[0]:
            values = values.squeeze(-1)
        elif not domain_is_callable and values.ndim in (2, 3, 4):
            if values.ndim == 2:
                grid_values = values.unsqueeze(0).unsqueeze(0)
            elif values.ndim == 3:
                grid_values = values.unsqueeze(0) if values.shape[0] == 1 else values.unsqueeze(1)
            else:
                grid_values = values

            if grid_values.shape[0] != 1 or grid_values.shape[1] != 1:
                raise ValueError(
                    f"{name} grid must be (H,W), (1,H,W), (1,1,H,W), or per-seed; "
                    f"got {tuple(values.shape)}"
                )

            uv_grid = seeds.reshape(1, -1, 1, 2) * 2.0 - 1.0
            sampled = F.grid_sample(
                grid_values,
                uv_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            values = sampled.reshape(seeds.shape[0])
        else:
            raise ValueError(
                f"{name} must be callable, per-seed, or UV grid; got {tuple(values.shape)}"
            )

        if values.shape != (seeds.shape[0],):
            raise ValueError(f"{name} must evaluate to ({seeds.shape[0]},), got {tuple(values.shape)}")
        return values

    @staticmethod
    def _domain_can_sample_count(
        domain: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None,
        count: int,
    ) -> bool:
        if domain is None:
            return False
        if callable(domain):
            return True
        values = domain if torch.is_tensor(domain) else torch.as_tensor(domain)
        if values.ndim == 0:
            return True
        if values.shape == (count,) or values.shape == (count, 1):
            return True
        if values.ndim == 2 and values.shape[1] == 1:
            return False
        return values.ndim in (2, 3, 4)

    def _seed_domain_validity_state(
        self,
        seeds: torch.Tensor,
        temp: torch.Tensor,
        seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = torch.ones((seeds.shape[0],), device=seeds.device, dtype=seeds.dtype)
        active = torch.ones((seeds.shape[0],), device=seeds.device, dtype=torch.bool)
        sdf_values = torch.empty((0,), device=seeds.device, dtype=seeds.dtype)
        mask_values = torch.empty((0,), device=seeds.device, dtype=seeds.dtype)

        if seed_domain_sdf is not None:
            sdf_values = self._sample_seed_domain_values(seeds, seed_domain_sdf, name="seed_domain_sdf")
            sdf_weight = torch.sigmoid(sdf_values / temp.clamp_min(self.eps))
            weight = weight * sdf_weight
            active = active & (sdf_values >= 0.0)

        if seed_domain_mask is not None:
            mask_values = self._sample_seed_domain_values(seeds, seed_domain_mask, name="seed_domain_mask")
            threshold = torch.as_tensor(
                seed_domain_mask_threshold,
                device=seeds.device,
                dtype=seeds.dtype,
            )
            mask_weight = torch.sigmoid((mask_values - threshold) / temp.clamp_min(self.eps))
            weight = weight * mask_weight
            active = active & (mask_values >= threshold)

        return weight.clamp(0.0, 1.0), active, sdf_values, mask_values

    def _sharpen_seed_activity(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.clamp(0.0, 1.0)
        if self.seed_activity_sharpness == 1.0:
            return weights

        sharpness = torch.as_tensor(
            self.seed_activity_sharpness,
            device=weights.device,
            dtype=weights.dtype,
        ).clamp_min(self.eps)
        threshold = torch.as_tensor(
            self.seed_activity_threshold,
            device=weights.device,
            dtype=weights.dtype,
        )
        temp = (0.25 / sharpness).clamp_min(self.eps)
        raw = torch.sigmoid((weights - threshold) / temp)
        lo = torch.sigmoid((torch.zeros_like(threshold) - threshold) / temp)
        hi = torch.sigmoid((torch.ones_like(threshold) - threshold) / temp)
        return ((raw - lo) / (hi - lo).clamp_min(self.eps)).clamp(0.0, 1.0)

    def _point_domain_validity_state(
        self,
        points_uv: torch.Tensor,
        temp: torch.Tensor,
        point_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        point_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        point_domain_mask_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = torch.ones((points_uv.shape[0],), device=points_uv.device, dtype=points_uv.dtype)
        sdf_values = torch.empty((0,), device=points_uv.device, dtype=points_uv.dtype)
        mask_values = torch.empty((0,), device=points_uv.device, dtype=points_uv.dtype)

        if point_domain_sdf is not None:
            sdf_values = self._sample_seed_domain_values(
                points_uv,
                point_domain_sdf,
                name="point_domain_sdf",
            )
            weight = weight * torch.sigmoid(sdf_values / temp.clamp_min(self.eps))

        if point_domain_mask is not None:
            mask_values = self._sample_seed_domain_values(
                points_uv,
                point_domain_mask,
                name="point_domain_mask",
            )
            threshold = torch.as_tensor(
                point_domain_mask_threshold,
                device=points_uv.device,
                dtype=points_uv.dtype,
            )
            mask_weight = torch.sigmoid((mask_values - threshold) / temp.clamp_min(self.eps))
            weight = weight * mask_weight

        return weight.clamp(0.0, 1.0), sdf_values, mask_values

    def _seed_activation_state(
        self,
        seeds: torch.Tensor,
        hard_seed_mask: bool = True,
        seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask_threshold: float = 0.5,
        seed_domain_temp: float | torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
        seed_metric: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        s = seeds.shape[0]
        temp = torch.as_tensor(
            max(
                self._metric_scale_value(self.duplicate_merge_sigma, self.duplicate_merge_sigma_metric)
                * float(self.duplicate_effect_temp_ratio),
                self.eps,
            ),
            device=seeds.device,
            dtype=seeds.dtype,
        )
        domain_temp = temp if seed_domain_temp is None else torch.as_tensor(
            seed_domain_temp,
            device=seeds.device,
            dtype=seeds.dtype,
        ).clamp_min(self.eps)
        duplicate_floor = torch.as_tensor(
            self.duplicate_effect_floor,
            device=seeds.device,
            dtype=seeds.dtype,
        )
        domain_floor = torch.as_tensor(
            self.domain_effect_floor,
            device=seeds.device,
            dtype=seeds.dtype,
        )

        if s <= 1:
            if s == 0:
                active = torch.ones((s,), device=seeds.device, dtype=torch.bool)
                empty = torch.empty((s,), device=seeds.device, dtype=seeds.dtype)
                ones = torch.ones((s,), device=seeds.device, dtype=seeds.dtype)
                return ones, active, empty, empty, empty, ones, ones
            u = seeds[:, 0]
            v = seeds[:, 1]
            outside_dist = torch.stack(
                [
                    -u,
                    u - 1.0,
                    -v,
                    v - 1.0,
                    torch.zeros_like(u),
                ],
                dim=0,
            ).amax(dim=0)
            inside_domain = outside_dist <= 0.0
            square_domain_weight = torch.where(
                inside_domain,
                torch.ones_like(outside_dist),
                torch.sigmoid(-outside_dist / temp.clamp_min(self.eps)),
            )
            active = inside_domain
            uv_domain_weight, uv_domain_active, sdf_values, mask_values = self._seed_domain_validity_state(
                seeds=seeds,
                temp=domain_temp,
                seed_domain_sdf=seed_domain_sdf,
                seed_domain_mask=seed_domain_mask,
                seed_domain_mask_threshold=seed_domain_mask_threshold,
            )
            domain_weight = square_domain_weight * uv_domain_weight
            active = active & uv_domain_active
            duplicate_weight = torch.ones_like(domain_weight)
            domain_activity = domain_floor + (1.0 - domain_floor) * domain_weight
            weights = duplicate_weight * domain_activity
            weights = self._sharpen_seed_activity(weights)
            if hard_seed_mask:
                weights = weights * active.to(seeds.dtype)
            return weights, active, domain_weight, sdf_values, mask_values, duplicate_weight, domain_activity

        dist = self._seed_seed_distance_metric(
            seeds=seeds,
            seed_face_id=seed_face_id,
            seed_metric=seed_metric,
            metric_aware=self._use_surface_distance_geometry() and seed_metric is not None,
        ).to(device=seeds.device, dtype=seeds.dtype)
        radius = torch.as_tensor(
            self._metric_scale_value(self.duplicate_merge_sigma, self.duplicate_merge_sigma_metric),
            device=seeds.device,
            dtype=seeds.dtype,
        )
        temp = (radius * float(self.duplicate_effect_temp_ratio)).clamp_min(self.eps)
        soft_close = torch.sigmoid((radius - dist) / temp)
        soft_close = soft_close.masked_fill(torch.eye(s, dtype=torch.bool, device=seeds.device), 0.0)
        lower_priority = torch.tril(
            torch.ones((s, s), dtype=seeds.dtype, device=seeds.device),
            diagonal=-1,
        )
        suppress_mass = (soft_close * lower_priority).sum(dim=1)
        raw_duplicate_weight = torch.exp(-float(self.duplicate_effect_strength) * suppress_mass)
        duplicate_weight = duplicate_floor + (1.0 - duplicate_floor) * raw_duplicate_weight
        u = seeds[:, 0]
        v = seeds[:, 1]
        outside_dist = torch.stack(
            [
                -u,
                u - 1.0,
                -v,
                v - 1.0,
                torch.zeros_like(u),
            ],
            dim=0,
        ).amax(dim=0)
        inside_domain = outside_dist <= 0.0
        square_domain_weight = torch.where(
            inside_domain,
            torch.ones_like(outside_dist),
            torch.sigmoid(-outside_dist / temp.clamp_min(self.eps)),
        )
        active = inside_domain
        uv_domain_weight, uv_domain_active, sdf_values, mask_values = self._seed_domain_validity_state(
            seeds=seeds,
            temp=domain_temp,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
        )
        domain_weight = square_domain_weight * uv_domain_weight
        active = active & uv_domain_active
        domain_activity = domain_floor + (1.0 - domain_floor) * domain_weight
        weights = duplicate_weight * domain_activity
        weights = self._sharpen_seed_activity(weights)
        if hard_seed_mask:
            weights = weights * active.to(seeds.dtype)
        if self.debug_metric_distances:
            self._debug_distance_stats("duplicate_suppression_dist", dist)
        return weights, active, domain_weight, sdf_values, mask_values, duplicate_weight, domain_activity

    def _pair_distinctness(
        self,
        seeds: torch.Tensor,
        device=None,
        dtype=None,
        seed_face_id: torch.Tensor | None = None,
        seed_metric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = seeds.device
        if dtype is None:
            dtype = seeds.dtype

        S = seeds.shape[0]
        pair_dist = self._seed_seed_distance_metric(
            seeds=seeds,
            seed_face_id=seed_face_id,
            seed_metric=seed_metric,
            metric_aware=self._use_surface_distance_geometry() and seed_metric is not None,
        ).to(device=device, dtype=dtype)
        sigma = torch.as_tensor(
            self._metric_scale_value(self.duplicate_merge_sigma, self.duplicate_merge_sigma_metric),
            device=device,
            dtype=dtype,
        )

        distinctness = -torch.expm1(-(pair_dist.pow(2)) / (sigma.pow(2) + self.eps))
        distinctness = distinctness.pow(2)
        distinctness = distinctness.clamp(0.0, 1.0)
        return distinctness * self._strict_upper_tri_mask(S, device, dtype)

    def width(
        self,
        w_raw: torch.Tensor,
        seeds: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
        seed_metric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T = self.raw_temp
        if w_raw.ndim != 2 or w_raw.shape[0] != w_raw.shape[1]:
            raise ValueError(f"w_raw must be square (S,S), got {tuple(w_raw.shape)}")
        if seeds is None:
            raise ValueError("seeds must be provided when w_raw is pairwise")
        if seeds.shape[0] != w_raw.shape[0]:
            raise ValueError(
                f"pairwise w_raw expects seeds with matching S, got {tuple(seeds.shape)} and {tuple(w_raw.shape)}"
            )

        pair_dist = self._seed_seed_distance_metric(
            seeds=seeds,
            seed_face_id=seed_face_id,
            seed_metric=seed_metric,
            metric_aware=self._use_surface_distance_geometry() and seed_metric is not None,
        ).to(device=w_raw.device, dtype=w_raw.dtype)
        pair_mask = torch.triu(
            torch.ones_like(pair_dist, dtype=torch.bool),
            diagonal=1,
        )
        if bool(pair_mask.any()):
            pair_dist_active = pair_dist[pair_mask]
            q = min(max(self.width_cap_pair_quantile, 0.0), 1.0)
            cap_pair_dist = torch.quantile(pair_dist_active, q)
            width_raw_global = w_raw[pair_mask].mean()
        else:
            cap_pair_dist = torch.zeros((), device=w_raw.device, dtype=w_raw.dtype)
            width_raw_global = w_raw.mean()

        w_min = self._metric_scale_value(self.w_min, self.w_min_metric)
        w_max = (self.w_max_ratio * cap_pair_dist).clamp_min(w_min)
        width_frac = 0.5 * (torch.tanh(width_raw_global / max(T, self.eps)) + 1.0)
        w_geo = w_min + (w_max - w_min) * width_frac
        if self.debug_metric_distances:
            self._debug_distance_stats("seed_seed_dist", pair_dist[pair_mask] if bool(pair_mask.any()) else pair_dist)
            self._debug_distance_stats("w_geo_eff", w_geo)
        return w_geo.expand_as(w_raw)

    def height(
        self,
        h_raw: torch.Tensor | None,
        ref_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fixed_height is not None:
            if ref_tensor is not None:
                return torch.tensor(
                    float(self.fixed_height),
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                )
            if h_raw is not None:
                return torch.tensor(
                    float(self.fixed_height),
                    device=h_raw.device,
                    dtype=h_raw.dtype,
                )
            return torch.tensor(float(self.fixed_height))

        if h_raw is None:
            raise ValueError("h_raw must be provided when fixed_height is None")

        return self.h_min + (self.h_max - self.h_min) * torch.sigmoid(h_raw)

    def _map_raw_to_range(
        self,
        x_raw: torch.Tensor,
        lo: float,
        hi: float,
        temp: float = 1.0,
    ) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(x_raw / temp)

    def raw_from_bounded_value(
        self,
        value: float,
        lo: float,
        hi: float,
        temp: float = 1.0,
    ) -> torch.Tensor:
        denom = max(hi - lo, self.eps)
        x = (value - lo) / denom
        x = min(max(x, 1e-6), 1.0 - 1e-6)
        raw = temp * math.log(x / (1.0 - x))
        return torch.tensor(raw, dtype=torch.float32)

    # -------------------- boundary control getters --------------------

    def boundary_width(
        self,
        ref_tensor: torch.Tensor,
        boundary_width_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_width_raw is None:
            fixed = (
                self.boundary_attach_width_metric_fixed
                if self._use_surface_distance_geometry()
                else self.boundary_attach_width_fixed
            )
            return fixed.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        raw = boundary_width_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        lo = self._metric_scale_value(
            self.boundary_attach_width_min,
            self.boundary_attach_width_min_metric,
        )
        hi = self._metric_scale_value(
            self.boundary_attach_width_max,
            self.boundary_attach_width_max_metric,
        )
        return self._map_raw_to_range(
            raw,
            lo,
            hi,
            temp=1.0,
        )

    def boundary_alpha(
        self,
        ref_tensor: torch.Tensor,
        boundary_alpha_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_alpha_raw is None:
            return self.boundary_attach_alpha_fixed.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        raw = boundary_alpha_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return self._map_raw_to_range(
            raw,
            self.boundary_attach_alpha_min,
            self.boundary_attach_alpha_max,
            temp=1.0,
        )

    def boundary_beta(
        self,
        ref_tensor: torch.Tensor,
        boundary_beta_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_beta_raw is None:
            fixed = (
                self.boundary_attach_beta_metric_fixed
                if self._use_surface_distance_geometry()
                else self.boundary_attach_beta_fixed
            )
            return fixed.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        raw = boundary_beta_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        lo = self._metric_scale_value(
            self.boundary_attach_beta_min,
            self.boundary_attach_beta_min_metric,
        )
        hi = self._metric_scale_value(
            self.boundary_attach_beta_max,
            self.boundary_attach_beta_max_metric,
        )
        return self._map_raw_to_range(
            raw,
            lo,
            hi,
            temp=1.0,
        )

    # -------------------- anisotropic metric --------------------

    def metric_matrices(
        self,
        theta: torch.Tensor,
        a_raw: torch.Tensor,
        a_min: float = 0.5,
        a_max: float = 2.0,
    ) -> torch.Tensor:
        if theta.ndim != 1 or a_raw.ndim != 1 or theta.shape != a_raw.shape:
            raise ValueError(
                f"metric_matrices expects theta and a_raw of shape (S,), got {theta.shape}, {a_raw.shape}"
            )

        S = theta.shape[0]
        t = torch.tanh(a_raw)
        a = 0.5 * (a_max - a_min) * t + 0.5 * (a_max + a_min)

        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack(
            [torch.stack([c, -s], -1), torch.stack([s, c], -1)],
            -2,
        )

        D = torch.zeros((S, 2, 2), device=R.device, dtype=R.dtype)
        D[:, 0, 0] = a
        D[:, 1, 1] = 1.0 / (a + self.eps)

        return R.transpose(1, 2) @ D @ R

    # -------------------- periodic helpers --------------------

    def _wrap_duv_points_to_seeds(
        self,
        diff: torch.Tensor,
        points_face_id: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._wrap_point_seed_diff(diff, points_face_id=points_face_id)

    def _surface_metric_components(
        self,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E = (Xu * Xu).sum(dim=-1)
        Fm = (Xu * Xv).sum(dim=-1)
        Gm = (Xv * Xv).sum(dim=-1)
        return E, Fm, Gm

    def _point_seed_distance(
        self,
        points_uv: torch.Tensor,
        seeds: torch.Tensor,
        Xu: torch.Tensor | None = None,
        Xv: torch.Tensor | None = None,
        metric_aware: bool = False,
        normalize_metric: bool = True,
        eps: float = 1e-8,
        points_face_id: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._point_seed_distance_metric(
            points_uv=points_uv,
            seeds=seeds,
            Xu=Xu,
            Xv=Xv,
            metric_aware=metric_aware,
            normalize_metric=normalize_metric,
            eps=eps,
            points_face_id=points_face_id,
            seed_face_id=seed_face_id,
        )

    def _metric_distance_from_diff(
        self,
        diff: torch.Tensor,
        metric: torch.Tensor | None = None,
        *,
        eps: float | None = None,
    ) -> torch.Tensor:
        eps_t = self.eps if eps is None else eps
        if metric is None:
            return torch.linalg.norm(diff, dim=-1)
        d2 = torch.einsum("...i,...ij,...j->...", diff, metric, diff)
        return torch.sqrt(d2.clamp_min(eps_t))

    def _point_seed_distance_metric(
        self,
        points_uv: torch.Tensor,
        seeds: torch.Tensor,
        Xu: torch.Tensor | None = None,
        Xv: torch.Tensor | None = None,
        point_metric: torch.Tensor | None = None,
        seed_metric: torch.Tensor | None = None,
        metric_aware: bool | None = None,
        normalize_metric: bool | None = None,
        eps: float | None = None,
        points_face_id: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
        mask_cross_face: bool = False,
        return_diff: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        eps_t = self.eps if eps is None else eps
        if metric_aware is None:
            metric_aware = self._use_surface_distance_geometry()
        if normalize_metric is None:
            normalize_metric = self.normalize_metric_voronoi_distance

        diff = points_uv[:, None, :] - seeds[None, :, :]
        diff = self._wrap_point_seed_diff(diff, points_face_id, seed_face_id)

        if not metric_aware:
            d = torch.linalg.norm(diff, dim=-1)
        else:
            if point_metric is None and Xu is not None and Xv is not None:
                point_metric, _ = self._point_metric_matrix(
                    Xu=Xu,
                    Xv=Xv,
                    normalize_metric=normalize_metric,
                    eps=eps_t,
                )
            if point_metric is not None:
                metric = point_metric[:, None, :, :]
            elif seed_metric is not None:
                metric = seed_metric[None, :, :, :]
            else:
                raise ValueError("point_metric, seed_metric, or Xu/Xv is required for metric-aware point-seed distance")
            d = self._metric_distance_from_diff(diff, metric, eps=eps_t)

        if mask_cross_face and points_face_id is not None and seed_face_id is not None:
            points_face_id = points_face_id.to(device=points_uv.device, dtype=torch.long)
            seed_face_id = seed_face_id.to(device=points_uv.device, dtype=torch.long)
            cross_face = points_face_id[:, None] != seed_face_id[None, :]
            d = d + cross_face.to(d.dtype) * 1e6

        if return_diff:
            return d, diff
        return d

    def _seed_seed_distance_metric(
        self,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
        seed_metric: torch.Tensor | None = None,
        metric_aware: bool | None = None,
        eps: float | None = None,
        return_diff: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        eps_t = self.eps if eps is None else eps
        if metric_aware is None:
            metric_aware = self._use_surface_distance_geometry() and seed_metric is not None

        diff = seeds.unsqueeze(0) - seeds.unsqueeze(1)
        diff = self._wrap_seed_seed_diff(diff, seed_face_id=seed_face_id)

        if metric_aware:
            if seed_metric is None:
                raise ValueError("seed_metric is required for metric-aware seed-seed distance")
            metric = 0.5 * (seed_metric[:, None, :, :] + seed_metric[None, :, :, :])
            d = self._metric_distance_from_diff(diff, metric, eps=eps_t)
        else:
            d = torch.linalg.norm(diff, dim=-1)

        if return_diff:
            return d, diff
        return d

    def _point_point_distance_metric(
        self,
        points_a: torch.Tensor,
        points_b: torch.Tensor,
        point_metric_a: torch.Tensor | None = None,
        point_metric_b: torch.Tensor | None = None,
        points_a_face_id: torch.Tensor | None = None,
        points_b_face_id: torch.Tensor | None = None,
        metric_aware: bool | None = None,
        eps: float | None = None,
        mask_cross_face: bool = False,
    ) -> torch.Tensor:
        eps_t = self.eps if eps is None else eps
        if metric_aware is None:
            metric_aware = self._use_surface_distance_geometry()

        diff = points_a[:, None, :] - points_b[None, :, :]
        diff = self._wrap_point_seed_diff(
            diff,
            points_face_id=points_a_face_id,
            seed_face_id=points_b_face_id,
        )

        if metric_aware:
            if point_metric_a is None and point_metric_b is None:
                raise ValueError("point_metric_a or point_metric_b is required for metric-aware point-point distance")
            if point_metric_a is not None and point_metric_b is not None:
                metric = 0.5 * (point_metric_a[:, None, :, :] + point_metric_b[None, :, :, :])
            elif point_metric_a is not None:
                metric = point_metric_a[:, None, :, :]
            else:
                metric = point_metric_b[None, :, :, :]
            d = self._metric_distance_from_diff(diff, metric, eps=eps_t)
        else:
            d = torch.linalg.norm(diff, dim=-1)

        if mask_cross_face and points_a_face_id is not None and points_b_face_id is not None:
            points_a_face_id = points_a_face_id.to(device=points_a.device, dtype=torch.long)
            points_b_face_id = points_b_face_id.to(device=points_a.device, dtype=torch.long)
            cross_face = points_a_face_id[:, None] != points_b_face_id[None, :]
            d = d + cross_face.to(d.dtype) * 1e6
        return d

    def _sample_point_metric_at_uv(
        self,
        samples_uv: torch.Tensor,
        points_uv: torch.Tensor,
        point_metric: torch.Tensor,
        samples_face_id: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d = self._point_seed_distance_metric(
            points_uv=samples_uv,
            seeds=points_uv,
            metric_aware=False,
            points_face_id=samples_face_id,
            seed_face_id=points_face_id,
            mask_cross_face=samples_face_id is not None and points_face_id is not None,
        )
        nn = d.argmin(dim=1)
        return point_metric.index_select(0, nn)

    def _point_metric_matrix(
        self,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        normalize_metric: bool = True,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E, Fm, Gm = self._surface_metric_components(Xu, Xv)

        metric = torch.zeros(
            Xu.shape[0],
            2,
            2,
            device=Xu.device,
            dtype=Xu.dtype,
        )
        metric[:, 0, 0] = E
        metric[:, 0, 1] = Fm
        metric[:, 1, 0] = Fm
        metric[:, 1, 1] = Gm

        local_scale = torch.sqrt(0.5 * (E + Gm)).clamp_min(eps)
        scale = local_scale.mean().clamp_min(eps)
        if normalize_metric:
            metric = metric / scale.square().clamp_min(eps)

        return metric, scale

    def _pairwise_uv_dirs(
        self,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v = seeds.unsqueeze(0) - seeds.unsqueeze(1)
        seed_face_id = self._seed_face_id_for(seeds, seed_face_id=seed_face_id)
        same_face = seed_face_id[:, None] == seed_face_id[None, :]

        uper_face = self.face_u_periodic[seed_face_id]
        vper_face = self.face_v_periodic[seed_face_id]

        uper_pair = uper_face[:, None] & uper_face[None, :] & same_face
        vper_pair = vper_face[:, None] & vper_face[None, :] & same_face

        du = v[..., 0]
        dv = v[..., 1]

        du = du - torch.round(du) * uper_pair.to(du.dtype)
        dv = dv - torch.round(dv) * vper_pair.to(dv.dtype)

        v[..., 0] = du
        v[..., 1] = dv

        t = torch.stack([-v[..., 1], v[..., 0]], dim=-1)
        n = torch.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)
        return t / n

    # -------------------- fiber helpers --------------------

    def _strict_upper_tri_mask(self, S: int, device, dtype) -> torch.Tensor:
        return torch.triu(torch.ones(S, S, device=device, dtype=dtype), diagonal=1)

    def _soft_pair_weights(
        self,
        weights: torch.Tensor,
        seeds: torch.Tensor | None = None,
        seed_metric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, S = weights.shape
        pair = weights.unsqueeze(2) * weights.unsqueeze(1)
        if seeds is None:
            pair_mask = self._strict_upper_tri_mask(S, weights.device, weights.dtype)
        else:
            pair_mask = self._pair_distinctness(
                seeds=seeds,
                device=weights.device,
                dtype=weights.dtype,
                seed_metric=seed_metric,
            )
        pair = pair * pair_mask.unsqueeze(0)
        denom = pair.sum(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        return pair / denom

    def _normalize_upper_tri_pair_weights(self, pair_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_weights.ndim != 3 or pair_weights.shape[1] != pair_weights.shape[2]:
            raise ValueError(f"pair_weights must be (N,S,S), got {tuple(pair_weights.shape)}")

        N, S, _ = pair_weights.shape
        tri = self._strict_upper_tri_mask(S, pair_weights.device, pair_weights.dtype).unsqueeze(0)
        pair = pair_weights * tri
        pair = pair.clamp_min(0.0)

        raw_sum = pair.sum(dim=(1, 2), keepdim=True)
        ok = raw_sum > self.eps
        pair_norm = pair / raw_sum.clamp_min(self.eps)
        return pair_norm, ok.expand(N, S, S)

    def _axial_tensor_from_pair_weights(
        self,
        pair_weights: torch.Tensor,
        seeds: torch.Tensor,
        seed_face_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pair_weights = torch.nan_to_num(pair_weights, nan=0.0, posinf=0.0, neginf=0.0)
        t_ij = self._pairwise_uv_dirs(seeds, seed_face_id=seed_face_id)               # (S,S,2)
        t_ij = torch.nan_to_num(t_ij, nan=0.0, posinf=0.0, neginf=0.0)
        Q_ij = t_ij.unsqueeze(-1) * t_ij.unsqueeze(-2)     # (S,S,2,2)
        return (pair_weights.unsqueeze(-1).unsqueeze(-1) * Q_ij.unsqueeze(0)).sum(dim=(1, 2))

    def _axial_tensor_from_local_pair_tangents(
        self,
        pair_weights: torch.Tensor,
        pair_tangent_ij: torch.Tensor,
    ) -> torch.Tensor:
        pair_weights = torch.nan_to_num(pair_weights, nan=0.0, posinf=0.0, neginf=0.0)
        pair_tangent_ij = torch.nan_to_num(pair_tangent_ij, nan=0.0, posinf=0.0, neginf=0.0)
        Q_ij = pair_tangent_ij.unsqueeze(-1) * pair_tangent_ij.unsqueeze(-2)
        return (pair_weights.unsqueeze(-1).unsqueeze(-1) * Q_ij).sum(dim=(1, 2))

    def _principal_axial_direction(self, Q: torch.Tensor) -> torch.Tensor:
        Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        Q = 0.5 * (Q + Q.transpose(-1, -2))

        q00 = Q[..., 0, 0]
        q01 = Q[..., 0, 1]
        q11 = Q[..., 1, 1]
        trace = q00 + q11
        diff = q00 - q11
        gap = torch.sqrt(diff * diff + 4.0 * q01 * q01 + self.eps)
        lambda_max = 0.5 * (trace + gap)

        v1 = torch.stack([q01, lambda_max - q00], dim=-1)
        v2 = torch.stack([lambda_max - q11, q01], dim=-1)
        v1_norm = torch.sqrt((v1 * v1).sum(dim=-1, keepdim=True) + self.eps)
        v2_norm = torch.sqrt((v2 * v2).sum(dim=-1, keepdim=True) + self.eps)
        use_v1 = v1_norm >= v2_norm
        t_uv = torch.where(use_v1, v1, v2)

        # Isotropic/zero tensors do not have a meaningful principal direction.
        fallback = torch.zeros_like(t_uv)
        fallback[..., 0] = 1.0
        t_norm = torch.sqrt((t_uv * t_uv).sum(dim=-1, keepdim=True) + self.eps)
        t_uv = t_uv / t_norm
        has_orientation = trace.reshape(*trace.shape, 1) > self.eps
        return torch.where(has_orientation, t_uv, torch.zeros_like(t_uv))

    def _axial_coherence_from_tensor(self, Q: torch.Tensor) -> torch.Tensor:
        Q = torch.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        if Q.shape[-1] == 3:
            eigvals = torch.linalg.eigvalsh(Q).clamp_min(0.0)
            trace = eigvals.sum(dim=-1).clamp_min(self.eps)
            return ((eigvals[..., -1] - eigvals[..., -2]) / trace).clamp(0.0, 1.0)

        q00 = Q[..., 0, 0]
        q01 = Q[..., 0, 1]
        q11 = Q[..., 1, 1]
        trace = (q00 + q11).clamp_min(self.eps)
        diff = q00 - q11
        gap = torch.sqrt(diff * diff + 4.0 * q01 * q01 + self.eps)
        return (gap / trace).clamp(0.0, 1.0)

    def _blended_uv_fiber_axial(
        self,
        weights: torch.Tensor,
        seeds: torch.Tensor,
        pair_weights: torch.Tensor | None = None,
        normalize_pair_weights: bool = True,
        seed_face_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pair_weights is None:
            pair_weights = self._soft_pair_weights(weights, seeds=seeds)
        elif normalize_pair_weights:
            pair_weights, _ = self._normalize_upper_tri_pair_weights(pair_weights)
        else:
            S = pair_weights.shape[1]
            tri = self._strict_upper_tri_mask(S, pair_weights.device, pair_weights.dtype).unsqueeze(0)
            pair_weights = pair_weights.clamp_min(0.0) * tri

        Q = self._axial_tensor_from_pair_weights(pair_weights, seeds, seed_face_id=seed_face_id)
        t_uv = self._principal_axial_direction(Q)
        return t_uv, Q, pair_weights

    def _blended_uv_fiber(self, weights: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        # Backward-compatible wrapper. Uses axial blending so (-t) and t
        # are treated as the same fiber direction.
        t_uv, _, _ = self._blended_uv_fiber_axial(weights, seeds)
        return t_uv

    def _fiber_pair_weights(
        self,
        w_soft: torch.Tensor,
        seeds: torch.Tensor,
        band_ij: torch.Tensor | None = None,
        pair_relevance: torch.Tensor | None = None,
        seed_active_weights: torch.Tensor | None = None,
        seed_duplicate_weights: torch.Tensor | None = None,
        seed_domain_weights: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
        seed_metric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seed_active_weights is None:
            soft_pair = self._soft_pair_weights(w_soft, seeds=seeds, seed_metric=seed_metric)
            if not self.use_band_weighted_fiber_pairs:
                return soft_pair

            if band_ij is None or pair_relevance is None:
                return soft_pair
        else:
            S = w_soft.shape[1]
            if seed_active_weights.ndim != 1 or seed_active_weights.shape[0] != S:
                raise ValueError(
                    f"seed_active_weights must have shape ({S},), got {tuple(seed_active_weights.shape)}"
                )
            g = seed_active_weights.to(device=w_soft.device, dtype=w_soft.dtype).clamp(0.0, 1.0)
            if seed_duplicate_weights is None:
                duplicate_activity = g
            else:
                if seed_duplicate_weights.ndim != 1 or seed_duplicate_weights.shape[0] != S:
                    raise ValueError(
                        f"seed_duplicate_weights must have shape ({S},), "
                        f"got {tuple(seed_duplicate_weights.shape)}"
                    )
                duplicate_activity = seed_duplicate_weights.to(
                    device=w_soft.device,
                    dtype=w_soft.dtype,
                ).clamp(0.0, 1.0)
            if seed_domain_weights is None:
                domain_activity = g
            else:
                if seed_domain_weights.ndim != 1 or seed_domain_weights.shape[0] != S:
                    raise ValueError(
                        f"seed_domain_weights must have shape ({S},), "
                        f"got {tuple(seed_domain_weights.shape)}"
                    )
                domain_activity = seed_domain_weights.to(
                    device=w_soft.device,
                    dtype=w_soft.dtype,
                ).clamp(0.0, 1.0)
            pair_activity = (
                (domain_activity[:, None] * domain_activity[None, :]).pow(float(self.domain_pair_power))
                * (duplicate_activity[:, None] * duplicate_activity[None, :]).pow(
                    float(self.duplicate_pair_power)
                )
            )
            pair_mask = self._pair_distinctness(
                seeds=seeds,
                device=w_soft.device,
                dtype=w_soft.dtype,
                seed_face_id=seed_face_id,
                seed_metric=seed_metric,
            )
            raw_pair = (
                w_soft.unsqueeze(2)
                * w_soft.unsqueeze(1)
                * pair_mask.unsqueeze(0)
                * pair_activity.unsqueeze(0)
            )
            if not self.use_band_weighted_fiber_pairs or band_ij is None or pair_relevance is None:
                return raw_pair

        # Prefer pairs whose visible band is present at this point, but keep a
        # small soft-pair floor so clipped ends/junctions do not jump abruptly.
        band_prior = band_ij.clamp(0.0, 1.0).pow(float(self.fiber_band_prior_power))
        floor = torch.as_tensor(
            self.fiber_band_prior_floor,
            device=band_prior.device,
            dtype=band_prior.dtype,
        )
        band_prior = floor + (1.0 - floor) * band_prior
        raw_pair = soft_pair * band_prior if seed_active_weights is None else raw_pair * band_prior
        if seed_active_weights is not None:
            return raw_pair
        pair_norm, ok_mask = self._normalize_upper_tri_pair_weights(raw_pair)
        return torch.where(ok_mask, pair_norm, soft_pair)

    def _estimate_boundary_sample_tangents_uv(
        self,
        boundary_uv: torch.Tensor,
        boundary_face_id: torch.Tensor | None = None,
        boundary_metric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_uv.numel() == 0:
            return torch.zeros_like(boundary_uv)

        B = boundary_uv.shape[0]
        if B < 2:
            return torch.zeros_like(boundary_uv)

        if boundary_face_id is None:
            boundary_face_id = torch.zeros(B, device=boundary_uv.device, dtype=torch.long)
        elif boundary_face_id.dtype != torch.long:
            boundary_face_id = boundary_face_id.to(torch.long)

        dmat = self._point_point_distance_metric(
            points_a=boundary_uv,
            points_b=boundary_uv,
            point_metric_a=boundary_metric,
            point_metric_b=boundary_metric,
            points_a_face_id=boundary_face_id,
            points_b_face_id=boundary_face_id,
            metric_aware=self._use_surface_distance_geometry() and boundary_metric is not None,
            mask_cross_face=True,
        )

        same_face = boundary_face_id[:, None] == boundary_face_id[None, :]
        eye = torch.eye(B, device=boundary_uv.device, dtype=torch.bool)
        valid_neighbor = same_face & (~eye)
        dmat = torch.where(valid_neighbor, dmat, torch.full_like(dmat, 1e6))

        k = min(max(1, self.boundary_knn_k), max(1, B - 1))
        d_knn, idx_knn = torch.topk(dmat, k=k, dim=1, largest=False)
        valid_knn = d_knn < 1e5

        all_diff = boundary_uv.unsqueeze(1) - boundary_uv.unsqueeze(0)
        all_diff = self._wrap_point_seed_diff(
            all_diff,
            points_face_id=boundary_face_id,
            seed_face_id=boundary_face_id,
        )
        local_diff = all_diff.gather(
            1,
            idx_knn.unsqueeze(-1).expand(-1, -1, 2),
        )
        sigma = d_knn[..., 0].clamp_min(self.boundary_softmin_tau)
        w = torch.exp(-0.5 * (d_knn / sigma.unsqueeze(1).clamp_min(self.eps)).pow(2))
        w = w * valid_knn.to(w.dtype)

        cov = (
            w.unsqueeze(-1).unsqueeze(-1)
            * (local_diff.unsqueeze(-1) * local_diff.unsqueeze(-2))
        ).sum(dim=1)
        cov = cov / w.sum(dim=1, keepdim=True).unsqueeze(-1).clamp_min(self.eps)

        tangent = self._principal_axial_direction(cov)
        has_support = valid_knn.any(dim=1, keepdim=True)
        return torch.where(has_support, tangent, torch.zeros_like(tangent))

    def _boundary_tangent_tensor_field(
        self,
        points_uv: torch.Tensor,
        boundary_uv: torch.Tensor,
        points_face_id: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        point_metric: torch.Tensor | None = None,
        boundary_metric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_uv.numel() == 0:
            return torch.zeros(
                (points_uv.shape[0], 2, 2),
                device=points_uv.device,
                dtype=points_uv.dtype,
            )

        B = boundary_uv.shape[0]
        if points_face_id is None:
            points_face_id = torch.zeros(points_uv.shape[0], device=points_uv.device, dtype=torch.long)
        elif points_face_id.dtype != torch.long:
            points_face_id = points_face_id.to(torch.long)

        if boundary_face_id is None:
            boundary_face_id = torch.zeros(B, device=boundary_uv.device, dtype=torch.long)
        elif boundary_face_id.dtype != torch.long:
            boundary_face_id = boundary_face_id.to(torch.long)

        tangent_uv = self._estimate_boundary_sample_tangents_uv(
            boundary_uv=boundary_uv,
            boundary_face_id=boundary_face_id,
            boundary_metric=boundary_metric,
        )
        sample_Q = tangent_uv.unsqueeze(-1) * tangent_uv.unsqueeze(-2)

        bb_dmat = self._point_point_distance_metric(
            points_a=boundary_uv,
            points_b=boundary_uv,
            point_metric_a=boundary_metric,
            point_metric_b=boundary_metric,
            points_a_face_id=boundary_face_id,
            points_b_face_id=boundary_face_id,
            metric_aware=self._use_surface_distance_geometry() and boundary_metric is not None,
            mask_cross_face=True,
        )
        same_face_bb = boundary_face_id[:, None] == boundary_face_id[None, :]
        eye = torch.eye(B, device=boundary_uv.device, dtype=torch.bool)
        bb_valid = same_face_bb & (~eye)
        bb_dmat = torch.where(bb_valid, bb_dmat, torch.full_like(bb_dmat, 1e6))
        local_scale = bb_dmat.min(dim=1).values
        local_scale = torch.where(
            local_scale < 1e5,
            local_scale,
            torch.full_like(local_scale, self.boundary_softmin_tau),
        ).clamp_min(self.boundary_softmin_tau)

        pb_dmat = self._point_seed_distance_metric(
            points_uv=points_uv,
            seeds=boundary_uv,
            point_metric=point_metric,
            seed_metric=boundary_metric,
            points_face_id=points_face_id,
            seed_face_id=boundary_face_id,
            metric_aware=self._use_surface_distance_geometry()
            and (point_metric is not None or boundary_metric is not None),
            mask_cross_face=True,
        )
        same_face_pb = points_face_id[:, None] == boundary_face_id[None, :]
        pb_dmat = torch.where(same_face_pb, pb_dmat, torch.full_like(pb_dmat, 1e6))

        weights = torch.exp(
            -0.5 * (pb_dmat / local_scale.unsqueeze(0).clamp_min(self.eps)).pow(2)
        )
        weights = weights * same_face_pb.to(weights.dtype)

        weight_sum = weights.sum(dim=1, keepdim=True)
        # Shell boundaries should inject tangent-aligned line directions, but
        # only where the shell-boundary field is active; away from the boundary
        # the Voronoi interior tensor remains in control.
        Q_boundary = (weights.unsqueeze(-1).unsqueeze(-1) * sample_Q.unsqueeze(0)).sum(dim=1)
        Q_boundary = Q_boundary / weight_sum.unsqueeze(-1).clamp_min(self.eps)

        has_support = weight_sum.squeeze(1) > self.eps
        return torch.where(has_support[:, None, None], Q_boundary, torch.zeros_like(Q_boundary))

    def map_to_3d(self, t_uv: torch.Tensor, Xu: torch.Tensor, Xv: torch.Tensor, eps: float = 1e-8):
        T = t_uv[:, 0:1] * Xu + t_uv[:, 1:2] * Xv
        return F.normalize(T, dim=1, eps=eps)

    # -------------------- boundary band --------------------

    def boundary_attachment_field(
        self,
        points_uv: torch.Tensor,
        boundary_uv: torch.Tensor | None,
        points_face_id: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        boundary_width_raw: torch.Tensor | None = None,
        boundary_beta_raw: torch.Tensor | None = None,
        point_metric: torch.Tensor | None = None,
        boundary_metric: torch.Tensor | None = None,
        alpha_union: float = 8.0,
    ) -> torch.Tensor:
        if boundary_uv is None or boundary_uv.numel() == 0:
            return torch.zeros(
                points_uv.shape[0],
                device=points_uv.device,
                dtype=points_uv.dtype,
            )

        dmat = self._point_seed_distance_metric(
            points_uv=points_uv,
            seeds=boundary_uv,
            point_metric=point_metric,
            seed_metric=boundary_metric,
            points_face_id=points_face_id,
            seed_face_id=boundary_face_id,
            metric_aware=self._use_surface_distance_geometry()
            and (point_metric is not None or boundary_metric is not None),
            mask_cross_face=True,
        )

        k = min(self.boundary_knn_k, int(dmat.shape[1]))
        d_knn = torch.topk(dmat, k=k, dim=1, largest=False).values

        tau = torch.as_tensor(self.boundary_softmin_tau, device=dmat.device, dtype=dmat.dtype)
        dmin = -tau * torch.logsumexp(-d_knn / (tau + self.eps), dim=1) + tau * math.log(k)

        tb = self.boundary_width(points_uv, boundary_width_raw=boundary_width_raw)
        bb = self.boundary_beta(points_uv, boundary_beta_raw=boundary_beta_raw)
        if k > 1 and self.boundary_spacing_blend > 0.0 and boundary_uv.shape[0] > 1:
            b2b = self._point_point_distance_metric(
                points_a=boundary_uv,
                points_b=boundary_uv,
                point_metric_a=boundary_metric,
                point_metric_b=boundary_metric,
                points_a_face_id=boundary_face_id,
                points_b_face_id=boundary_face_id,
                metric_aware=self._use_surface_distance_geometry() and boundary_metric is not None,
                mask_cross_face=True,
            )
            big = torch.eye(boundary_uv.shape[0], device=b2b.device, dtype=b2b.dtype) * 1e6
            b2b = b2b + big
            h_boundary = b2b.min(dim=1).values.median()
            bb = bb + self.boundary_spacing_blend * h_boundary

        rho_b_raw = torch.sigmoid((tb - dmin) / (bb + self.eps))
        norm = torch.sigmoid(tb / (bb + self.eps))
        rho_b_norm = (rho_b_raw / (norm + self.eps)).clamp(0.0, 1.0)

        rho_b = 1.0 - torch.exp(-alpha_union * rho_b_norm)
        return rho_b.clamp(0.0, 1.0)

    def boundary_centerline_thickness_density(
        self,
        points_xyz: torch.Tensor,
        points_uv: torch.Tensor,
        boundary_uv: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        boundary_width_raw: torch.Tensor | None = None,
        boundary_beta_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError(f"points_xyz must be (N,3), got {tuple(points_xyz.shape)}")
        if points_uv.ndim != 2 or points_uv.shape[0] != points_xyz.shape[0]:
            raise ValueError(
                "points_uv must be (N,2) matching points_xyz, got "
                f"{tuple(points_uv.shape)} and {tuple(points_xyz.shape)}"
            )

        device = points_xyz.device
        dtype = points_xyz.dtype
        N = points_xyz.shape[0]

        core_idx = self.boundary_solid_idx.to(device=device, dtype=torch.long)
        core_idx = core_idx[(core_idx >= 0) & (core_idx < N)]

        if core_idx.numel() == 0 and boundary_uv is not None and boundary_uv.numel() > 0:
            boundary_uv_t = boundary_uv.to(device=points_uv.device, dtype=points_uv.dtype)
            d_uv = torch.cdist(boundary_uv_t, points_uv)
            if (
                points_face_id is not None
                and boundary_face_id is not None
                and boundary_face_id.numel() == boundary_uv_t.shape[0]
            ):
                p_face = points_face_id.to(device=device, dtype=torch.long)
                b_face = boundary_face_id.to(device=device, dtype=torch.long)
                same_face = b_face[:, None] == p_face[None, :]
                d_uv = torch.where(same_face, d_uv, torch.full_like(d_uv, 1e6))
            core_idx = torch.argmin(d_uv, dim=1)

        if core_idx.numel() == 0:
            return torch.zeros(N, device=device, dtype=dtype)

        core_idx = torch.unique(core_idx)
        core_xyz = points_xyz.index_select(0, core_idx)
        d_core = torch.cdist(points_xyz, core_xyz).min(dim=1).values

        radius = self.boundary_width(points_xyz, boundary_width_raw=boundary_width_raw)
        beta = self.boundary_beta(points_xyz, boundary_beta_raw=boundary_beta_raw)
        rho = torch.sigmoid((radius - d_core) / (beta + self.eps))
        return rho.clamp(0.0, 1.0)

    def smooth_union(
        self,
        rho_a: torch.Tensor,
        rho_b: torch.Tensor,
        alpha_b: float | torch.Tensor,
    ) -> torch.Tensor:
        alpha_b = torch.as_tensor(alpha_b, device=rho_a.device, dtype=rho_a.dtype)
        rho = 1.0 - (1.0 - rho_a) * (1.0 - alpha_b * rho_b)
        return rho.clamp(0.0, 1.0)

    def soft_project_density(self, rho: torch.Tensor) -> torch.Tensor:
        strength = float(self.density_projection_strength)
        if strength <= 0.0:
            return rho

        threshold = torch.as_tensor(
            self.density_projection_threshold,
            device=rho.device,
            dtype=rho.dtype,
        )
        gamma = torch.as_tensor(
            self.density_projection_gamma,
            device=rho.device,
            dtype=rho.dtype,
        ).clamp_min(self.eps)
        rho_proj = torch.sigmoid((rho - threshold) / gamma)
        rho_blend = (1.0 - strength) * rho + strength * rho_proj
        return rho_blend.clamp(0.0, 1.0)

    def centerline_radius(
        self,
        radius_raw: torch.Tensor | None = None,
        radius_fixed: float | torch.Tensor | None = None,
        ref_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ref_tensor is None:
            if radius_raw is not None:
                device = radius_raw.device
                dtype = radius_raw.dtype
            elif isinstance(radius_fixed, torch.Tensor):
                device = radius_fixed.device
                dtype = radius_fixed.dtype
            else:
                device = None
                dtype = torch.float32
        else:
            device = ref_tensor.device
            dtype = ref_tensor.dtype

        fixed_value = self.centerline_radius_fixed if radius_fixed is None else radius_fixed
        if fixed_value is not None:
            return torch.as_tensor(fixed_value, device=device, dtype=dtype)

        if radius_raw is None:
            return torch.as_tensor(
                0.5 * (self.centerline_radius_min + self.centerline_radius_max),
                device=device,
                dtype=dtype,
            )
        else:
            radius_raw_t = radius_raw.to(device=device, dtype=dtype)

        r_min = torch.as_tensor(
            self.centerline_radius_min,
            device=device,
            dtype=dtype,
        )
        r_max = torch.as_tensor(
            self.centerline_radius_max,
            device=device,
            dtype=dtype,
        )
        return r_min + (r_max - r_min) * torch.sigmoid(radius_raw_t)

    def centerline_thickness_density(
        self,
        points_xyz: torch.Tensor,
        skeleton_field: torch.Tensor,
        radius_raw: torch.Tensor | None = None,
        radius_fixed: float | torch.Tensor | None = None,
        threshold: float | None = None,
        threshold_softness: float | None = None,
        softmin_tau: float | None = None,
        beta: float | None = None,
        debugging = False,
    ) -> torch.Tensor:

        def stats(name, x):
            x_det = x.detach()
            finite = x_det[torch.isfinite(x_det)]
            if finite.numel() == 0:
                print(f"{name}: no finite values")
                return
            print(
                f"{name}: "
                f"min={finite.min().item():.6g}, "
                f"mean={finite.mean().item():.6g}, "
                f"max={finite.max().item():.6g}, "
                f"numel={finite.numel()}"
            )
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError(f"points_xyz must be (N,3), got {tuple(points_xyz.shape)}")

        if skeleton_field.ndim != 1 or skeleton_field.shape[0] != points_xyz.shape[0]:
            raise ValueError(
                "skeleton_field must be (N,) matching points_xyz, got "
                f"{tuple(skeleton_field.shape)} and {tuple(points_xyz.shape)}"
            )

        device = points_xyz.device
        dtype = points_xyz.dtype

        skeleton = skeleton_field.to(device=device, dtype=dtype).clamp(0.0, 1.0)

        threshold_t = torch.as_tensor(
            self.centerline_threshold if threshold is None else threshold,
            device=device,
            dtype=dtype,
        )

        threshold_softness_t = torch.as_tensor(
            self.centerline_threshold_softness if threshold_softness is None else threshold_softness,
            device=device,
            dtype=dtype,
        ).clamp_min(self.eps)

        tau_t = torch.as_tensor(
            self.centerline_softmin_tau if softmin_tau is None else softmin_tau,
            device=device,
            dtype=dtype,
        ).clamp_min(self.eps)

        beta_t = torch.as_tensor(
            self.centerline_beta if beta is None else beta,
            device=device,
            dtype=dtype,
        ).clamp_min(self.eps)

        radius = self.centerline_radius(
            radius_raw,
            radius_fixed=radius_fixed,
            ref_tensor=points_xyz,
        )

        if debugging:
            print("\n========== centerline_thickness_density DEBUG ==========")
            print(f"points_xyz shape: {tuple(points_xyz.shape)}")
            print(f"skeleton_field shape: {tuple(skeleton_field.shape)}")
            print(f"threshold: {threshold_t.item():.6g}")
            print(f"threshold_softness: {threshold_softness_t.item():.6g}")
            print(f"softmin_tau: {tau_t.item():.6g}")
            print(f"centerline_beta: {beta_t.item():.6g}")
            print(f"radius: {radius.detach().item() if radius.numel() == 1 else radius.detach()}")

            stats("points_xyz[:,0]", points_xyz[:, 0])
            stats("points_xyz[:,1]", points_xyz[:, 1])
            stats("points_xyz[:,2]", points_xyz[:, 2])
            stats("skeleton", skeleton)

        # Soft core extraction
        core_weight = torch.sigmoid(
            (skeleton - threshold_t) / threshold_softness_t
        ).clamp(0.0, 1.0)



        # Pairwise 3D distances
        D = torch.cdist(points_xyz, points_xyz)

        if debugging:

            stats("core_weight", core_weight)

            active_soft = (core_weight > 0.5).sum()
            active_strong = (core_weight > 0.9).sum()
            print(f"core_weight > 0.5 count: {int(active_soft.item())}")
            print(f"core_weight > 0.9 count: {int(active_strong.item())}")

            stats("D", D)

        # Hard nearest-core distance for debugging
        hard_mask = core_weight > 0.5
        if bool(hard_mask.any()):
            core_xyz_hard = points_xyz[hard_mask]
            D_hard = torch.cdist(points_xyz, core_xyz_hard)
            d_core_hard = D_hard.min(dim=1).values
            if debugging:
              stats("d_core_hard", d_core_hard)
        else:
            d_core_hard = torch.full(
                (points_xyz.shape[0],),
                float("inf"),
                device=device,
                dtype=dtype,
            )
            
            print("WARNING: no hard core points with core_weight > 0.5")

        # Normalized weighted soft-min distance
        core_mask = core_weight > 0.5

        if bool(core_mask.any()):
            D_core = D[:, core_mask]

            d_core_soft = -tau_t * torch.logsumexp(
                -D_core / tau_t,
                dim=1
            )

            d_core_soft = d_core_soft.clamp_min(0.0)
        else:
            d_core_soft = torch.full(
                (points_xyz.shape[0],),
                float("inf"),
                device=device,
                dtype=dtype,
            )
        d_core_soft = d_core_soft.clamp_min(0.0)
        if debugging:
           stats("d_core_soft_normalized", d_core_soft)

        # Compare soft and hard distances
        if bool(hard_mask.any()):
            diff_soft_hard = d_core_soft - d_core_hard
            if debugging:
               stats("d_core_soft - d_core_hard", diff_soft_hard)

        # Use soft distance for final density
        d_core = d_core_soft

        rho = torch.sigmoid((radius - d_core) / beta_t).clamp(0.0, 1.0)

        if debugging:

            stats("rho_centerline", rho)

            print("rho > 0.1 count:", int((rho > 0.1).sum().item()))
            print("rho > 0.5 count:", int((rho > 0.5).sum().item()))
            print("rho > 0.9 count:", int((rho > 0.9).sum().item()))
            print("=======================================================\n")

        self._last_centerline_diagnostics = {
            "skeleton_field": skeleton.detach(),
            "rho_centerline": rho.detach(),
            "centerline_core_weight": core_weight.detach(),
            "centerline_d_core": d_core.detach(),
            "centerline_d_core_hard": d_core_hard.detach(),
            "centerline_radius": radius.detach(),
            "skeleton_field_min_mean_max": self._distance_stats_tensor(skeleton),
            "core_weight_min_mean_max": self._distance_stats_tensor(core_weight),
            "d_core_min_mean_max": self._distance_stats_tensor(d_core),
            "d_core_hard_min_mean_max": self._distance_stats_tensor(d_core_hard),
            "radius": radius.detach(),
            "rho_centerline_min_mean_max": self._distance_stats_tensor(rho),
        }


        return rho

    # -------------------- higher-order helpers --------------------

    def _triple_junction_score(self, w_soft: torch.Tensor) -> torch.Tensor:
        N, S = w_soft.shape
        if S < 3:
            return torch.zeros(N, device=w_soft.device, dtype=w_soft.dtype)

        # Sum over i<j<k of (w_i w_j w_k)^p without forming an N x S x S x S tensor.
        # Let a_i = w_i^p. Then e3(a) = sum_{i<j<k} a_i a_j a_k
        # and Newton's identity gives:
        # e3 = (p1^3 - 3 p1 p2 + 2 p3) / 6,
        # where p1=sum(a_i), p2=sum(a_i^2), p3=sum(a_i^3).
        power = float(self.junction_triple_power)
        a = w_soft if power == 1.0 else w_soft.pow(power)

        p1 = a.sum(dim=1)
        p2 = (a * a).sum(dim=1)
        p3 = (a * a * a).sum(dim=1)
        e3 = (p1 * p1 * p1 - 3.0 * p1 * p2 + 2.0 * p3) / 6.0
        return e3.clamp_min(0.0)
    
        # -------------------- bisector band density --------------------
    def _bisector_band_density(
        self,
        points: torch.Tensor,
        seeds: torch.Tensor,
        d: torch.Tensor,
        w_soft: torch.Tensor,
        w_geo: torch.Tensor,
        beta: float | torch.Tensor,
        metric: torch.Tensor,
        metric_mode: str,
        seed_active_weights: torch.Tensor | None = None,
        seed_duplicate_weights: torch.Tensor | None = None,
        seed_domain_weights: torch.Tensor | None = None,
        hard_seed_mask: bool = True,
        seed_face_id: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
        point_metric: torch.Tensor | None = None,
        seed_metric: torch.Tensor | None = None,
        skeleton_sigma: float | torch.Tensor | None = None,
    ):
        N, S = d.shape
        device = d.device
        dtype = d.dtype

        w_struct = w_soft

        seed_activity = torch.ones(S, device=device, dtype=dtype)
        duplicate_activity = torch.ones(S, device=device, dtype=dtype)
        domain_activity = torch.ones(S, device=device, dtype=dtype)
        active_seed = torch.ones(S, device=device, dtype=torch.bool)

        if seed_active_weights is not None:
            seed_activity = seed_active_weights.to(device=device, dtype=dtype).clamp(0.0, 1.0)

            duplicate_activity = (
                seed_activity
                if seed_duplicate_weights is None
                else seed_duplicate_weights.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            )

            domain_activity = (
                seed_activity
                if seed_domain_weights is None
                else seed_domain_weights.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            )

            if hard_seed_mask:
                active_seed = seed_activity > 0.0

            w_struct = w_soft * seed_activity.unsqueeze(0)
            w_sum = w_struct.sum(dim=1, keepdim=True)
            w_struct = torch.where(
                w_sum > self.eps,
                w_struct / w_sum.clamp_min(self.eps),
                w_soft,
            )

        pair_activity = (
            (domain_activity[:, None] * domain_activity[None, :]).pow(float(self.domain_pair_power))
            *
            (duplicate_activity[:, None] * duplicate_activity[None, :]).pow(float(self.duplicate_pair_power))
        )

        if hard_seed_mask and seed_active_weights is not None:
            active_count = active_seed.to(dtype=dtype).sum()
            global_activity = (
                seed_activity[active_seed].amax().clamp(0.0, 1.0)
                if bool(active_seed.any())
                else torch.zeros((), device=device, dtype=dtype)
            )
        else:
            active_count = torch.as_tensor(float(S), device=device, dtype=dtype)
            global_activity = seed_activity.amax().clamp(0.0, 1.0)

        global_activity = global_activity.pow(float(self.global_activity_power))

        d_metric, x_minus_s = self._point_seed_distance_metric(
            points_uv=points,
            seeds=seeds,
            point_metric=point_metric if metric_mode == "point" else None,
            seed_metric=seed_metric if metric_mode == "seed" else None,
            metric_aware=metric_mode in ("point", "seed"),
            points_face_id=points_face_id,
            seed_face_id=seed_face_id,
            return_diff=True,
        )
        if d_metric.shape == d.shape:
            d = d_metric
        d_i = d.unsqueeze(2)
        d_j = d.unsqueeze(1)
        delta = d_i - d_j
        abs_delta = torch.sqrt(delta * delta + self.eps)
        metric = metric.to(device=device, dtype=dtype)

        if metric_mode == "point":
            if metric.ndim != 3 or metric.shape != (N, 2, 2):
                raise ValueError(f"point metric must be (N,2,2), got {tuple(metric.shape)}")

            grad_d = torch.einsum("nij,nsj->nsi", metric, x_minus_s)
            grad_d = grad_d / d.unsqueeze(2).clamp_min(self.eps)

            grad_vec = grad_d.unsqueeze(2) - grad_d.unsqueeze(1)

            Ginv = torch.linalg.inv(metric)
            grad_norm = torch.sqrt(
                torch.einsum(
                    "nijk,nkl,nijl->nij",
                    grad_vec,
                    Ginv,
                    grad_vec,
                ) + self.eps
            )

            grad_norm_safe = grad_norm.clamp_min(self.eps)
            true_dist = abs_delta / grad_norm_safe

        elif metric_mode == "seed":
            if metric.ndim != 3 or metric.shape != (S, 2, 2):
                raise ValueError(f"seed metric must be (S,2,2), got {tuple(metric.shape)}")

            grad_d = torch.einsum("sij,nsj->nsi", metric, x_minus_s)
            grad_d = grad_d / d.unsqueeze(2).clamp_min(self.eps)

            grad_vec = grad_d.unsqueeze(2) - grad_d.unsqueeze(1)

            grad_norm = torch.sqrt((grad_vec * grad_vec).sum(dim=-1) + self.eps)
            grad_norm_safe = grad_norm.clamp_min(self.eps)
            true_dist = abs_delta / grad_norm_safe

        else:
            raise ValueError(f"metric_mode must be 'point' or 'seed', got {metric_mode!r}")

        pair_tangent_ij = torch.stack(
            [-grad_vec[..., 1], grad_vec[..., 0]],
            dim=-1,
        )
        pair_tangent_ij = pair_tangent_ij / torch.sqrt(
            (pair_tangent_ij * pair_tangent_ij).sum(dim=-1, keepdim=True) + self.eps
        )

        pair_distinctness = self._pair_distinctness(
            seeds=seeds,
            device=device,
            dtype=dtype,
            seed_face_id=seed_face_id,
            seed_metric=seed_metric,
        )

        if hard_seed_mask and seed_active_weights is not None:
            active_pair = active_seed[:, None] & active_seed[None, :]
            pair_distinctness = pair_distinctness * active_pair.to(dtype=dtype)

        pair_validity = pair_distinctness * pair_activity

        tri = self._strict_upper_tri_mask(S, device, dtype).unsqueeze(0)
        valid_pair_mask = tri * pair_validity.unsqueeze(0)

        pair_prod = w_struct.unsqueeze(2) * w_struct.unsqueeze(1)

        # Effective seed count gate.
        # k_eff ≈ 1 inside cells, ≈2 near bisectors, >2 near junctions.
        k_eff = 1.0 / w_struct.pow(2).sum(dim=1).clamp_min(self.eps)

        k_eff_threshold = 1.25
        k_eff_softness = 0.2

        keff_gate = torch.sigmoid(
            (k_eff - k_eff_threshold) / k_eff_softness
        ).clamp(0.0, 1.0)

        if skeleton_sigma is None:
            if N > 1:
                skeleton_sigma_t = torch.sqrt(
                    ((points[1:] - points[:-1]) ** 2)
                    .sum(dim=1)
                    .mean()
                    .clamp_min(self.eps)
                )
            else:
                skeleton_sigma_t = torch.as_tensor(1e-3, device=device, dtype=dtype)
        else:
            skeleton_sigma_t = torch.as_tensor(skeleton_sigma, device=device, dtype=dtype)

        skeleton_sigma_t = skeleton_sigma_t.clamp_min(self.eps)

        skeleton_ij = torch.exp(
            -true_dist / skeleton_sigma_t
        )
        skeleton_ij = skeleton_ij * valid_pair_mask

        skeleton_pair_base = (4.0 * pair_prod).clamp(0.0, 1.0)
        pow_eps = torch.as_tensor(
            max(self.eps, torch.finfo(dtype).eps),
            device=device,
            dtype=dtype,
        )
        skeleton_pair_relevance = (
            (skeleton_pair_base + pow_eps).pow(0.25) - pow_eps.pow(0.25)
        ).clamp_min(0.0)
        skeleton_pair_relevance = skeleton_pair_relevance * keff_gate[:, None, None]
        skeleton_pair_relevance = skeleton_pair_relevance * valid_pair_mask

        skeleton_pair_strength = skeleton_ij * skeleton_pair_relevance

        skeleton_R = skeleton_pair_strength.amax(dim=(1, 2))
        skeleton_field = skeleton_R.clamp(0.0, 1.0)
        skeleton_field = (skeleton_field * global_activity).clamp(0.0, 1.0)

        ambiguity = (
            1.0 - w_struct.pow(2).sum(dim=1)
        ).clamp(0.0, 1.0)

        beta_t = torch.as_tensor(beta, device=device, dtype=dtype)

        beta_eff = beta_t * (
            1.0
            + self.junction_beta_scale
            * ambiguity.unsqueeze(1).unsqueeze(2)
        )

        w_geo_eff = w_geo.to(device=device, dtype=dtype) * (
            1.0
            + self.junction_width_bonus
            * ambiguity.unsqueeze(1).unsqueeze(2)
        )

        band_raw = torch.sigmoid(
            (w_geo_eff - true_dist) / (beta_eff + self.eps)
        )

        band_peak = torch.sigmoid(
            w_geo_eff / (beta_eff + self.eps)
        )

        band_ij = (
            band_raw / (band_peak + self.eps)
        ).clamp(0.0, 1.0)

        band_ij = band_ij * valid_pair_mask

        

        # Pair relevance:
        # - pair_prod handles local pair ownership.
        # - keff_gate kills density inside one-seed cell interiors.
        # - relevance_floor prevents broken/dotted struts near valid bisectors.
        relevance_power = 0.25
        relevance_floor = 0.250

        pair_relevance_base = (4.0 * pair_prod).clamp(0.0, 1.0)
        pair_relevance_raw = (
            (pair_relevance_base + pow_eps).pow(relevance_power)
            - pow_eps.pow(relevance_power)
        ).clamp_min(0.0)

        pair_relevance = relevance_floor + (1.0 - relevance_floor) * pair_relevance_raw
        pair_relevance = pair_relevance * keff_gate[:, None, None]
        pair_relevance = pair_relevance * valid_pair_mask

        pair_strength = band_ij * pair_relevance

        if self.pair_boost_enabled:
            active_pair_distinctness = pair_distinctness * pair_activity
            valid_pair_count = active_pair_distinctness.sum().clamp_min(1.0)
            reference_pair_count = (active_count - 1.0).clamp_min(1.0)

            pair_boost = 1.0 + self.pair_boost_strength * torch.sigmoid(
                (valid_pair_count - reference_pair_count)
                / (reference_pair_count + self.eps)
            )

            pair_strength = pair_strength * pair_boost

        R_pair = pair_strength.amax(dim=(1, 2))

        rho = R_pair.clamp(0.0, 1.0)
        rho = (rho * global_activity).clamp(0.0, 1.0)

        band_soft = band_ij.clamp(0.0, 1.0)

        eye = torch.eye(
            S,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)

        one_minus = torch.where(
            eye,
            torch.ones_like(band_soft),
            1.0 - band_soft,
        )

        edge_field = 1.0 - one_minus.prod(dim=2).prod(dim=1)
        edge_field = edge_field.clamp(0.0, 1.0)

        valid_pair_prod = pair_prod * valid_pair_mask
        debug = self.debug_metric_distances

        if debug:
            print(f"metric_mode={metric_mode}")
            self._debug_distance_stats("point_seed_dist", d)
            self._debug_distance_stats("true_dist", true_dist)
            self._debug_distance_stats("w_geo_eff", w_geo_eff)
            print("abs_delta:", abs_delta.min(), abs_delta.mean(), abs_delta.max())
            print("grad_norm_safe:", grad_norm_safe.min(), grad_norm_safe.mean(), grad_norm_safe.max())
            print("k_eff:", k_eff.min().item(), k_eff.mean().item(), k_eff.max().item())
            print("keff_gate:", keff_gate.min().item(), keff_gate.mean().item(), keff_gate.max().item())
            print("band_ij:", band_ij.min().item(), band_ij.mean().item(), band_ij.max().item())
            print("pair_prod:", pair_prod.min().item(), pair_prod.mean().item(), pair_prod.max().item())
            print("valid_pair_prod:", valid_pair_prod.min().item(), valid_pair_prod.mean().item(), valid_pair_prod.max().item())
            print("pair_relevance:", pair_relevance.min().item(), pair_relevance.mean().item(), pair_relevance.max().item())
            print("R_pair:", R_pair.min().item(), R_pair.mean().item(), R_pair.max().item())
            print("rho:", rho.min().item(), rho.mean().item(), rho.max().item())
            print("true_dist:", true_dist.min().item(), true_dist.mean().item(), true_dist.max().item())

        return (
            rho,
            pair_strength,
            band_ij,
            pair_relevance,
            edge_field,
            pair_tangent_ij,
            skeleton_field,
            skeleton_ij,
            skeleton_pair_strength,
            true_dist,
        )
    def _validate_inputs(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        tau: float,
        seeds_raw: torch.Tensor,
        w_raw: torch.Tensor,
        theta: torch.Tensor | None,
        a_raw: torch.Tensor | None,
    ) -> None:
        if points_uv.ndim != 2 or points_uv.shape[1] != 2:
            raise ValueError(f"points_uv must be (N,2), got {tuple(points_uv.shape)}")
        if Xu.ndim != 2 or Xu.shape[1] != 3:
            raise ValueError(f"Xu must be (N,3), got {tuple(Xu.shape)}")
        if Xv.ndim != 2 or Xv.shape[1] != 3:
            raise ValueError(f"Xv must be (N,3), got {tuple(Xv.shape)}")
        if Xu.shape[0] != points_uv.shape[0] or Xv.shape[0] != points_uv.shape[0]:
            raise ValueError("points_uv, Xu, and Xv must have the same first dimension")
        if seeds_raw.shape != (self.n_seeds, 2):
            raise ValueError(
                f"seeds_raw must be (S,2) with S={self.n_seeds}, got {tuple(seeds_raw.shape)}"
            )
        if w_raw.shape != (self.n_seeds, self.n_seeds):
            raise ValueError(
                f"w_raw must be (S,S) with S={self.n_seeds}, got {tuple(w_raw.shape)}"
            )
        if not (tau > 0.0):
            raise ValueError(f"tau must be > 0, got {tau}")
        if (
            self.use_Metric_anisotropy
            and not self.use_surface_metric
            and not self.use_metric_voronoi_distance
        ):
            if theta is None or a_raw is None:
                raise ValueError("use_Metric_anisotropy=True requires theta and a_raw.")
            if theta.shape != (self.n_seeds,) or a_raw.shape != (self.n_seeds,):
                raise ValueError(
                    f"theta/a_raw must be (S,) with S={self.n_seeds}, got {theta.shape}, {a_raw.shape}"
                )

    # -------------------- field evaluation --------------------

    def evaluate_at_uv(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor,
        Xv: torch.Tensor,
        tau: float,
        seeds_raw: torch.Tensor,
        w_raw: torch.Tensor,
        h_raw: torch.Tensor | None,
        theta: torch.Tensor | None = None,
        a_raw: torch.Tensor | None = None,
        points_face_id: torch.Tensor | None = None,
        boundary_uv: torch.Tensor | None = None,
        boundary_face_id: torch.Tensor | None = None,
        boundary_width_raw: torch.Tensor | None = None,
        boundary_alpha_raw: torch.Tensor | None = None,
        boundary_beta_raw: torch.Tensor | None = None,
        hard_seed_mask: bool = True,
        seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask_threshold: float = 0.5,
        seed_domain_temp: float | torch.Tensor | None = None,
        point_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        point_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        point_domain_mask_threshold: float | None = None,
        point_domain_temp: float | torch.Tensor | None = None,
        points_3d: torch.Tensor | None = None,
        centerline_radius_raw: torch.Tensor | None = None,
        centerline_radius_fixed: float | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        self._validate_inputs(
            points_uv=points_uv,
            Xu=Xu,
            Xv=Xv,
            tau=tau,
            seeds_raw=seeds_raw,
            w_raw=w_raw,
            theta=theta,
            a_raw=a_raw,
        )

        seeds = self.seeds_uv(seeds_raw)
        S = seeds.shape[0]
        seed_face_id = self.seed_face_id.to(device=seeds.device, dtype=torch.long)

        d_uv = self._point_seed_distance(
            points_uv=points_uv,
            seeds=seeds,
            metric_aware=False,
            eps=self.eps,
            points_face_id=points_face_id,
            seed_face_id=seed_face_id,
        )
        d_metric = self._point_seed_distance(
            points_uv=points_uv,
            seeds=seeds,
            Xu=Xu,
            Xv=Xv,
            metric_aware=True,
            normalize_metric=self.normalize_metric_voronoi_distance,
            eps=self.eps,
            points_face_id=points_face_id,
            seed_face_id=seed_face_id,
        )
        metric_voronoi_G, metric_voronoi_scale = self._point_metric_matrix(
            Xu=Xu,
            Xv=Xv,
            normalize_metric=self.normalize_metric_voronoi_distance,
            eps=self.eps,
        )

        I = torch.eye(2, device=points_uv.device, dtype=points_uv.dtype)
        point_metric_for_distance = None
        seed_metric_for_distance = None
        boundary_metric_for_distance = None
        if self.use_metric_voronoi_distance:
            point_metric_for_distance = metric_voronoi_G
            seed_metric_for_distance = self._sample_point_metric_at_uv(
                samples_uv=seeds,
                points_uv=points_uv,
                point_metric=point_metric_for_distance,
                samples_face_id=seed_face_id,
                points_face_id=points_face_id,
            )
            if boundary_uv is not None and boundary_uv.numel() > 0:
                boundary_metric_for_distance = self._sample_point_metric_at_uv(
                    samples_uv=boundary_uv.to(device=points_uv.device, dtype=points_uv.dtype),
                    points_uv=points_uv,
                    point_metric=point_metric_for_distance,
                    samples_face_id=boundary_face_id,
                    points_face_id=points_face_id,
                )
            d = self._point_seed_distance_metric(
                points_uv=points_uv,
                seeds=seeds,
                point_metric=point_metric_for_distance,
                points_face_id=points_face_id,
                seed_face_id=seed_face_id,
                metric_aware=True,
                mask_cross_face=points_face_id is not None,
            )
            metric_for_bisector = metric_voronoi_G
            metric_mode = "point"

            # Kept for backward-compatible debug output; geometry uses metric_voronoi_G.
            M = I.unsqueeze(0).expand(S, 2, 2)
            G = metric_voronoi_G
        elif self.use_surface_metric:
            G, _ = self._point_metric_matrix(
                Xu=Xu,
                Xv=Xv,
                normalize_metric=False,
                eps=self.eps,
            )
            detG = (
                G[:, 0, 0] * G[:, 1, 1]
                - G[:, 0, 1] * G[:, 1, 0]
            ).clamp_min(self.eps)
            scale = torch.sqrt(detG)
            G = G / scale[:, None, None].clamp_min(self.eps)
            I2 = torch.eye(2, device=points_uv.device, dtype=points_uv.dtype)
            G = G + self.eps * I2.unsqueeze(0)

            point_metric_for_distance = G
            seed_metric_for_distance = self._sample_point_metric_at_uv(
                samples_uv=seeds,
                points_uv=points_uv,
                point_metric=point_metric_for_distance,
                samples_face_id=seed_face_id,
                points_face_id=points_face_id,
            )
            if boundary_uv is not None and boundary_uv.numel() > 0:
                boundary_metric_for_distance = self._sample_point_metric_at_uv(
                    samples_uv=boundary_uv.to(device=points_uv.device, dtype=points_uv.dtype),
                    points_uv=points_uv,
                    point_metric=point_metric_for_distance,
                    samples_face_id=boundary_face_id,
                    points_face_id=points_face_id,
                )
            metric_for_bisector = G
            metric_mode = "point"
            d = self._point_seed_distance_metric(
                points_uv=points_uv,
                seeds=seeds,
                point_metric=point_metric_for_distance,
                points_face_id=points_face_id,
                seed_face_id=seed_face_id,
                metric_aware=True,
                mask_cross_face=points_face_id is not None,
            )

            # Kept for backward-compatible debug output; geometry uses G.
            M = I.unsqueeze(0).expand(S, 2, 2)
        else:
            G = torch.empty(0, 2, 2, device=points_uv.device, dtype=points_uv.dtype)
            if self.use_Metric_anisotropy:
                M = self.metric_matrices(theta, a_raw)
            else:
                M = I.unsqueeze(0).expand(S, 2, 2)

            seed_metric_for_distance = M
            d = self._point_seed_distance_metric(
                points_uv=points_uv,
                seeds=seeds,
                seed_metric=M,
                points_face_id=points_face_id,
                seed_face_id=seed_face_id,
                metric_aware=True,
                mask_cross_face=points_face_id is not None,
            )
            metric_for_bisector = M
            metric_mode = "seed"

        logits = -d / tau

        (
            seed_active_weights,
            seed_active_mask,
            seed_domain_weight,
            seed_domain_sdf_values,
            seed_domain_mask_values,
            seed_duplicate_weights,
            seed_domain_activity_weights,
        ) = self._seed_activation_state(
            seeds=seeds,
            hard_seed_mask=hard_seed_mask,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
            seed_domain_temp=seed_domain_temp,
            seed_face_id=seed_face_id,
            seed_metric=seed_metric_for_distance,
        )
        duplicate_suppression_dist_debug = self._seed_seed_distance_metric(
            seeds=seeds,
            seed_face_id=seed_face_id,
            seed_metric=seed_metric_for_distance,
            metric_aware=self._use_surface_distance_geometry() and seed_metric_for_distance is not None,
        )
        if point_domain_sdf is None and self._domain_can_sample_count(seed_domain_sdf, points_uv.shape[0]):
            point_domain_sdf = seed_domain_sdf
        if point_domain_mask is None and self._domain_can_sample_count(seed_domain_mask, points_uv.shape[0]):
            point_domain_mask = seed_domain_mask
        point_domain_temp_t = (
            seed_domain_temp
            if point_domain_temp is None and seed_domain_temp is not None
            else point_domain_temp
        )
        point_domain_temp_t = torch.as_tensor(
            max(float(self.duplicate_merge_sigma) * float(self.duplicate_effect_temp_ratio), self.eps)
            if point_domain_temp_t is None
            else point_domain_temp_t,
            device=points_uv.device,
            dtype=points_uv.dtype,
        ).clamp_min(self.eps)
        point_domain_weight, point_domain_sdf_values, point_domain_mask_values = (
            self._point_domain_validity_state(
                points_uv=points_uv,
                temp=point_domain_temp_t,
                point_domain_sdf=point_domain_sdf,
                point_domain_mask=point_domain_mask,
                point_domain_mask_threshold=(
                    seed_domain_mask_threshold
                    if point_domain_mask_threshold is None
                    else point_domain_mask_threshold
                ),
            )
        )
        point_domain_floor = torch.as_tensor(
            self.point_domain_floor,
            device=points_uv.device,
            dtype=points_uv.dtype,
        )
        point_domain_activity = point_domain_floor + (1.0 - point_domain_floor) * point_domain_weight

        seeds_eval = seeds
        d_eval = d
        metric_for_bisector_eval = metric_for_bisector
        w_raw_eval = w_raw
        seed_active_weights_eval = seed_active_weights
        seed_duplicate_weights_eval = seed_duplicate_weights
        seed_domain_activity_weights_eval = seed_domain_activity_weights
        seed_active_mask_eval = seed_active_mask
        seed_face_id_eval = seed_face_id
        seed_metric_eval = seed_metric_for_distance

        if hard_seed_mask and bool(seed_active_mask.any()) and not bool(seed_active_mask.all()):
            active_idx = torch.nonzero(seed_active_mask, as_tuple=False).flatten()
            seeds_eval = seeds.index_select(0, active_idx)
            d_eval = d.index_select(1, active_idx)
            if metric_mode == "seed":
                metric_for_bisector_eval = metric_for_bisector.index_select(0, active_idx)
            else:
                metric_for_bisector_eval = metric_for_bisector
            w_raw_eval = w_raw.index_select(0, active_idx).index_select(1, active_idx)
            seed_active_weights_eval = seed_active_weights.index_select(0, active_idx)
            seed_duplicate_weights_eval = seed_duplicate_weights.index_select(0, active_idx)
            seed_domain_activity_weights_eval = seed_domain_activity_weights.index_select(0, active_idx)
            seed_active_mask_eval = torch.ones_like(seed_active_weights_eval, dtype=torch.bool)
            seed_face_id_eval = seed_face_id.index_select(0, active_idx)
            if seed_metric_eval is not None:
                seed_metric_eval = seed_metric_eval.index_select(0, active_idx)

        logits = -d_eval / tau
        logits = logits + torch.log(seed_active_weights_eval.clamp_min(self.eps)).unsqueeze(0)
        invalid_domain_assignment_mask = (
            seed_domain_activity_weights_eval < self.invalid_domain_assignment_threshold
        )
        logits = logits.masked_fill(invalid_domain_assignment_mask.unsqueeze(0), -1e6)

        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits = logits.clamp(min=-80.0, max=0.0)
        w_soft = torch.softmax(logits, dim=-1)
        if hard_seed_mask and bool(seed_active_mask_eval.any()):
            active_float = seed_active_mask_eval.to(device=w_soft.device, dtype=w_soft.dtype).unsqueeze(0)
            w_soft = w_soft * active_float
            w_soft = w_soft / w_soft.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        w_geo = self.width(
            w_raw_eval,
            seeds=seeds_eval,
            seed_face_id=seed_face_id_eval,
            seed_metric=seed_metric_eval,
        )

        (
            rho_v,
            pair_strength,
            band_ij,
            pair_relevance,
            edge_field,
            pair_tangent_ij,
            skeleton_field,
            skeleton_ij,
            skeleton_pair_strength,
            skeleton_true_dist,
        ) = self._bisector_band_density(
            points=points_uv,
            seeds=seeds_eval,
            d=d_eval,
            w_soft=w_soft,
            w_geo=w_geo,
            beta=self.beta,
            metric=metric_for_bisector_eval,
            metric_mode=metric_mode,
            seed_active_weights=seed_active_weights_eval,
            seed_duplicate_weights=seed_duplicate_weights_eval,
            seed_domain_weights=seed_domain_activity_weights_eval,
            hard_seed_mask=hard_seed_mask,
            seed_face_id=seed_face_id_eval,
            points_face_id=points_face_id,
            point_metric=point_metric_for_distance,
            seed_metric=seed_metric_eval,
            skeleton_sigma=self._metric_scale_value(self.skeleton_sigma, self.skeleton_sigma_metric),
        )
        rho_voronoi = rho_v

        if self.use_boundary_attachment:
            if self.use_centerline_thickness:
                if points_3d is None:
                    raise ValueError(
                        "points_3d must be provided when use_centerline_thickness=True"
                    )
                rho_b = self.boundary_centerline_thickness_density(
                    points_xyz=points_3d.to(device=points_uv.device, dtype=points_uv.dtype),
                    points_uv=points_uv,
                    boundary_uv=boundary_uv,
                    points_face_id=points_face_id,
                    boundary_face_id=boundary_face_id,
                    boundary_width_raw=boundary_width_raw,
                    boundary_beta_raw=boundary_beta_raw,
                )
            else:
                rho_b = self.boundary_attachment_field(
                    points_uv=points_uv,
                    boundary_uv=boundary_uv,
                    points_face_id=points_face_id,
                    boundary_face_id=boundary_face_id,
                    boundary_width_raw=boundary_width_raw,
                    boundary_beta_raw=boundary_beta_raw,
                    point_metric=point_metric_for_distance,
                    boundary_metric=boundary_metric_for_distance,
                )
            alpha_b = self.boundary_alpha(points_uv, boundary_alpha_raw=boundary_alpha_raw)
            rho = self.smooth_union(rho_a=rho_v, rho_b=rho_b, alpha_b=alpha_b)
        else:
            rho_b = torch.zeros_like(rho_v)
            alpha_b = torch.zeros((), device=points_uv.device, dtype=points_uv.dtype)
            rho = rho_v

        if self.use_centerline_thickness:
            if points_3d is None:
                raise ValueError(
                    "points_3d must be provided when use_centerline_thickness=True"
                )
            rho_centerline = self.centerline_thickness_density(
                points_xyz=points_3d.to(device=points_uv.device, dtype=points_uv.dtype),
                skeleton_field=skeleton_field,
                radius_raw=centerline_radius_raw,
                radius_fixed=centerline_radius_fixed,
                threshold=self.centerline_threshold,
                threshold_softness=self.centerline_threshold_softness,
                softmin_tau=self.centerline_softmin_tau,
                beta=self.centerline_beta,
                debugging =True,
            )
            centerline_diagnostics = self._last_centerline_diagnostics
            if self.use_boundary_attachment:
                rho = self.smooth_union(rho_a=rho_centerline, rho_b=rho_b, alpha_b=alpha_b)
            else:
                rho = rho_centerline
        else:
            rho_centerline = torch.zeros_like(rho_v)
            centerline_core_weight = torch.zeros_like(rho_v)
            centerline_d_core = torch.zeros_like(rho_v)
            centerline_diagnostics = {}

        rho_v = rho_v * point_domain_activity
        rho_voronoi = rho_voronoi * point_domain_activity
        rho_b = rho_b * point_domain_activity
        rho = rho * point_domain_activity
        rho_centerline = rho_centerline * point_domain_activity
        if self.use_centerline_thickness:
            centerline_core_weight = centerline_diagnostics["centerline_core_weight"].to(
                device=points_uv.device,
                dtype=points_uv.dtype,
            )
            centerline_d_core = centerline_diagnostics["centerline_d_core"].to(
                device=points_uv.device,
                dtype=points_uv.dtype,
            )
        #rho = self.soft_project_density(rho)
        rho = rho.clamp(0.0, 1.0)

        eps_rho = 1e-3
        rho0_solid = 0.55
        gamma_solid = 0.02
        rho_s = eps_rho + (1.0 - eps_rho) * torch.sigmoid((rho - rho0_solid) / gamma_solid)

        fiber_pair_weights = self._fiber_pair_weights(
            w_soft=w_soft,
            seeds=seeds_eval,
            band_ij=band_ij,
            pair_relevance=pair_relevance,
            seed_active_weights=seed_active_weights_eval,
            seed_duplicate_weights=seed_duplicate_weights_eval,
            seed_domain_weights=seed_domain_activity_weights_eval,
            seed_face_id=seed_face_id_eval,
            seed_metric=seed_metric_eval,
        )

        fiber_tensor_Q = self._axial_tensor_from_local_pair_tangents(
            fiber_pair_weights,
            pair_tangent_ij,
        )
        t_uv_raw = self._principal_axial_direction(fiber_tensor_Q)
        fiber_tensor_Q_interior = fiber_tensor_Q

        if (
            self.use_boundary_attachment
            and self.use_boundary_tangent_fibers
            and boundary_uv is not None
            and boundary_uv.numel() > 0
        ):
            # Boundary tangents are also an axial line field, so blend them as
            # tensors rather than signed vectors to avoid sign-flip cancellation.
            fiber_tensor_Q_boundary = self._boundary_tangent_tensor_field(
                points_uv=points_uv,
                boundary_uv=boundary_uv,
                points_face_id=points_face_id,
                boundary_face_id=boundary_face_id,
                point_metric=point_metric_for_distance,
                boundary_metric=boundary_metric_for_distance,
            )
            boundary_tangent_weight = rho_b.clamp(0.0, 1.0)
            lam_b = boundary_tangent_weight.unsqueeze(-1).unsqueeze(-1)
            fiber_tensor_Q_final = (
                (1.0 - lam_b) * fiber_tensor_Q_interior
                + lam_b * fiber_tensor_Q_boundary
            )
            t_uv_raw = self._principal_axial_direction(fiber_tensor_Q_final)
        else:
            fiber_tensor_Q_boundary = torch.zeros_like(fiber_tensor_Q_interior)
            boundary_tangent_weight = torch.zeros_like(rho_b)
            fiber_tensor_Q_final = fiber_tensor_Q_interior

        fiber_coherence = self._axial_coherence_from_tensor(fiber_tensor_Q_final)

        rho0, gamma = 0.5, 0.05
        m = torch.sigmoid((rho - rho0) / gamma).unsqueeze(1)
        fiber_strength = m.squeeze(1)

        fallback_uv = torch.zeros_like(t_uv_raw)
        fallback_uv[:, 0] = 1.0
        t_uv_norm = torch.linalg.norm(t_uv_raw, dim=1, keepdim=True)
        t_uv = torch.where(t_uv_norm > self.eps, t_uv_raw, fallback_uv)
        fiber3d = self.map_to_3d(t_uv, Xu=Xu, Xv=Xv)
        h = self.height(h_raw, ref_tensor=points_uv)

        return {
            "w_soft": w_soft,
            "d": d,
            "d_uv_mean": d_uv.mean(),
            "d_metric_mean": d_metric.mean(),
            "d_metric_scale_mean": metric_voronoi_scale,
            "M": M,
            "surface_metric_G": G,
            "metric_mode": metric_mode,
            "seeds": seeds,
            "seed_active_weights": seed_active_weights,
            "seed_active_mask": seed_active_mask,
            "seed_domain_weight": seed_domain_weight,
            "seed_duplicate_weights": seed_duplicate_weights,
            "seed_domain_activity_weights": seed_domain_activity_weights,
            "seed_domain_sdf_values": seed_domain_sdf_values,
            "seed_domain_mask_values": seed_domain_mask_values,
            "point_domain_weight": point_domain_weight,
            "point_domain_activity": point_domain_activity,
            "point_domain_sdf_values": point_domain_sdf_values,
            "point_domain_mask_values": point_domain_mask_values,
            "invalid_domain_assignment_mask": invalid_domain_assignment_mask,
            "inactive_seed_indices": torch.nonzero(~seed_active_mask, as_tuple=False).flatten(),
            "active_seed_count": seed_active_mask.to(seeds.dtype).sum(),
            "inactive_seed_count": (~seed_active_mask).to(seeds.dtype).sum(),
            "rho": rho,
            "rho_s": rho_s,
            "rho_v": rho_v,
            "rho_voronoi": rho_voronoi,
            "rho_b": rho_b,
            "rho_centerline": rho_centerline,
            "centerline_core_weight": centerline_core_weight,
            "centerline_d_core": centerline_d_core,
            "centerline_radius": self.centerline_radius(
                centerline_radius_raw,
                radius_fixed=centerline_radius_fixed,
                ref_tensor=points_3d if points_3d is not None else points_uv,
            ),
            "centerline_diagnostics": centerline_diagnostics,
            "skeleton_field": skeleton_field,
            "skeleton_ij": skeleton_ij,
            "skeleton_pair_strength": skeleton_pair_strength,
            "skeleton_true_dist": skeleton_true_dist,
            "t_uv_raw": t_uv_raw,
            "t_uv": t_uv,
            "fiber3d": fiber3d,
            "fiber_strength": fiber_strength,
            "fiber_coherence": fiber_coherence,
            "fiber_pair_weights": fiber_pair_weights,
            "fiber_tensor_Q": fiber_tensor_Q_final,
            "fiber_tensor_Q_interior": fiber_tensor_Q_interior,
            "fiber_tensor_Q_boundary": fiber_tensor_Q_boundary,
            "fiber_tensor_Q_final": fiber_tensor_Q_final,
            "boundary_tangent_weight": boundary_tangent_weight,
            "h": h,
            "w_geo": w_geo,
            "pair_strength": pair_strength,
            "band_ij": band_ij,
            "pair_relevance": pair_relevance,
            "edge_field": edge_field,
            "pair_tangent_ij": pair_tangent_ij,
            "boundary_alpha": alpha_b,
            "boundary_width": (
                self.boundary_width(points_uv, boundary_width_raw)
                if self.use_boundary_attachment
                else torch.zeros((), device=points_uv.device, dtype=points_uv.dtype)
            ),
            "boundary_beta": (
                self.boundary_beta(points_uv, boundary_beta_raw)
                if self.use_boundary_attachment
                else torch.zeros((), device=points_uv.device, dtype=points_uv.dtype)
            ),
            "metric_distance_debug": {
                "seed_seed_dist_min_mean_max": self._distance_stats_tensor(
                    self._seed_seed_distance_metric(
                        seeds=seeds_eval,
                        seed_face_id=seed_face_id_eval,
                        seed_metric=seed_metric_eval,
                        metric_aware=self._use_surface_distance_geometry() and seed_metric_eval is not None,
                    )
                ),
                "point_seed_dist_min_mean_max": self._distance_stats_tensor(d_eval),
                "duplicate_suppression_dist_min_mean_max": self._distance_stats_tensor(
                    duplicate_suppression_dist_debug
                ),
                "true_dist_min_mean_max": self._distance_stats_tensor(skeleton_true_dist),
                "w_geo_eff_min_mean_max": self._distance_stats_tensor(w_geo),
            },
        }

    def forward(
        self,
        points_uv,
        Xu,
        Xv,
        tau,
        seeds_raw,
        w_raw,
        h_raw=None,
        theta=None,
        a_raw=None,
        points_face_id=None,
        boundary_uv=None,
        boundary_face_id=None,
        boundary_width_raw=None,
        boundary_alpha_raw=None,
        boundary_beta_raw=None,
        hard_seed_mask=False,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold=0.5,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold=None,
        point_domain_temp=None,
        points_3d=None,
        centerline_radius_raw=None,
        centerline_radius_fixed=None,
    ):
        return self.evaluate_at_uv(
            points_uv=points_uv,
            Xu=Xu,
            Xv=Xv,
            tau=tau,
            seeds_raw=seeds_raw,
            w_raw=w_raw,
            h_raw=h_raw,
            theta=theta,
            a_raw=a_raw,
            points_face_id=points_face_id,
            boundary_uv=boundary_uv,
            boundary_face_id=boundary_face_id,
            boundary_width_raw=boundary_width_raw,
            boundary_alpha_raw=boundary_alpha_raw,
            boundary_beta_raw=boundary_beta_raw,
            hard_seed_mask=hard_seed_mask,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
            seed_domain_temp=seed_domain_temp,
            point_domain_sdf=point_domain_sdf,
            point_domain_mask=point_domain_mask,
            point_domain_mask_threshold=point_domain_mask_threshold,
            point_domain_temp=point_domain_temp,
            points_3d=points_3d,
            centerline_radius_raw=centerline_radius_raw,
            centerline_radius_fixed=centerline_radius_fixed,
        )

@dataclass
class MeshQueryData:
    points_uv: torch.Tensor
    Xu: torch.Tensor
    Xv: torch.Tensor
    points_xyz: torch.Tensor
    faces_ijk: torch.Tensor
    tau: float
    points_face_id: torch.Tensor | None = None
    boundary_uv: torch.Tensor | None = None
    boundary_face_id: torch.Tensor | None = None
    seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None
    seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None
    seed_domain_mask_threshold: float = 0.5
    seed_domain_temp: float | torch.Tensor | None = None
    point_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None
    point_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None
    point_domain_mask_threshold: float = 0.5
    point_domain_temp: float | torch.Tensor | None = None

class VoronoiModelVisualizer:
    """
    Helper for evaluating a VoronoiDecoder on a fixed mesh/query set and
    visualizing results in UV and 3D.

    Boundary data can be supplied either:
    - at initialization as defaults
    - or per evaluation call to override defaults
    """

    def __init__(
        self,
        *,
        points_uv,
        Xu,
        Xv,
        points_xyz,
        faces_ijk,
        tau: float,
        n_seeds: int,
        points_face_id=None,
        boundary_uv=None,
        boundary_face_id=None,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold: float = 0.5,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold: float | None = None,
        point_domain_temp=None,
        duplicate_merge_sigma =None,
        eps: float = 1e-8,
        use_metric_anisotropy: bool = False,
        use_surface_metric: bool = False,
        use_metric_voronoi_distance: bool = False,
        normalize_metric_voronoi_distance: bool = True,
        w_min: float = 0.005,
        fixed_height: float | None = None,
        use_boundary_attachment: bool = False,
        boundary_solid_idx: torch.Tensor | None = None,
        face_u_periodic: torch.Tensor | None = None,
        face_v_periodic: torch.Tensor | None = None,
        seed_face_id: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        density_projection_strength: float = 0.0,
        density_projection_threshold: float = 0.5,
        density_projection_gamma: float = 0.05,
        seed_activity_sharpness: float = 1.0,
        skeleton_sigma: float = 0.0,
        **decoder_kwargs,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.n_seeds = int(n_seeds)
        self.eps = float(eps)
        point_count = int(points_uv.shape[0])

        self.query = MeshQueryData(
            points_uv=self._to_tensor(points_uv, dtype=self.dtype),
            Xu=self._to_tensor(Xu, dtype=self.dtype),
            Xv=self._to_tensor(Xv, dtype=self.dtype),
            points_xyz=self._to_tensor(points_xyz, dtype=self.dtype),
            faces_ijk=self._to_tensor(faces_ijk, dtype=torch.long),
            tau=float(tau),
            points_face_id=self._to_tensor(points_face_id, dtype=torch.long),
            boundary_uv=self._to_tensor(boundary_uv, dtype=self.dtype),
            boundary_face_id=self._to_tensor(boundary_face_id, dtype=torch.long),
            seed_domain_sdf=self._to_domain_input(seed_domain_sdf),
            seed_domain_mask=self._to_domain_input(seed_domain_mask),
            seed_domain_mask_threshold=float(seed_domain_mask_threshold),
            seed_domain_temp=self._to_tensor(seed_domain_temp, dtype=self.dtype),
            point_domain_sdf=self._to_domain_input(
                seed_domain_sdf
                if (
                    point_domain_sdf is None
                    and VoronoiDecoder._domain_can_sample_count(seed_domain_sdf, point_count)
                )
                else point_domain_sdf
            ),
            point_domain_mask=self._to_domain_input(
                seed_domain_mask
                if (
                    point_domain_mask is None
                    and VoronoiDecoder._domain_can_sample_count(seed_domain_mask, point_count)
                )
                else point_domain_mask
            ),
            point_domain_mask_threshold=float(
                seed_domain_mask_threshold
                if point_domain_mask_threshold is None
                else point_domain_mask_threshold
            ),
            point_domain_temp=self._to_tensor(
                seed_domain_temp if point_domain_temp is None else point_domain_temp,
                dtype=self.dtype,
            ),
        )

        self.decoder = VoronoiDecoder(
            n_seeds=self.n_seeds,
            eps=eps,
            use_Metric_anisotropy=use_metric_anisotropy,
            use_surface_metric=use_surface_metric,
            use_metric_voronoi_distance=use_metric_voronoi_distance,
            normalize_metric_voronoi_distance=normalize_metric_voronoi_distance,
            w_min=w_min,
            fixed_height=fixed_height,
            use_boundary_attachment=use_boundary_attachment,
            boundary_solid_idx=boundary_solid_idx,
            face_u_periodic=face_u_periodic,
            face_v_periodic=face_v_periodic,
            seed_face_id=seed_face_id,
            duplicate_merge_sigma = duplicate_merge_sigma,
            density_projection_strength = density_projection_strength,
            density_projection_threshold = density_projection_threshold,
            density_projection_gamma = density_projection_gamma,
            seed_activity_sharpness = seed_activity_sharpness,
            skeleton_sigma = skeleton_sigma,
            **decoder_kwargs,
        ).to(device=self.device, dtype=self.dtype)
        self.decoder.eval()

        try:
            pv.set_jupyter_backend("trame")
        except Exception:
            pass

    # ---------------------------
    # tensor helpers
    # ---------------------------

    def _to_tensor(
        self,
        value,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=device or self.device, dtype=dtype or value.dtype)
        return torch.as_tensor(
            value,
            device=device or self.device,
            dtype=dtype or self.dtype,
        )

    def _to_domain_input(self, value):
        if value is None or callable(value):
            return value
        return self._to_tensor(value, dtype=self.dtype)

    def make_query_data(
        self,
        *,
        points_uv=None,
        Xu=None,
        Xv=None,
        points_xyz=None,
        faces_ijk=None,
        tau: float | None = None,
        points_face_id=None,
        boundary_uv=None,
        boundary_face_id=None,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold: float | None = None,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold: float | None = None,
        point_domain_temp=None,
    ) -> MeshQueryData:
        """
        Create a query object, using stored defaults for omitted values.
        """
        return MeshQueryData(
            points_uv=self._to_tensor(
                self.query.points_uv if points_uv is None else points_uv,
                dtype=self.dtype,
            ),
            Xu=self._to_tensor(
                self.query.Xu if Xu is None else Xu,
                dtype=self.dtype,
            ),
            Xv=self._to_tensor(
                self.query.Xv if Xv is None else Xv,
                dtype=self.dtype,
            ),
            points_xyz=self._to_tensor(
                self.query.points_xyz if points_xyz is None else points_xyz,
                dtype=self.dtype,
            ),
            faces_ijk=self._to_tensor(
                self.query.faces_ijk if faces_ijk is None else faces_ijk,
                dtype=torch.long,
            ),
            tau=float(self.query.tau if tau is None else tau),
            points_face_id=self._to_tensor(
                self.query.points_face_id if points_face_id is None else points_face_id,
                dtype=torch.long,
            ),
            boundary_uv=self._to_tensor(
                self.query.boundary_uv if boundary_uv is None else boundary_uv,
                dtype=self.dtype,
            ),
            boundary_face_id=self._to_tensor(
                self.query.boundary_face_id if boundary_face_id is None else boundary_face_id,
                dtype=torch.long,
            ),
            seed_domain_sdf=self._to_domain_input(
                self.query.seed_domain_sdf if seed_domain_sdf is None else seed_domain_sdf
            ),
            seed_domain_mask=self._to_domain_input(
                self.query.seed_domain_mask if seed_domain_mask is None else seed_domain_mask
            ),
            seed_domain_mask_threshold=float(
                self.query.seed_domain_mask_threshold
                if seed_domain_mask_threshold is None
                else seed_domain_mask_threshold
            ),
            seed_domain_temp=self._to_tensor(
                self.query.seed_domain_temp if seed_domain_temp is None else seed_domain_temp,
                dtype=self.dtype,
            ),
            point_domain_sdf=self._to_domain_input(
                self.query.point_domain_sdf if point_domain_sdf is None else point_domain_sdf
            ),
            point_domain_mask=self._to_domain_input(
                self.query.point_domain_mask if point_domain_mask is None else point_domain_mask
            ),
            point_domain_mask_threshold=float(
                self.query.point_domain_mask_threshold
                if point_domain_mask_threshold is None
                else point_domain_mask_threshold
            ),
            point_domain_temp=self._to_tensor(
                self.query.point_domain_temp if point_domain_temp is None else point_domain_temp,
                dtype=self.dtype,
            ),
        )

    @classmethod
    def from_face_tensor(
        cls,
        face_tensor: dict[str, Any],
        *,
        tau: float,
        n_seeds: int,
        use_face_seed_domain_mask: bool = True,
        seed_domain_mask_threshold: float = 0.5,
        seed_domain_temp=None,
        **kwargs,
    ) -> "VoronoiModelVisualizer":
        points_uv = face_tensor["uv"]
        device = points_uv.device if isinstance(points_uv, torch.Tensor) else None
        boundary_uv = kwargs.pop("boundary_uv", None)
        boundary_face_id = kwargs.pop("boundary_face_id", None)
        if boundary_uv is None and face_tensor.get("boundary_idx_ring1", None) is not None:
            bidx = torch.unique(face_tensor["boundary_idx_ring1"].to(dtype=torch.long))
            if bidx.numel() > 0:
                boundary_uv = face_tensor["uv"][bidx]
                boundary_face_id = torch.zeros(bidx.numel(), dtype=torch.long, device=device)

        seed_domain_mask = kwargs.pop("seed_domain_mask", None)
        if seed_domain_mask is None and use_face_seed_domain_mask:
            seed_domain_mask = face_tensor.get("seed_domain_mask_grid", None)
            if seed_domain_mask is None:
                seed_domain_mask = face_tensor.get("seed_domain_mask", None)

        return cls(
            points_uv=face_tensor["uv"],
            Xu=face_tensor["Xu"],
            Xv=face_tensor["Xv"],
            points_xyz=face_tensor["points_xyz"],
            faces_ijk=face_tensor["faces_ijk"],
            tau=tau,
            n_seeds=n_seeds,
            points_face_id=torch.zeros(
                face_tensor["uv"].shape[0],
                dtype=torch.long,
                device=device,
            ),
            boundary_uv=boundary_uv,
            boundary_face_id=boundary_face_id,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
            seed_domain_temp=seed_domain_temp,
            face_u_periodic=torch.tensor([bool(face_tensor.get("u_periodic", False))], dtype=torch.bool),
            face_v_periodic=torch.tensor([bool(face_tensor.get("v_periodic", False))], dtype=torch.bool),
            seed_face_id=torch.zeros(n_seeds, dtype=torch.long),
            **kwargs,
        )

    # ---------------------------
    # geometry helpers
    # ---------------------------

    @staticmethod
    def faces_ijk_to_pv_faces(faces_ijk: torch.Tensor) -> np.ndarray:
        f = faces_ijk.detach().cpu().numpy().astype(np.int64)
        pv_faces = np.empty((f.shape[0], 4), dtype=np.int64)
        pv_faces[:, 0] = 3
        pv_faces[:, 1:] = f
        return pv_faces.reshape(-1)

    @staticmethod
    def seeds_uv_to_xyz_nearest(
        seeds_uv: torch.Tensor,
        uv: torch.Tensor,
        points_xyz: torch.Tensor,
    ) -> torch.Tensor:
        device = uv.device
        seeds_uv = seeds_uv.to(device=device, dtype=uv.dtype)
        points_xyz = points_xyz.to(device=device, dtype=points_xyz.dtype)
        nn = torch.cdist(seeds_uv, uv).argmin(dim=1)
        return points_xyz[nn]

    # ---------------------------
    # evaluation
    # ---------------------------

    def run_case(
        self,
        *,
        seeds_raw,
        w_raw,
        h_raw=None,
        theta=None,
        a_raw=None,
        query: MeshQueryData | None = None,
        boundary_uv=None,
        boundary_face_id=None,
        boundary_width_raw=None,
        boundary_alpha_raw=None,
        boundary_beta_raw=None,
        centerline_radius_raw=None,
        centerline_radius_fixed=None,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold=None,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold=None,
        point_domain_temp=None,
        hard_seed_mask = False,
    ) -> dict[str, torch.Tensor]:
        q = self.query if query is None else query

        q_boundary_uv = (
            q.boundary_uv
            if boundary_uv is None
            else self._to_tensor(boundary_uv, dtype=self.dtype)
        )
        q_boundary_face_id = (
            q.boundary_face_id
            if boundary_face_id is None
            else self._to_tensor(boundary_face_id, dtype=torch.long)
        )
        q_seed_domain_sdf = (
            q.seed_domain_sdf
            if seed_domain_sdf is None
            else self._to_domain_input(seed_domain_sdf)
        )
        q_seed_domain_mask = (
            q.seed_domain_mask
            if seed_domain_mask is None
            else self._to_domain_input(seed_domain_mask)
        )
        q_seed_domain_temp = (
            q.seed_domain_temp
            if seed_domain_temp is None
            else self._to_tensor(seed_domain_temp, dtype=self.dtype)
        )
        q_point_domain_sdf = (
            q.point_domain_sdf
            if point_domain_sdf is None
            else self._to_domain_input(point_domain_sdf)
        )
        q_point_domain_mask = (
            q.point_domain_mask
            if point_domain_mask is None
            else self._to_domain_input(point_domain_mask)
        )
        q_point_domain_temp = (
            q.point_domain_temp
            if point_domain_temp is None
            else self._to_tensor(point_domain_temp, dtype=self.dtype)
        )

        with torch.no_grad():
            return self.decoder.evaluate_at_uv(
                points_uv=q.points_uv,
                Xu=q.Xu,
                Xv=q.Xv,
                tau=float(q.tau),
                seeds_raw=self._to_tensor(seeds_raw, dtype=self.dtype),
                w_raw=self._to_tensor(w_raw, dtype=self.dtype),
                h_raw=self._to_tensor(h_raw, dtype=self.dtype),
                theta=self._to_tensor(theta, dtype=self.dtype),
                a_raw=self._to_tensor(a_raw, dtype=self.dtype),
                points_face_id=q.points_face_id,
                boundary_uv=q_boundary_uv,
                boundary_face_id=q_boundary_face_id,
                boundary_width_raw=self._to_tensor(boundary_width_raw, dtype=self.dtype),
                boundary_alpha_raw=self._to_tensor(boundary_alpha_raw, dtype=self.dtype),
                boundary_beta_raw=self._to_tensor(boundary_beta_raw, dtype=self.dtype),
                centerline_radius_raw=self._to_tensor(centerline_radius_raw, dtype=self.dtype),
                centerline_radius_fixed=centerline_radius_fixed,
                hard_seed_mask=hard_seed_mask,
                seed_domain_sdf=q_seed_domain_sdf,
                seed_domain_mask=q_seed_domain_mask,
                seed_domain_mask_threshold=(
                    q.seed_domain_mask_threshold
                    if seed_domain_mask_threshold is None
                    else seed_domain_mask_threshold
                ),
                seed_domain_temp=q_seed_domain_temp,
                point_domain_sdf=q_point_domain_sdf,
                point_domain_mask=q_point_domain_mask,
                point_domain_mask_threshold=(
                    q.point_domain_mask_threshold
                    if point_domain_mask_threshold is None
                    else point_domain_mask_threshold
                ),
                point_domain_temp=q_point_domain_temp,
                points_3d=q.points_xyz,
            )

    def compute_case_volume(
        self,
        case_or_result: dict[str, Any],
        *,
        query: MeshQueryData | None = None,
        use_sharpened: bool = False,
    ) -> dict[str, float]:
        q = self.query if query is None else query
        case = case_or_result["case"] if "case" in case_or_result else case_or_result

        rho_key = "rho_s" if use_sharpened else "rho"
        rho = self._to_tensor(case[rho_key], dtype=self.dtype, device=q.points_uv.device)
        h = self._to_tensor(case["h"], dtype=self.dtype, device=q.points_uv.device)

        area_w = torch.linalg.norm(torch.cross(q.Xu, q.Xv, dim=1), dim=1).clamp_min(self.eps)
        if h.ndim == 0:
            h = h.expand_as(rho)
        elif h.shape != rho.shape:
            h = h.expand_as(rho)

        surface_area = area_w.sum()
        volume = (rho * h * area_w).sum()
        volume_fraction = (rho * area_w).sum() / surface_area.clamp_min(self.eps)

        rho_cont = self._to_tensor(case["rho"], dtype=self.dtype, device=q.points_uv.device)
        rho_sharp = self._to_tensor(case["rho_s"], dtype=self.dtype, device=q.points_uv.device)
        volume_cont = (rho_cont * h * area_w).sum()
        volume_sharp = (rho_sharp * h * area_w).sum()
        volfrac_cont = (rho_cont * area_w).sum() / surface_area.clamp_min(self.eps)
        volfrac_sharp = (rho_sharp * area_w).sum() / surface_area.clamp_min(self.eps)

        return {
            "surface_area": float(surface_area.detach().cpu().item()),
            "mean_height": float(h.mean().detach().cpu().item()),
            "volume": float(volume.detach().cpu().item()),
            "volume_cont": float(volume_cont.detach().cpu().item()),
            "volume_sharp": float(volume_sharp.detach().cpu().item()),
            "volume_fraction": float(volume_fraction.detach().cpu().item()),
            "volfrac_cont": float(volfrac_cont.detach().cpu().item()),
            "volfrac_sharp": float(volfrac_sharp.detach().cpu().item()),
        }

    # ---------------------------
    # plotting
    # ---------------------------

    def plot_uv_fields(
        self,
        *,
        out: dict[str, torch.Tensor],
        seeds_raw,
        cmap: str = "viridis",
        figsize: tuple[float, float] = (12.0, 5.0),
        fiber_stride: int = 20,
        fiber_scale: float = 0.06,
        fiber_min_strength: float = 0.05,
        show_fiber_density_background: bool = True,
        color_seeds_by_activation: bool = True,
        seed_cmap: str = "plasma",
        query: MeshQueryData | None = None,
    ):
        q = self.query if query is None else query
        uv_plot = q.points_uv.detach().cpu()
        seeds_plot = self._to_tensor(seeds_raw, dtype=self.dtype).detach().cpu()

        active_mask_out = out.get("seed_active_mask")
        if active_mask_out is None:
            active_mask = torch.ones(seeds_plot.shape[0], dtype=torch.bool)
        else:
            active_mask = active_mask_out.detach().cpu().bool()
        seed_weight_out = out.get("seed_active_weights")
        if seed_weight_out is None:
            seed_activity = torch.ones(seeds_plot.shape[0], dtype=torch.float32)
        else:
            seed_activity = seed_weight_out.detach().cpu().to(torch.float32).clamp(0.0, 1.0)

        rho_plot = out["rho"].detach().cpu()
        t_uv_plot = out["t_uv_raw"].detach().cpu()
        fiber_strength = out["fiber_strength"].detach().cpu()

        fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
        ax_rho, ax_fiber = axes[0]

        sc = ax_rho.scatter(
            uv_plot[:, 0],
            uv_plot[:, 1],
            c=rho_plot,
            s=10,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )

        if (~active_mask).any():
            ax_rho.scatter(
                seeds_plot[~active_mask, 0],
                seeds_plot[~active_mask, 1],
                s=90,
                c="lightgray",
                edgecolors="black",
                linewidths=1.0,
                label="inactive seed",
            )

        if active_mask.any():
            if color_seeds_by_activation:
                seed_sc = ax_rho.scatter(
                    seeds_plot[active_mask, 0],
                    seeds_plot[active_mask, 1],
                    s=95,
                    c=seed_activity[active_mask],
                    cmap=seed_cmap,
                    vmin=0.0,
                    vmax=1.0,
                    edgecolors="white",
                    linewidths=1.0,
                    label="active seed",
                )
                fig.colorbar(seed_sc, ax=ax_rho, fraction=0.046, pad=0.10, label="seed activity")
            else:
                ax_rho.scatter(
                    seeds_plot[active_mask, 0],
                    seeds_plot[active_mask, 1],
                    s=90,
                    c="red",
                    edgecolors="white",
                    linewidths=1.0,
                    label="active seed",
                )

        ax_rho.set_title("Density In UV")
        ax_rho.set_aspect("equal")
        ax_rho.set_xlabel("u")
        ax_rho.set_ylabel("v")
        fig.colorbar(sc, ax=ax_rho, fraction=0.046, pad=0.04, label="rho")

        if show_fiber_density_background:
            ax_fiber.scatter(
                uv_plot[:, 0],
                uv_plot[:, 1],
                c=rho_plot,
                s=8,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                alpha=0.35,
            )

        sample_mask = fiber_strength > fiber_min_strength
        if fiber_stride > 1:
            stride_mask = torch.zeros_like(sample_mask, dtype=torch.bool)
            stride_mask[::fiber_stride] = True
            sample_mask = sample_mask & stride_mask

        if bool(sample_mask.any()):
            uv_s = uv_plot[sample_mask]
            t_uv_s = t_uv_plot[sample_mask]
            strength_s = fiber_strength[sample_mask]

            ax_fiber.quiver(
                uv_s[:, 0].numpy(),
                uv_s[:, 1].numpy(),
                t_uv_s[:, 0].numpy(),
                t_uv_s[:, 1].numpy(),
                strength_s.numpy(),
                cmap=cmap,
                angles="xy",
                scale_units="xy",
                scale=max(fiber_scale, 1e-8) ** -1,
                width=0.003,
                pivot="mid",
            )

        if (~active_mask).any():
            ax_fiber.scatter(
                seeds_plot[~active_mask, 0],
                seeds_plot[~active_mask, 1],
                s=90,
                c="lightgray",
                edgecolors="black",
                linewidths=1.0,
            )

        if active_mask.any():
            if color_seeds_by_activation:
                ax_fiber.scatter(
                    seeds_plot[active_mask, 0],
                    seeds_plot[active_mask, 1],
                    s=95,
                    c=seed_activity[active_mask],
                    cmap=seed_cmap,
                    vmin=0.0,
                    vmax=1.0,
                    edgecolors="white",
                    linewidths=1.0,
                )
            else:
                ax_fiber.scatter(
                    seeds_plot[active_mask, 0],
                    seeds_plot[active_mask, 1],
                    s=90,
                    c="red",
                    edgecolors="white",
                    linewidths=1.0,
                )

        ax_fiber.set_title("Fiber Directions In UV")
        ax_fiber.set_aspect("equal")
        ax_fiber.set_xlabel("u")
        ax_fiber.set_ylabel("v")

        handles, labels = ax_rho.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
            fig.subplots_adjust(top=0.85, wspace=0.25)
        else:
            fig.subplots_adjust(wspace=0.25)

        return fig

    def plot_3d_fields(
        self,
        *,
        out: dict[str, torch.Tensor],
        seeds_raw,
        cmap: str = "viridis",
        window_size: tuple[int, int] = (1500, 700),
        clim: tuple[float, float] = (0.0, 1.0),
        show_edges: bool = False,
        fiber_stride: int = 20,
        fiber_scale: float = 0.08,
        fiber_min_strength: float = 0.05,
        show_fiber_density_background: bool = True,
        color_seeds_by_activation: bool = True,
        seed_cmap: str = "plasma",
        query: MeshQueryData | None = None,
    ):
        q = self.query if query is None else query

        seed_xyz = self.seeds_uv_to_xyz_nearest(
            seeds_uv=self._to_tensor(seeds_raw, dtype=self.dtype),
            uv=q.points_uv,
            points_xyz=q.points_xyz,
        )
        active_mask_out = out.get("seed_active_mask")
        if active_mask_out is None:
            active_mask = torch.ones(seed_xyz.shape[0], dtype=torch.bool)
        else:
            active_mask = active_mask_out.detach().cpu().bool()
        seed_weight_out = out.get("seed_active_weights")
        if seed_weight_out is None:
            seed_activity = torch.ones(seed_xyz.shape[0], dtype=torch.float32)
        else:
            seed_activity = seed_weight_out.detach().cpu().to(torch.float32).clamp(0.0, 1.0)

        pv_faces = self.faces_ijk_to_pv_faces(q.faces_ijk)
        mesh = pv.PolyData(
            q.points_xyz.detach().cpu().numpy(),
            pv_faces,
        )
        mesh["rho"] = out["rho"].detach().cpu().numpy().astype(np.float32)

        plotter = pv.Plotter(shape=(1, 2), window_size=window_size)

        plotter.subplot(0, 0)
        plotter.add_text("Density In 3D", font_size=10)
        plotter.add_mesh(
            mesh.copy(),
            scalars="rho",
            cmap=cmap,
            clim=list(clim),
            show_edges=show_edges,
        )

        if active_mask.any():
            active_cloud = pv.PolyData(seed_xyz[active_mask].detach().cpu().numpy())
            if color_seeds_by_activation:
                active_cloud["seed_activity"] = seed_activity[active_mask].numpy().astype(np.float32)
                plotter.add_mesh(
                    active_cloud,
                    scalars="seed_activity",
                    cmap=seed_cmap,
                    clim=[0.0, 1.0],
                    render_points_as_spheres=True,
                    point_size=14,
                    scalar_bar_args={"title": "seed activity"},
                )
            else:
                plotter.add_mesh(
                    active_cloud,
                    color="red",
                    render_points_as_spheres=True,
                    point_size=14,
                )

        if (~active_mask).any():
            inactive_cloud = pv.PolyData(seed_xyz[~active_mask].detach().cpu().numpy())
            plotter.add_mesh(
                inactive_cloud,
                color="gray",
                opacity=0.45,
                render_points_as_spheres=True,
                point_size=12,
            )

        plotter.show_axes()

        plotter.subplot(0, 1)
        plotter.add_text("Fiber Directions In 3D", font_size=10)
        if show_fiber_density_background:
            plotter.add_mesh(
                mesh.copy(),
                scalars="rho",
                cmap=cmap,
                clim=list(clim),
                show_edges=show_edges,
                opacity=0.30,
            )

        fiber_xyz = out["fiber3d"].detach().cpu()
        fiber_strength = out["fiber_strength"].detach().cpu()
        sample_mask = fiber_strength > fiber_min_strength
        if fiber_stride > 1:
            stride_mask = torch.zeros_like(sample_mask, dtype=torch.bool)
            stride_mask[::fiber_stride] = True
            sample_mask = sample_mask & stride_mask

        if bool(sample_mask.any()):
            pts = q.points_xyz.detach().cpu()[sample_mask].numpy()
            vecs = fiber_xyz[sample_mask].numpy()
            mags = fiber_strength[sample_mask].numpy().astype(np.float32)

            fiber_cloud = pv.PolyData(pts)
            fiber_cloud["vectors"] = vecs
            fiber_cloud["strength"] = mags

            glyphs = fiber_cloud.glyph(
                orient="vectors",
                scale="strength",
                factor=fiber_scale,
            )
            plotter.add_mesh(glyphs, scalars="strength", cmap=cmap, clim=list(clim))

        if active_mask.any():
            active_cloud = pv.PolyData(seed_xyz[active_mask].detach().cpu().numpy())
            if color_seeds_by_activation:
                active_cloud["seed_activity"] = seed_activity[active_mask].numpy().astype(np.float32)
                plotter.add_mesh(
                    active_cloud,
                    scalars="seed_activity",
                    cmap=seed_cmap,
                    clim=[0.0, 1.0],
                    render_points_as_spheres=True,
                    point_size=14,
                    show_scalar_bar=False,
                )
            else:
                plotter.add_mesh(
                    active_cloud,
                    color="red",
                    render_points_as_spheres=True,
                    point_size=14,
                )

        if (~active_mask).any():
            inactive_cloud = pv.PolyData(seed_xyz[~active_mask].detach().cpu().numpy())
            plotter.add_mesh(
                inactive_cloud,
                color="gray",
                opacity=0.45,
                render_points_as_spheres=True,
                point_size=12,
            )

        plotter.show_axes()
        plotter.link_views()
        return plotter

    def visualize_fields(
        self,
        *,
        seeds_raw,
        w_raw,
        h_raw=None,
        theta=None,
        a_raw=None,
        query: MeshQueryData | None = None,
        boundary_uv=None,
        boundary_face_id=None,
        boundary_width_raw=None,
        boundary_alpha_raw=None,
        boundary_beta_raw=None,
        centerline_radius_raw=None,
        centerline_radius_fixed=None,
        seed_domain_sdf=None,
        seed_domain_mask=None,
        seed_domain_mask_threshold=None,
        seed_domain_temp=None,
        point_domain_sdf=None,
        point_domain_mask=None,
        point_domain_mask_threshold=None,
        point_domain_temp=None,
        show_uv: bool = True,
        show_3d: bool = True,
        cmap: str = "viridis",
        fiber_stride: int = 20,
        fiber_scale_uv: float = 0.06,
        fiber_scale_3d: float = 0.08,
        fiber_min_strength: float = 0.05,
        show_fiber_density_background: bool = True,
        color_seeds_by_activation: bool = True,
        seed_cmap: str = "plasma",
        hard_seed_mask = False,
    ) -> dict[str, Any]:
        out = self.run_case(
            seeds_raw=seeds_raw,
            w_raw=w_raw,
            h_raw=h_raw,
            theta=theta,
            a_raw=a_raw,
            query=query,
            boundary_uv=boundary_uv,
            boundary_face_id=boundary_face_id,
            boundary_width_raw=boundary_width_raw,
            boundary_alpha_raw=boundary_alpha_raw,
            boundary_beta_raw=boundary_beta_raw,
            centerline_radius_raw=centerline_radius_raw,
            centerline_radius_fixed=centerline_radius_fixed,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=seed_domain_mask_threshold,
            seed_domain_temp=seed_domain_temp,
            point_domain_sdf=point_domain_sdf,
            point_domain_mask=point_domain_mask,
            point_domain_mask_threshold=point_domain_mask_threshold,
            point_domain_temp=point_domain_temp,
            hard_seed_mask= hard_seed_mask
        )

        result: dict[str, Any] = {"case": out}

        if show_uv:
            result["uv_fig"] = self.plot_uv_fields(
                out=out,
                seeds_raw=seeds_raw,
                cmap=cmap,
                fiber_stride=fiber_stride,
                fiber_scale=fiber_scale_uv,
                fiber_min_strength=fiber_min_strength,
                show_fiber_density_background=show_fiber_density_background,
                color_seeds_by_activation=color_seeds_by_activation,
                seed_cmap=seed_cmap,
                query=query,
            )

        if show_3d:
            result["plotter"] = self.plot_3d_fields(
                out=out,
                seeds_raw=seeds_raw,
                cmap=cmap,
                fiber_stride=fiber_stride,
                fiber_scale=fiber_scale_3d,
                fiber_min_strength=fiber_min_strength,
                show_fiber_density_background=show_fiber_density_background,
                color_seeds_by_activation=color_seeds_by_activation,
                seed_cmap=seed_cmap,
                query=query,
            )

        return result

class ExactUVVoronoi:
    def __init__(self, seeds_all, active_mask=None):
        self.seeds_all = seeds_all
        self.active_mask = active_mask

    def _to_numpy(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def get_seeds(self):
        seeds = self.seeds_all

        if self.active_mask is not None:
            mask = self.active_mask
            if torch.is_tensor(mask):
                mask = mask.detach().cpu().bool()
            seeds = seeds[mask]

        seeds_np = self._to_numpy(seeds).reshape(-1, 2)
        return seeds_np

    def show(self, figsize=(7, 7), title="Exact Voronoi Skeleton in UV"):
        seeds_np = self.get_seeds()

        print("seeds used:", seeds_np.shape[0])

        if self.active_mask is not None:
            mask = self.active_mask
            inactive = (~mask).sum().item() if torch.is_tensor(mask) else np.sum(~mask)
            print("inactive seeds:", inactive)

        if seeds_np.shape[0] < 4:
            raise ValueError("Voronoi needs at least 4 points in 2D.")

        vor = Voronoi(seeds_np, qhull_options="QJ")

        fig, ax = plt.subplots(figsize=figsize)

        voronoi_plot_2d(
            vor,
            ax=ax,
            show_vertices=False,
            show_points=True,
            line_width=1.5,
            line_alpha=0.8,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(title)
        ax.grid(False)

        return fig, ax
