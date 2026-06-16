from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree


class ContinuousVoronoiDecoder(nn.Module):
    """Differentiable UV Voronoi candidate-vertex decoder.

    This decoder only works in normalized UV seed coordinates. It returns every
    seed triple's smooth circumcenter candidate plus soft validity weights.
    tau_keff is the width of the Keff acceptance bell: small values sharply
    accept only Keff near 3, while larger values allow 4-seed degeneracies.
    """

    def __init__(
        self,
        eps: float = 1e-8, #Min value to avoid dead gradients.
        solve_reg: float = 1e-6, #Regularization for solving the linear system
        min_seed_dist: float = 1e-3, #Min acceptable distance between seeds.
        min_area: float = 1e-5, #Min triangle area before considering a triple valid.
        tau_close: float = 0.01, #Softness of the seed-distance validity gate. 
        tau_area: float = 0.01, #Softness of the area validity gate.
        tau_voronoi: float = 0.01, #Softness of the empty-circle (Voronoi) gate.
        tau_box: float = 0.01, #Softness of the UV boundary gate.
        tau_trim: float = 0.01, #Softness of CAD trim-region gate.
        tau_vertex_weight: float = 0.02, #Temperature for seed-weight computation around a vertex.
        tau_keff: float = 1.0, #Width of the effective-neighbor-count acceptance bell around 3.
        use_seed_weight_gate: bool = True, #Whether to apply the keff gate.
        use_close_gate: bool = False, #Whether to include close-seed rejection in alpha.
        use_trim_activity: bool = True, #Whether to use CAD trim activity.
        return_xyz: bool = True, #Whether XYZ coordinates should be returned if a CAD domain is provided.
        empty_circle_margin: float | None = None, #Margin used in empty-circle validation. If None, becomes 0.5 * tau_voronoi.
        duplicate_merge_sigma: float = 0.01,
        duplicate_effect_temp_ratio: float = 0.20,
        duplicate_effect_strength: float = 6.0,
        duplicate_effect_floor: float = 5e-2,
        seed_activity_sharpness: float = 1.0,
        seed_activity_threshold: float = 0.5,
        domain_effect_floor: float = 1e-8,
        hard_seed_mask: bool = False,
    ):
        super().__init__()
        self.eps = float(eps)
        self.solve_reg = float(solve_reg)
        self.min_seed_dist = float(min_seed_dist)
        self.min_area = float(min_area)
        self.tau_close = float(tau_close)
        self.tau_area = float(tau_area)
        self.tau_voronoi = float(tau_voronoi)
        self.tau_box = float(tau_box)
        self.tau_trim = float(tau_trim)
        self.tau_vertex_weight = float(tau_vertex_weight)
        self.tau_keff = float(tau_keff)
        self.use_seed_weight_gate = bool(use_seed_weight_gate)
        self.use_close_gate = bool(use_close_gate)
        self.use_trim_activity = bool(use_trim_activity)
        self.return_xyz = bool(return_xyz)
        self.duplicate_merge_sigma = float(duplicate_merge_sigma)
        self.duplicate_effect_temp_ratio = float(duplicate_effect_temp_ratio)
        self.duplicate_effect_strength = float(duplicate_effect_strength)
        self.duplicate_effect_floor = float(duplicate_effect_floor)
        self.seed_activity_sharpness = float(seed_activity_sharpness)
        self.seed_activity_threshold = float(seed_activity_threshold)
        self.domain_effect_floor = float(domain_effect_floor)
        self.hard_seed_mask = bool(hard_seed_mask)
        if empty_circle_margin is None:
            empty_circle_margin = 0.5 * tau_voronoi

        self.empty_circle_margin = float(empty_circle_margin)

    def _tau_tensor(self, value: float, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(max(float(value), self.eps), dtype=ref.dtype, device=ref.device)

    def make_triples(self, S: int, device: torch.device | str) -> torch.Tensor:
        if S < 3:
            return torch.empty((0, 3), dtype=torch.long, device=device)
        return torch.combinations(torch.arange(S, device=device), r=3)

    def periodic_difference(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> torch.Tensor:
        diff = a - b
        if u_periodic:
            diff_u = diff[..., 0] - torch.round(diff[..., 0])
            diff = torch.cat((diff_u.unsqueeze(-1), diff[..., 1:2]), dim=-1)
        if v_periodic:
            diff_v = diff[..., 1] - torch.round(diff[..., 1])
            diff = torch.cat((diff[..., 0:1], diff_v.unsqueeze(-1)), dim=-1)
        return diff

    def periodic_distance(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> torch.Tensor:
        diff = self.periodic_difference(a, b, u_periodic, v_periodic)
        return torch.sqrt((diff * diff).sum(dim=-1) + self.eps)

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

    def _seed_domain_activity(
        self,
        seeds_uv: torch.Tensor,
        cad_domain: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cad_domain is None:
            ones = torch.ones((seeds_uv.shape[0],), dtype=seeds_uv.dtype, device=seeds_uv.device)
            return ones, ones

        if hasattr(cad_domain, "sample_trim_sdf"):
            sdf = cad_domain.sample_trim_sdf(seeds_uv)
            sdf = torch.as_tensor(sdf, dtype=seeds_uv.dtype, device=seeds_uv.device)
            domain_weight = torch.sigmoid(sdf / self._tau_tensor(self.tau_trim, seeds_uv))
        elif hasattr(cad_domain, "smooth_inside_activity"):
            domain_weight = cad_domain.smooth_inside_activity(seeds_uv, tau=self.tau_trim)
            domain_weight = torch.as_tensor(domain_weight, dtype=seeds_uv.dtype, device=seeds_uv.device)
            sdf = torch.empty((0,), dtype=seeds_uv.dtype, device=seeds_uv.device)
        else:
            ones = torch.ones((seeds_uv.shape[0],), dtype=seeds_uv.dtype, device=seeds_uv.device)
            return ones, torch.empty((0,), dtype=seeds_uv.dtype, device=seeds_uv.device)

        domain_weight = domain_weight.clamp(0.0, 1.0)
        domain_floor = torch.as_tensor(
            self.domain_effect_floor,
            dtype=seeds_uv.dtype,
            device=seeds_uv.device,
        )
        domain_activity = domain_floor + (1.0 - domain_floor) * domain_weight
        return domain_activity.clamp(0.0, 1.0), sdf

    def _duplicate_seed_activity(
        self,
        seeds_uv: torch.Tensor,
        u_periodic: bool,
        v_periodic: bool,
    ) -> torch.Tensor:
        S = seeds_uv.shape[0]
        if S <= 1:
            return torch.ones((S,), dtype=seeds_uv.dtype, device=seeds_uv.device)

        d = self.periodic_distance(
            seeds_uv[:, None, :],
            seeds_uv[None, :, :],
            u_periodic,
            v_periodic,
        )

        radius = torch.as_tensor(
            self.duplicate_merge_sigma,
            dtype=seeds_uv.dtype,
            device=seeds_uv.device,
        ).clamp_min(self.eps)

        temp = (radius * float(self.duplicate_effect_temp_ratio)).clamp_min(self.eps)

        soft_close = torch.sigmoid((radius - d) / temp)
        eye = torch.eye(S, dtype=torch.bool, device=seeds_uv.device)
        soft_close = soft_close.masked_fill(eye, 0.0)

        lower_priority = torch.tril(
            torch.ones((S, S), dtype=seeds_uv.dtype, device=seeds_uv.device),
            diagonal=-1,
        )

        suppress_mass = (soft_close * lower_priority).sum(dim=1)

        raw_duplicate_weight = torch.exp(
            -float(self.duplicate_effect_strength) * suppress_mass
        )

        duplicate_floor = torch.as_tensor(
            self.duplicate_effect_floor,
            dtype=seeds_uv.dtype,
            device=seeds_uv.device,
        )

        duplicate_weight = duplicate_floor + (1.0 - duplicate_floor) * raw_duplicate_weight
        return duplicate_weight.clamp(0.0, 1.0)

    def _seed_activation_state(
        self,
        seeds_uv: torch.Tensor,
        cad_domain: Any | None = None,
        u_periodic: bool = False,
        v_periodic: bool = False,
        hard_seed_mask: bool | None = None,
    ) -> dict[str, torch.Tensor]:

        use_hard_seed_mask = (
            self.hard_seed_mask
            if hard_seed_mask is None
            else hard_seed_mask
        )

        domain_activity, seed_sdf = self._seed_domain_activity(seeds_uv, cad_domain)

        duplicate_weight = self._duplicate_seed_activity(
            seeds_uv,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )

        weights = duplicate_weight * domain_activity
        weights = self._sharpen_seed_activity(weights)

        if use_hard_seed_mask:
            hard_active = (domain_activity > self.seed_activity_threshold) & (
                duplicate_weight > self.seed_activity_threshold
            )
            weights = weights * hard_active.to(seeds_uv.dtype)

        return {
            "weights": weights.clamp(0.0, 1.0),
            "domain_activity": domain_activity.clamp(0.0, 1.0),
            "duplicate_weight": duplicate_weight.clamp(0.0, 1.0),
            "seed_sdf": seed_sdf,
        }

    def unwrap_triple_seeds(
        self,
        seeds_uv: torch.Tensor,
        triples: torch.Tensor,
        u_periodic: bool,
        v_periodic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qi = seeds_uv[triples[:, 0]]
        sj = seeds_uv[triples[:, 1]]
        sk = seeds_uv[triples[:, 2]]
        qj = qi + self.periodic_difference(sj, qi, u_periodic, v_periodic)
        qk = qi + self.periodic_difference(sk, qi, u_periodic, v_periodic)
        return qi, qj, qk

    def wrap_uv(self, P: torch.Tensor, u_periodic: bool, v_periodic: bool) -> torch.Tensor:
        if not (u_periodic or v_periodic):
            return P

        P_uv = P.clone()
        if u_periodic:
            P_uv[..., 0] = torch.remainder(P_uv[..., 0], 1.0)
        if v_periodic:
            P_uv[..., 1] = torch.remainder(P_uv[..., 1], 1.0)
        return P_uv

    def circumcenters_from_triples(
        self,
        seeds_uv: torch.Tensor,
        triples: torch.Tensor,
        u_periodic: bool,
        v_periodic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        qi, qj, qk = self.unwrap_triple_seeds(seeds_uv, triples, u_periodic, v_periodic)

        row_j = 2.0 * (qj - qi)
        row_k = 2.0 * (qk - qi)
        A = torch.stack((row_j, row_k), dim=-2)

        qi2 = (qi * qi).sum(dim=-1)
        b = torch.stack(
            (
                (qj * qj).sum(dim=-1) - qi2,
                (qk * qk).sum(dim=-1) - qi2,
            ),
            dim=-1,
        )

        At = A.transpose(-1, -2)
        normal = At @ A
        rhs = (At @ b.unsqueeze(-1)).squeeze(-1)
        eye = torch.eye(2, dtype=seeds_uv.dtype, device=seeds_uv.device).expand_as(normal)
        reg = max(self.solve_reg, self.eps)
        normal = normal + reg * eye

        P_unwrapped = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
        P_unwrapped = torch.nan_to_num(P_unwrapped, nan=0.0, posinf=0.0, neginf=0.0)
        P_uv = self.wrap_uv(P_unwrapped, u_periodic, v_periodic)

        area2 = torch.abs(self._cross2(qj - qi, qk - qi))
        pair_dists = {
            "dij": self.periodic_distance(qi, qj, False, False),
            "dik": self.periodic_distance(qi, qk, False, False),
            "djk": self.periodic_distance(qj, qk, False, False),
        }
        return P_unwrapped, P_uv, area2, pair_dists

    def _cross2(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def close_gate(self, qi: torch.Tensor, qj: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
        dij = self.periodic_distance(qi, qj, False, False)
        dik = self.periodic_distance(qi, qk, False, False)
        djk = self.periodic_distance(qj, qk, False, False)
        tau = self._tau_tensor(self.tau_close, qi)
        return (
            torch.sigmoid((dij - self.min_seed_dist) / tau)
            * torch.sigmoid((dik - self.min_seed_dist) / tau)
            * torch.sigmoid((djk - self.min_seed_dist) / tau)
        )

    def area_gate(self, qi: torch.Tensor, qj: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
        area2 = torch.abs(self._cross2(qj - qi, qk - qi))
        tau = self._tau_tensor(self.tau_area, qi)
        return torch.sigmoid((area2 - self.min_area) / tau)

    def box_gate(self, P_uv: torch.Tensor, u_periodic: bool, v_periodic: bool) -> torch.Tensor:
        gate = torch.ones(P_uv.shape[:-1], dtype=P_uv.dtype, device=P_uv.device)
        tau = self._tau_tensor(self.tau_box, P_uv)

        if not u_periodic:
            u = P_uv[..., 0]
            gate = gate * torch.sigmoid(u / tau) * torch.sigmoid((1.0 - u) / tau)
        if not v_periodic:
            v = P_uv[..., 1]
            gate = gate * torch.sigmoid(v / tau) * torch.sigmoid((1.0 - v) / tau)
        return gate

    def trim_gate(self, P_uv: torch.Tensor, cad_domain: Any | None) -> torch.Tensor:
        if cad_domain is None or not self.use_trim_activity:
            return torch.ones(P_uv.shape[:-1], dtype=P_uv.dtype, device=P_uv.device)

        g_trim = cad_domain.smooth_inside_activity(P_uv, tau=self.tau_trim)
        return g_trim.to(dtype=P_uv.dtype, device=P_uv.device)
    def vertex_soft_competition_gate(
        self,
        P_uv: torch.Tensor,
        alpha_base: torch.Tensor,
        sigma: float = 0.01,
        temperature: float = 0.05,
        floor: float = 0.05,
    ) -> torch.Tensor:
        """
        Softly suppress nearby duplicate vertices using alpha-based competition.

        P_uv: [M, 2]
        alpha_base: [M], alpha before competition
        returns: [M] gate in [floor, 1]
        """

        M = P_uv.shape[0]
        if M == 0:
            return torch.empty((0,), dtype=P_uv.dtype, device=P_uv.device)

        d = torch.cdist(P_uv, P_uv)  # [M, M]

        # nearby candidates compete
        sim = torch.exp(-(d / sigma).pow(2))  # [M, M]

        # candidate strength comes from alpha, not index
        score = torch.log(alpha_base.clamp_min(self.eps)) / temperature  # [M]

        # for each vertex i, compare it against nearby vertices j
        logits = score[None, :] + torch.log(sim.clamp_min(self.eps))

        ownership = torch.softmax(logits, dim=1)  # [M, M]

        gate = ownership.diagonal()

        return floor + (1.0 - floor) * gate.clamp(0.0, 1.0)
    def empty_circle_gate(
        self,
        seeds_uv: torch.Tensor,
        P_uv: torch.Tensor,
        triples: torch.Tensor,
        u_periodic: bool,
        v_periodic: bool,
    ) -> torch.Tensor:
        if triples.numel() == 0:
            return torch.empty((0,), dtype=seeds_uv.dtype, device=seeds_uv.device)

        r = self.periodic_distance(
            P_uv,
            seeds_uv[triples[:, 0]],
            u_periodic,
            v_periodic,
        )
        all_dist = self.periodic_distance(
            P_uv[:, None, :],
            seeds_uv[None, :, :],
            u_periodic,
            v_periodic,
        )
        delta = all_dist - r[:, None]

        in_triple = torch.zeros_like(delta, dtype=torch.bool)
        in_triple.scatter_(1, triples, True)
        large_positive = torch.full_like(delta, 1.0e6)
        delta = torch.where(in_triple, large_positive, delta)

        tau = self._tau_tensor(self.tau_voronoi, seeds_uv)
        margin = torch.as_tensor(
            self.empty_circle_margin,
            device=delta.device,
            dtype=delta.dtype,
        )

        log_g = F.logsigmoid((delta + margin) / tau).sum(dim=-1)
        return torch.exp(log_g)
    
    def vertex_soft_nms_alpha(
        self,
        P_uv: torch.Tensor,
        alpha_base: torch.Tensor,
        sigma: float = 0.012,
    ) -> torch.Tensor:
        d = torch.cdist(P_uv, P_uv)
        sim = torch.exp(-(d / sigma).pow(2))

        denom = sim @ alpha_base.clamp_min(self.eps)
        alpha_nms = alpha_base / denom.clamp_min(self.eps)

        return alpha_nms.clamp(0.0, 1.0)


    def seed_weight_vertex_gate(
        self,
        seeds_uv: torch.Tensor,
        P_uv: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.periodic_distance(
            P_uv[:, None, :],
            seeds_uv[None, :, :],
            u_periodic,
            v_periodic,
        )

        tau_w = self._tau_tensor(self.tau_vertex_weight, seeds_uv)
        w = torch.softmax(-d / tau_w, dim=1)

        keff = 1.0 / w.pow(2).sum(dim=1).clamp_min(self.eps)

        tau_k = self._tau_tensor(self.tau_keff, seeds_uv)
        g_keff = torch.exp(-0.5 * ((keff - 3.0) / tau_k).pow(2))

        return g_keff, w, keff

    def forward(
        self,
        seeds_uv: torch.Tensor, #Tensor of seed points in normalized UV space, shape [S, 2].
        seed_activity: torch.Tensor | None = None, #Optional tensor of seed activity values, shape [S]
        cad_domain: Any | None = None, #Optional CAD/domain object used for trimming and UV-to-XYZ conversion.
        u_periodic: bool = False,
        v_periodic: bool = False,
        return_xyz: bool | None = None,
    ) -> dict[str, Any]:
        if not isinstance(seeds_uv, torch.Tensor):
            raise TypeError("seeds_uv must be a torch.Tensor.")
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError(f"seeds_uv must have shape [S, 2], got {tuple(seeds_uv.shape)}.")
        if not seeds_uv.is_floating_point():
            raise TypeError("seeds_uv must be a floating point tensor.")

        S = seeds_uv.shape[0]
        # make or possible combinations
        triples = self.make_triples(S, seeds_uv.device)
        want_xyz = self.return_xyz if return_xyz is None else bool(return_xyz)

        if seed_activity is None:
            seed_state = self._seed_activation_state(
                seeds_uv=seeds_uv,
                cad_domain=cad_domain,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
                hard_seed_mask=None,
            )
            act = seed_state["weights"]
        else:
            act = torch.as_tensor(seed_activity, dtype=seeds_uv.dtype, device=seeds_uv.device)
            if act.ndim != 1 or act.shape[0] != S:
                raise ValueError(f"seed_activity must have shape [S], got {tuple(act.shape)}.")
            act = act.clamp(0.0, 1.0)
            seed_state = {
                "weights": act,
                "domain_activity": torch.ones_like(act),
                "duplicate_weight": torch.ones_like(act),
                "seed_sdf": torch.empty((0,), dtype=seeds_uv.dtype, device=seeds_uv.device),
            }

        if triples.numel() == 0:
            return self._empty_output(
                seeds_uv,
                triples,
                cad_domain,
                want_xyz,
                u_periodic,
                v_periodic,
                seed_state,
            )

        qi, qj, qk = self.unwrap_triple_seeds(seeds_uv, triples, u_periodic, v_periodic)
        P_unwrapped, P_uv, area2, pair_dists = self.circumcenters_from_triples(
            seeds_uv,
            triples,
            u_periodic,
            v_periodic,
        )

        g_seed = (
            act[triples[:, 0]]
            * act[triples[:, 1]]
            * act[triples[:, 2]]
        )

        g_close = self.close_gate(qi, qj, qk)
        g_area = self.area_gate(qi, qj, qk)
        g_box = self.box_gate(P_uv, u_periodic, v_periodic)
        g_trim = self.trim_gate(P_uv, cad_domain)
        if cad_domain is None:
            g_domain = g_box
        else:
            g_domain = g_trim
        g_vor = self.empty_circle_gate(seeds_uv, P_uv, triples, u_periodic, v_periodic)

        if self.use_seed_weight_gate:
            g_keff, vertex_seed_weights, vertex_keff = self.seed_weight_vertex_gate(
                seeds_uv=seeds_uv,
                P_uv=P_uv,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
        else:
            g_keff = torch.ones_like(g_vor)
            vertex_seed_weights = torch.empty(
                (P_uv.shape[0], seeds_uv.shape[0]),
                dtype=seeds_uv.dtype,
                device=seeds_uv.device,
            )
            vertex_keff = torch.zeros(
                (P_uv.shape[0],),
                dtype=seeds_uv.dtype,
                device=seeds_uv.device,
            )

        alpha_base = (
            g_seed
            * g_area
            * g_domain
            * g_vor
        )
        if self.use_close_gate:
            alpha = alpha * g_close
        g_compete = self.vertex_soft_competition_gate(
            P_uv,
            alpha_base,
            sigma=0.01,
            temperature=0.05,
            floor=0.05,
        )

        alpha = alpha_base * g_compete
        
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        P_uv = torch.nan_to_num(P_uv, nan=0.0, posinf=0.0, neginf=0.0)

        topk = min(5, seeds_uv.shape[0])
        topk_weights, topk_seed_idx = torch.topk(vertex_seed_weights, k=topk, dim=1)

        validity = {
            "seed": g_seed,
            "close": g_close,
            "area": g_area,
            "box": g_box,
            "trim": g_trim,
            "empty_circle": g_vor,
            "keff": g_keff,
        }
        diagnostics = {
            "vertices_uv_unwrapped": P_unwrapped,
            "area2": area2,
            "pair_distances": pair_dists,
            "vertex_seed_weights": vertex_seed_weights,
            "vertex_keff": vertex_keff,
            "topk_seed_weights": topk_weights,
            "topk_seed_idx": topk_seed_idx,
            "num_seeds": S,
            "num_triples": triples.shape[0],
            "u_periodic": bool(u_periodic),
            "v_periodic": bool(v_periodic),
        }
        diagnostics.update({
            "seed_activity_weights": seed_state["weights"],
            "seed_domain_activity": seed_state["domain_activity"],
            "seed_duplicate_weight": seed_state["duplicate_weight"],
            "seed_sdf": seed_state["seed_sdf"],
            "triple_seed_activity": g_seed,

        })

        out: dict[str, Any] = {
            "vertices_uv": P_uv,
            "alpha": alpha,
            "triple_idx": triples,
            "validity": validity,
            "diagnostics": diagnostics,
        }

        if cad_domain is not None and want_xyz:
            xyz = cad_domain.eval_uv_norm_batch(P_uv, return_inside_mask=False)["xyz"]
            out["vertices_xyz"] = torch.as_tensor(xyz, dtype=seeds_uv.dtype, device=seeds_uv.device)

        return out

    def _empty_output(
        self,
        seeds_uv: torch.Tensor,
        triples: torch.Tensor,
        cad_domain: Any | None,
        want_xyz: bool,
        u_periodic: bool,
        v_periodic: bool,
        seed_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        empty = torch.empty((0,), dtype=seeds_uv.dtype, device=seeds_uv.device)
        vertices_uv = torch.empty((0, 2), dtype=seeds_uv.dtype, device=seeds_uv.device)
        if seed_state is None:
            ones = torch.ones((seeds_uv.shape[0],), dtype=seeds_uv.dtype, device=seeds_uv.device)
            seed_state = {
                "weights": ones,
                "domain_activity": ones,
                "duplicate_weight": ones,
                "seed_sdf": empty,
            }
        validity = {
            "seed": empty,
            "close": empty,
            "area": empty,
            "box": empty,
            "trim": empty,
            "empty_circle": empty,
            "keff": empty,
        }
        diagnostics = {
            "vertices_uv_unwrapped": vertices_uv,
            "area2": empty,
            "pair_distances": {"dij": empty, "dik": empty, "djk": empty},
            "vertex_seed_weights": torch.empty(
                (0, seeds_uv.shape[0]),
                dtype=seeds_uv.dtype,
                device=seeds_uv.device,
            ),
            "vertex_keff": empty,
            "topk_seed_weights": torch.empty(
                (0, min(5, seeds_uv.shape[0])),
                dtype=seeds_uv.dtype,
                device=seeds_uv.device,
            ),
            "topk_seed_idx": torch.empty(
                (0, min(5, seeds_uv.shape[0])),
                dtype=torch.long,
                device=seeds_uv.device,
            ),
            "num_seeds": seeds_uv.shape[0],
            "num_triples": 0,
            "u_periodic": bool(u_periodic),
            "v_periodic": bool(v_periodic),
        }
        diagnostics.update({
            "seed_activity_weights": seed_state["weights"],
            "seed_domain_activity": seed_state["domain_activity"],
            "seed_duplicate_weight": seed_state["duplicate_weight"],
            "seed_sdf": seed_state["seed_sdf"],
            "triple_seed_activity": empty,
        })
        out: dict[str, Any] = {
            "vertices_uv": vertices_uv,
            "alpha": empty,
            "triple_idx": triples,
            "validity": validity,
            "diagnostics": diagnostics,
        }
        if cad_domain is not None and want_xyz:
            out["vertices_xyz"] = torch.empty((0, 3), dtype=seeds_uv.dtype, device=seeds_uv.device)
        return out
    
    def plot_voronoi_debug(
    self,
    seeds_uv,
    out=None,
    cad_domain=None,
    alpha_threshold=0.5,
    trim_res=300,
    miss_tol=0.02,
):
        if out is None:
            out = self(
                seeds_uv,
                cad_domain=cad_domain,
                u_periodic=False,
                v_periodic=False,
                return_xyz=False,
            )

        seeds_np = seeds_uv.detach().cpu().numpy()
        vertices = out["vertices_uv"]
        alpha = out["alpha"]

        vertices_np = vertices.detach().cpu().numpy()
        alpha_np = alpha.detach().cpu().numpy()

        keep = alpha > alpha_threshold
        pred_np = vertices[keep].detach().cpu().numpy()
        pred_alpha_np = alpha[keep].detach().cpu().numpy()

        vor = Voronoi(seeds_np)
        exact_np = vor.vertices

        inside_exact = (
            (exact_np[:, 0] >= 0.0) & (exact_np[:, 0] <= 1.0) &
            (exact_np[:, 1] >= 0.0) & (exact_np[:, 1] <= 1.0)
        )

        if cad_domain is not None and hasattr(cad_domain, "smooth_inside_activity"):
            exact_t = torch.as_tensor(exact_np, dtype=seeds_uv.dtype, device=seeds_uv.device)
            trim_exact = cad_domain.smooth_inside_activity(exact_t, tau=self.tau_trim)
            inside_exact = inside_exact & (trim_exact.detach().cpu().numpy() > alpha_threshold)

        exact_inside_np = exact_np[inside_exact]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        def draw_trim(ax):
            if cad_domain is None or not hasattr(cad_domain, "smooth_inside_activity"):
                return
            u = torch.linspace(0, 1, trim_res, device=seeds_uv.device, dtype=seeds_uv.dtype)
            v = torch.linspace(0, 1, trim_res, device=seeds_uv.device, dtype=seeds_uv.dtype)
            uu, vv = torch.meshgrid(u, v, indexing="xy")
            grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
            activity = cad_domain.smooth_inside_activity(grid, tau=self.tau_trim)
            activity = activity.reshape(trim_res, trim_res).detach().cpu().numpy()
            ax.contourf(
                np.linspace(0, 1, trim_res),
                np.linspace(0, 1, trim_res),
                activity,
                levels=[0.5, 1.0],
                alpha=0.15,
            )
            ax.contour(
                np.linspace(0, 1, trim_res),
                np.linspace(0, 1, trim_res),
                activity,
                levels=[0.5],
                linewidths=1,
            )

        # 1) Predicted points colored by alpha
        ax = axes[0]
        draw_trim(ax)
        sc = ax.scatter(
            vertices_np[:, 0],
            vertices_np[:, 1],
            c=alpha_np,
            s=30,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax.scatter(seeds_np[:, 0], seeds_np[:, 1], c="red", s=50, label="Seeds")
        ax.set_title("Predicted vertices colored by alpha")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # 2) Exact Voronoi diagram
        ax = axes[1]
        draw_trim(ax)
        voronoi_plot_2d(
            vor,
            ax=ax,
            show_vertices=False,
            show_points=False,
            line_colors="black",
            line_width=1,
            line_alpha=0.7,
            point_size=0,
        )
        ax.scatter(seeds_np[:, 0], seeds_np[:, 1], c="red", s=50, label="Seeds")
        ax.scatter(
            exact_inside_np[:, 0],
            exact_inside_np[:, 1],
            facecolors="none",
            edgecolors="blue",
            s=80,
            label="Exact vertices",
        )
        ax.set_title("Exact SciPy Voronoi diagram")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        # 3) Thresholded predicted vs exact
        ax = axes[2]
        draw_trim(ax)
        voronoi_plot_2d(
            vor,
            ax=ax,
            show_vertices=False,
            show_points=False,
            line_colors="black",
            line_width=1,
            line_alpha=0.4,
            point_size=0,
        )
        ax.scatter(seeds_np[:, 0], seeds_np[:, 1], c="red", s=50, label="Seeds")
        ax.scatter(
            exact_inside_np[:, 0],
            exact_inside_np[:, 1],
            facecolors="none",
            edgecolors="blue",
            s=90,
            label="Exact vertices",
        )
        ax.scatter(
            pred_np[:, 0],
            pred_np[:, 1],
            c="orange",
            marker="x",
            s=120,
            label=f"Predicted alpha > {alpha_threshold}",
        )
        ax.set_title("Thresholded prediction on exact Voronoi")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend()

        plt.tight_layout()
        plt.show()

        if len(pred_np) > 0 and len(exact_inside_np) > 0:
            tree_exact = cKDTree(exact_inside_np)
            pred_to_exact_dist, _ = tree_exact.query(pred_np)

            tree_pred = cKDTree(pred_np)
            exact_to_pred_dist, exact_to_pred_idx = tree_pred.query(exact_inside_np)

            print("Predicted active vertices:", len(pred_np))
            print("Exact inside vertices:", len(exact_inside_np))
            print("Pred -> exact mean:", pred_to_exact_dist.mean())
            print("Pred -> exact max :", pred_to_exact_dist.max())
            print("Exact -> pred mean:", exact_to_pred_dist.mean())
            print("Exact -> pred max :", exact_to_pred_dist.max())

            missed = exact_to_pred_dist > miss_tol

            if missed.any():
                print("\nMissing exact vertices:")
                print(f"miss_tol = {miss_tol}")

                tree_all = cKDTree(vertices_np)

                validity = out.get("validity", {})
                diagnostics = out.get("diagnostics", {})

                for exact_id in np.where(missed)[0]:
                    exact_v = exact_inside_np[exact_id]

                    d_all, cand_idx = tree_all.query(exact_v)

                    print("\n--------------------------------")
                    print(f"Exact vertex {exact_id}")
                    print(f"exact uv        = ({exact_v[0]:.6f}, {exact_v[1]:.6f})")
                    print(f"nearest pred d  = {exact_to_pred_dist[exact_id]:.6f}")
                    print(f"nearest raw idx = {cand_idx}")
                    print(f"nearest raw uv  = ({vertices_np[cand_idx,0]:.6f}, {vertices_np[cand_idx,1]:.6f})")
                    print(f"raw distance    = {d_all:.6f}")
                    print(f"raw alpha       = {alpha_np[cand_idx]:.6e}")

                    if "triple_idx" in out:
                        triple = out["triple_idx"][cand_idx].detach().cpu().numpy()
                        print(f"triple_idx      = {triple.tolist()}")

                    for name, value in validity.items():
                        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > cand_idx:
                            print(f"gate {name:>12s} = {value[cand_idx].detach().cpu().item():.6e}")

                    if "vertex_keff" in diagnostics:
                        keff = diagnostics["vertex_keff"]
                        if torch.is_tensor(keff) and keff.ndim > 0 and keff.shape[0] > cand_idx:
                            print(f"vertex_keff     = {keff[cand_idx].detach().cpu().item():.6f}")

                    if "alpha_base" in diagnostics:
                        ab = diagnostics["alpha_base"]
                        if torch.is_tensor(ab) and ab.ndim > 0 and ab.shape[0] > cand_idx:
                            print(f"alpha_base      = {ab[cand_idx].detach().cpu().item():.6e}")

                    if "vertex_competition_gate" in diagnostics:
                        cg = diagnostics["vertex_competition_gate"]
                        if torch.is_tensor(cg) and cg.ndim > 0 and cg.shape[0] > cand_idx:
                            print(f"competition     = {cg[cand_idx].detach().cpu().item():.6e}")
            else:
                print(f"\nNo missing exact vertices with miss_tol={miss_tol}.")

        else:
            print("Cannot compare: one set is empty.")
            print("Predicted active vertices:", len(pred_np))
            print("Exact inside vertices:", len(exact_inside_np))

__all__ = ["ContinuousVoronoiDecoder"]
