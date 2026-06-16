from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        use_trim_activity: bool = True, #Whether to use CAD trim activity.
        return_xyz: bool = True, #Whether XYZ coordinates should be returned if a CAD domain is provided.
        empty_circle_margin: float | None = None, #Margin used in empty-circle validation. If None, becomes 0.5 * tau_voronoi.
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
        self.use_trim_activity = bool(use_trim_activity)
        self.return_xyz = bool(return_xyz)
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

        if triples.numel() == 0:
            return self._empty_output(
                seeds_uv,
                triples,
                cad_domain,
                want_xyz,
                u_periodic,
                v_periodic,
            )

        qi, qj, qk = self.unwrap_triple_seeds(seeds_uv, triples, u_periodic, v_periodic)
        P_unwrapped, P_uv, area2, pair_dists = self.circumcenters_from_triples(
            seeds_uv,
            triples,
            u_periodic,
            v_periodic,
        )

        if seed_activity is None:
            g_seed = torch.ones((triples.shape[0],), dtype=seeds_uv.dtype, device=seeds_uv.device)
        else:
            seed_activity = torch.as_tensor(seed_activity, dtype=seeds_uv.dtype, device=seeds_uv.device)
            if seed_activity.ndim != 1 or seed_activity.shape[0] != S:
                raise ValueError(
                    f"seed_activity must have shape [S], got {tuple(seed_activity.shape)}."
                )
            act = torch.clamp(seed_activity, 0.0, 1.0)
            g_seed = act[triples[:, 0]] * act[triples[:, 1]] * act[triples[:, 2]]

        g_close = self.close_gate(qi, qj, qk)
        g_area = self.area_gate(qi, qj, qk)
        g_box = self.box_gate(P_uv, u_periodic, v_periodic)
        g_trim = self.trim_gate(P_uv, cad_domain)
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

        alpha = (
            g_seed
            * g_close
            * g_area
            * g_box
            * g_trim
            * g_vor
            * g_keff
        )
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
    ) -> dict[str, Any]:
        empty = torch.empty((0,), dtype=seeds_uv.dtype, device=seeds_uv.device)
        vertices_uv = torch.empty((0, 2), dtype=seeds_uv.dtype, device=seeds_uv.device)
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


__all__ = ["ContinuousVoronoiDecoder"]
