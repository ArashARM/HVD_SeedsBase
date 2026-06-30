from __future__ import annotations
from typing import Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, cKDTree

class ContinuousVoronoiDecoder(nn.Module):
    """Differentiable UV Voronoi decoder using SciPy topology and Torch geometry."""

    def __init__(self, eps: float=1e-08, solve_reg: float=1e-06, tau_voronoi: float=0.01, tau_box: float=0.01, tau_trim: float=0.01, use_trim_activity: bool=True, return_xyz: bool=True, vertex_boundary_margin: float=0.02, edge_trim_samples: int=32, edge_trim_reduction: str='softmin', edge_trim_reduce_tau: float=0.05, use_edge_trim_gate: bool=True, n_seeds: int | None=None, w_min: float=0.02, w_max_ratio: float=0.5, raw_temp: float=1.0, beta: float=0.02, centerline_softmin_tau: float=0.02, centerline_beta: float | None=None, tube_curve_samples: int=64, tube_lift_tau: float=0.02, tube_distance_tau: float | None=None, tube_density_tau: float | None=None, tube_fiber_tau: float | None=None, rho_min: float=0.0, face_u_periodic: Any=False, face_v_periodic: Any=False, nearest_segment_k: int=4, use_segment_distance: bool=True, use_spatial_pruning: bool=True, min_tube_spacing: float=1e-3, tube_target_spacing_ratio: float=0.75, use_seed_activation: bool=True, duplicate_merge_sigma: float=1e-4, duplicate_effect_temp_ratio: float=0.25, seed_domain_mask_threshold: float=0.5, min_active_seeds: int=3, boundary_snap_tol: float=1e-5, **unused_kwargs: Any):
        super().__init__()
        self.eps = float(eps)
        self.solve_reg = float(solve_reg)
        self.tau_voronoi = float(tau_voronoi)
        self.tau_box = float(tau_box)
        self.tau_trim = float(tau_trim)
        self.use_trim_activity = bool(use_trim_activity)
        self.return_xyz = bool(return_xyz)
        self.vertex_boundary_margin = float(vertex_boundary_margin)
        self.edge_trim_samples = int(edge_trim_samples)
        self.edge_trim_reduction = str(edge_trim_reduction)
        self.edge_trim_reduce_tau = float(edge_trim_reduce_tau)
        self.use_edge_trim_gate = bool(use_edge_trim_gate)
        self.n_seeds = None if n_seeds is None else int(n_seeds)
        self.w_min = float(w_min)
        self.w_max_ratio = float(w_max_ratio)
        self.raw_temp = float(raw_temp)
        self.beta = float(beta)
        self.centerline_softmin_tau = float(centerline_softmin_tau)
        self.centerline_beta = self.beta if centerline_beta is None else float(centerline_beta)
        self.tube_curve_samples = int(tube_curve_samples)
        self.tube_lift_tau = float(tube_lift_tau)
        self.tube_distance_tau = self.centerline_softmin_tau if tube_distance_tau is None else float(tube_distance_tau)
        self.tube_density_tau = self.centerline_beta if tube_density_tau is None else float(tube_density_tau)
        self.tube_fiber_tau = self.tube_distance_tau if tube_fiber_tau is None else float(tube_fiber_tau)
        self.rho_min = float(rho_min)
        self.face_u_periodic = face_u_periodic
        self.face_v_periodic = face_v_periodic
        self.nearest_segment_k = max(1, int(nearest_segment_k))
        self.use_segment_distance = bool(use_segment_distance)
        self.use_spatial_pruning = bool(use_spatial_pruning)
        self.min_tube_spacing = float(min_tube_spacing)
        self.tube_target_spacing_ratio = float(tube_target_spacing_ratio)
        self.use_seed_activation = bool(use_seed_activation)
        self.duplicate_merge_sigma = float(duplicate_merge_sigma)
        self.duplicate_effect_temp_ratio = float(duplicate_effect_temp_ratio)
        self.seed_domain_mask_threshold = float(seed_domain_mask_threshold)
        self.min_active_seeds = int(min_active_seeds)
        self.boundary_snap_tol = float(boundary_snap_tol)

    @staticmethod
    def _bool_value(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(value.detach().reshape(-1)[0].item()) if value.numel() > 0 else False
        if isinstance(value, (list, tuple)):
            return bool(value[0]) if value else False
        return bool(value)

    def _tau_tensor(self, value: float, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(max(float(value), self.eps), dtype=ref.dtype, device=ref.device)

    def _seed_domain_validity_state(
        self,
        seeds: torch.Tensor,
        temp: torch.Tensor,
        seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return:
            domain_weight: differentiable soft validity weight [S]
            domain_active: hard bool validity mask [S]
            sdf_values: SDF values or empty tensor [S]
            mask_values: mask values or empty tensor [S]
        """
        if seeds.ndim != 2 or seeds.shape[-1] != 2:
            raise ValueError(f'seeds must have shape [S, 2], got {tuple(seeds.shape)}.')
        s = seeds.shape[0]
        device = seeds.device
        dtype = seeds.dtype
        weight = torch.ones((s,), dtype=dtype, device=device)
        active = torch.ones((s,), dtype=torch.bool, device=device)
        empty = torch.empty((0,), dtype=dtype, device=device)
        sdf_values = empty
        mask_values = empty
        temp_t = torch.as_tensor(temp, dtype=dtype, device=device).clamp_min(self.eps)

        if seed_domain_sdf is not None:
            sdf_raw = seed_domain_sdf(seeds) if callable(seed_domain_sdf) else seed_domain_sdf
            sdf_values = torch.as_tensor(sdf_raw, dtype=dtype, device=device).reshape(-1)
            if sdf_values.shape[0] != s:
                raise ValueError(
                    f'seed_domain_sdf must produce shape [{s}], got {tuple(sdf_values.shape)}.'
                )
            weight = weight * torch.sigmoid(sdf_values / temp_t)
            active = active & (sdf_values >= 0.0)

        if seed_domain_mask is not None:
            mask_raw = seed_domain_mask(seeds) if callable(seed_domain_mask) else seed_domain_mask
            mask_values = torch.as_tensor(mask_raw, dtype=dtype, device=device).reshape(-1)
            if mask_values.shape[0] != s:
                raise ValueError(
                    f'seed_domain_mask must produce shape [{s}], got {tuple(mask_values.shape)}.'
                )
            threshold = torch.as_tensor(
                float(seed_domain_mask_threshold),
                dtype=dtype,
                device=device,
            )
            weight = weight * torch.sigmoid((mask_values - threshold) / temp_t)
            active = active & (mask_values >= threshold)

        return weight, active, sdf_values, mask_values

    def _seed_activation_state(
        self,
        seeds: torch.Tensor,
        seed_domain_sdf: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed_domain_mask_threshold: float | None = None,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return:
            active_ids: original seed indices that survive filtering [A]
            active_mask: bool mask over original seeds [S]
            activity_weight: soft diagnostic weight [S]
        """
        if not isinstance(seeds, torch.Tensor):
            raise TypeError('seeds must be a torch.Tensor.')
        if seeds.ndim != 2 or seeds.shape[-1] != 2:
            raise ValueError(f'seeds must have shape [S, 2], got {tuple(seeds.shape)}.')
        if not seeds.is_floating_point():
            raise TypeError('seeds must be a floating point tensor.')
        s = seeds.shape[0]
        device = seeds.device
        dtype = seeds.dtype
        if s == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.bool, device=device),
                torch.empty((0,), dtype=dtype, device=device),
            )

        radius = torch.as_tensor(self.duplicate_merge_sigma, dtype=dtype, device=device)
        temp = (radius * float(self.duplicate_effect_temp_ratio)).clamp_min(self.eps)
        u = seeds[:, 0]
        v = seeds[:, 1]
        inside_box = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        outside_dist = torch.stack(
            [-u, u - 1.0, -v, v - 1.0, torch.zeros_like(u)],
            dim=0,
        ).amax(dim=0)
        box_weight = torch.where(
            outside_dist <= 0.0,
            torch.ones_like(outside_dist),
            torch.sigmoid(-outside_dist / temp),
        )
        domain_weight, domain_active, _, _ = self._seed_domain_validity_state(
            seeds=seeds,
            temp=temp,
            seed_domain_sdf=seed_domain_sdf,
            seed_domain_mask=seed_domain_mask,
            seed_domain_mask_threshold=(
                self.seed_domain_mask_threshold
                if seed_domain_mask_threshold is None
                else float(seed_domain_mask_threshold)
            ),
        )
        active_mask = inside_box & domain_active
        duplicate_candidate_mask = active_mask.clone()
        candidate_ids = torch.nonzero(active_mask, as_tuple=False).flatten()
        keep = torch.ones(candidate_ids.shape[0], dtype=torch.bool, device=device)

        for local_i in range(candidate_ids.shape[0]):
            if not bool(keep[local_i].detach().cpu().item()):
                continue
            i = candidate_ids[local_i]
            pi = seeds[i]
            for local_j in range(local_i + 1, candidate_ids.shape[0]):
                if not bool(keep[local_j].detach().cpu().item()):
                    continue
                j = candidate_ids[local_j]
                pj = seeds[j]
                if u_periodic or v_periodic:
                    d = self.periodic_distance(pi, pj, u_periodic=u_periodic, v_periodic=v_periodic)
                else:
                    d = torch.linalg.vector_norm(pi - pj)
                if bool((d < radius).detach().cpu().item()):
                    keep[local_j] = False

        active_ids = candidate_ids[keep]
        active_mask = torch.zeros((s,), dtype=torch.bool, device=device)
        active_mask[active_ids] = True

        duplicate_weight = torch.ones((s,), dtype=dtype, device=device)
        if s > 1:
            diff = self.periodic_difference(
                seeds[:, None, :],
                seeds[None, :, :],
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
            dist = torch.sqrt((diff * diff).sum(dim=-1) + self.eps)
            soft_close = torch.sigmoid((radius - dist) / temp)
            soft_close = soft_close.masked_fill(torch.eye(s, dtype=torch.bool, device=device), 0.0)
            candidate_pair = duplicate_candidate_mask[:, None] & duplicate_candidate_mask[None, :]
            lower_priority = torch.tril(torch.ones((s, s), dtype=torch.bool, device=device), diagonal=-1)
            suppress_mass = (soft_close * (candidate_pair & lower_priority).to(dtype)).sum(dim=1)
            duplicate_weight = torch.exp(-suppress_mass)

        activity_weight = box_weight * domain_weight * duplicate_weight
        return active_ids, active_mask, activity_weight

    def periodic_difference(self, a: torch.Tensor, b: torch.Tensor, u_periodic: bool=False, v_periodic: bool=False) -> torch.Tensor:
        diff = a - b
        if u_periodic:
            diff_u = diff[..., 0] - torch.round(diff[..., 0])
            diff = torch.cat((diff_u.unsqueeze(-1), diff[..., 1:2]), dim=-1)
        if v_periodic:
            diff_v = diff[..., 1] - torch.round(diff[..., 1])
            diff = torch.cat((diff[..., 0:1], diff_v.unsqueeze(-1)), dim=-1)
        return diff

    def periodic_distance(self, a: torch.Tensor, b: torch.Tensor, u_periodic: bool=False, v_periodic: bool=False) -> torch.Tensor:
        diff = self.periodic_difference(a, b, u_periodic, v_periodic)
        return torch.sqrt((diff * diff).sum(dim=-1) + self.eps)

    def unwrap_edge_uv(self, p0: torch.Tensor, p1: torch.Tensor, u_periodic: bool=False, v_periodic: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an edge endpoint pair on the nearest periodic UV image."""
        p1_unwrapped = p0 + self.periodic_difference(p1, p0, u_periodic=u_periodic, v_periodic=v_periodic)
        return (p0, p1_unwrapped)

    def sample_boundary_box_edge_uv(self, p0: torch.Tensor, p1: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Sample a UV-box boundary path between two boundary points."""
        if p0.shape != (2,) or p1.shape != (2,):
            raise ValueError('p0 and p1 must each have shape [2].')
        if p0.device != p1.device or p0.dtype != p1.dtype:
            raise ValueError('p0 and p1 must share dtype and device.')
        if n_samples < 2:
            raise ValueError('n_samples must be at least 2.')
        tol = 1e-05
        zero = p0.new_tensor(0.0)
        one = p0.new_tensor(1.0)
        s = torch.linspace(0.0, 1.0, n_samples, dtype=p0.dtype, device=p0.device)
        same_left = bool((torch.abs(p0[0]) <= tol) & (torch.abs(p1[0]) <= tol))
        same_right = bool((torch.abs(p0[0] - 1.0) <= tol) & (torch.abs(p1[0] - 1.0) <= tol))
        same_bottom = bool((torch.abs(p0[1]) <= tol) & (torch.abs(p1[1]) <= tol))
        same_top = bool((torch.abs(p0[1] - 1.0) <= tol) & (torch.abs(p1[1] - 1.0) <= tol))
        if same_left:
            curve = torch.stack((torch.zeros_like(s), p0[1] + s * (p1[1] - p0[1])), dim=-1)
        elif same_right:
            curve = torch.stack((torch.ones_like(s), p0[1] + s * (p1[1] - p0[1])), dim=-1)
        elif same_bottom:
            curve = torch.stack((p0[0] + s * (p1[0] - p0[0]), torch.zeros_like(s)), dim=-1)
        elif same_top:
            curve = torch.stack((p0[0] + s * (p1[0] - p0[0]), torch.ones_like(s)), dim=-1)
        else:
            if n_samples == 2:
                return torch.stack((p0, p1)).clamp(0.0, 1.0)
            corners = torch.stack((torch.stack((zero, zero)), torch.stack((one, zero)), torch.stack((one, one)), torch.stack((zero, one))))
            corner_distance = torch.linalg.vector_norm(corners - p0, dim=-1) + torch.linalg.vector_norm(corners - p1, dim=-1)
            corner = corners[torch.argmin(corner_distance)]
            first_count = n_samples // 2 + 1
            second_count = n_samples - first_count + 1
            first_s = torch.linspace(0.0, 1.0, first_count, dtype=p0.dtype, device=p0.device)[:, None]
            second_s = torch.linspace(0.0, 1.0, second_count, dtype=p0.dtype, device=p0.device)[:, None]
            first = p0 + first_s * (corner - p0)
            second = corner + second_s * (p1 - corner)
            curve = torch.cat((first[:-1], second), dim=0)
        return curve.clamp(0.0, 1.0)

    def sample_smooth_edge_curves_uv(self, seeds_uv: torch.Tensor, vertices_uv: torch.Tensor, edges: torch.Tensor, edge_seed_pairs: torch.Tensor, n_samples: int=64, tangent_scale: float=0.5, u_periodic: bool=False, v_periodic: bool=False) -> torch.Tensor:
        """Sample differentiable straight Voronoi edge segments on fixed graph topology.

                Edge connectivity (possibly supplied by SciPy) is discrete and is not
                differentiable. Once that topology is fixed, vertex coordinates and
                straight edge samples remain differentiable with respect to seed
                positions.
                """
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError('seeds_uv must have shape [S, 2].')
        if vertices_uv.ndim != 2 or vertices_uv.shape[-1] != 2:
            raise ValueError('vertices_uv must have shape [V, 2].')
        if edges.ndim != 2 or edges.shape[-1] != 2:
            raise ValueError('edges must have shape [E, 2].')
        if edge_seed_pairs.shape != edges.shape:
            raise ValueError('edge_seed_pairs must have shape [E, 2].')
        if n_samples < 2:
            raise ValueError('n_samples must be at least 2.')
        if edges.device != vertices_uv.device or edge_seed_pairs.device != seeds_uv.device:
            raise ValueError('seeds, vertices, edges, and seed pairs must share a device.')
        num_edges = edges.shape[0]
        if num_edges == 0:
            return vertices_uv.new_empty((0, n_samples, 2))
        p0 = vertices_uv[edges[:, 0]]
        p1 = vertices_uv[edges[:, 1]]
        p0, p1 = self.unwrap_edge_uv(p0, p1, u_periodic=u_periodic, v_periodic=v_periodic)
        s = torch.linspace(0.0, 1.0, n_samples, dtype=vertices_uv.dtype, device=vertices_uv.device).view(1, n_samples, 1)
        curve = (1.0 - s) * p0[:, None, :] + s * p1[:, None, :]
        return self.wrap_uv(curve, u_periodic=u_periodic, v_periodic=v_periodic)

    def sample_graph_edge_curves_uv(self, seeds_uv: torch.Tensor, graph: dict[str, torch.Tensor], n_samples: int=64, tangent_scale: float=0.5, u_periodic: bool=False, v_periodic: bool=False) -> torch.Tensor:
        """Sample graph edges according to their discrete edge-type semantics.

                edge_type meanings:
                    0 = interior Voronoi edge
                    1 = interior-to-boundary clipped Voronoi edge
                    2 = reserved
                    3 = boundary-to-boundary clipped Voronoi edge
                    4 = boundary shell / UV box loop edge

                Only type 4 follows the UV-box boundary. Types 0, 1, and 3 are
                differentiable straight Voronoi edge segments.
                """
        nodes_uv = graph['nodes_uv']
        edge_index = graph['edge_index']
        edge_seed_pair = graph['edge_seed_pair']
        edge_type = graph.get('edge_type')
        if edge_type is None:
            edge_type = torch.zeros(edge_index.shape[0], dtype=torch.long, device=nodes_uv.device)
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[0]:
            raise ValueError("graph['edge_type'] must have shape [E].")
        curves = self.sample_smooth_edge_curves_uv(seeds_uv=seeds_uv, vertices_uv=nodes_uv, edges=edge_index, edge_seed_pairs=edge_seed_pair, n_samples=n_samples, tangent_scale=tangent_scale, u_periodic=u_periodic, v_periodic=v_periodic)
        shell_ids = torch.nonzero(edge_type == 4, as_tuple=False).flatten()
        if shell_ids.numel() == 0:
            return curves
        result_curves = []
        shell_id_set = set(shell_ids.detach().cpu().tolist())
        for edge_id in range(edge_index.shape[0]):
            if edge_id in shell_id_set:
                a, b = edge_index[edge_id]
                result_curves.append(self.sample_boundary_box_edge_uv(nodes_uv[a], nodes_uv[b], n_samples=n_samples))
            else:
                result_curves.append(curves[edge_id])
        return torch.stack(result_curves, dim=0)

    def sample_smooth_edge_curves_xyz(self, cad_domain: Any, curves_uv: torch.Tensor) -> torch.Tensor:
        """Lift UV curves through a differentiable Torch UV-to-XYZ evaluator."""
        evaluator = getattr(cad_domain, 'eval_uv_norm_batch_torch', None)
        if evaluator is None or not callable(evaluator):
            raise TypeError('cad_domain must provide differentiable eval_uv_norm_batch_torch(flat_uv) to sample edge curves in XYZ.')
        if curves_uv.ndim != 3 or curves_uv.shape[-1] != 2:
            raise ValueError('curves_uv must have shape [E, n_samples, 2].')
        flat_uv = curves_uv.reshape(-1, 2)
        evaluated = evaluator(flat_uv)
        xyz = evaluated['xyz'] if isinstance(evaluated, dict) else evaluated
        if not isinstance(xyz, torch.Tensor):
            raise TypeError("eval_uv_norm_batch_torch must return a torch.Tensor or {'xyz': tensor}.")
        if xyz.ndim != 2 or xyz.shape != (flat_uv.shape[0], 3):
            raise ValueError('Torch CAD evaluator must return XYZ with shape [E*n_samples, 3].')
        return xyz.reshape(curves_uv.shape[0], curves_uv.shape[1], 3)

    def adaptive_sample_count_from_curves(
        self,
        curves: torch.Tensor,
        min_samples: int,
        target_spacing: torch.Tensor | float,
        max_samples: int=4096,
        max_total_samples: int=65536,
        turn_angle_step: float=np.pi / 24.0,
    ) -> int:
        """Choose one dense sample count for a batch of polylines."""
        min_samples = max(int(min_samples), 2)
        max_samples = max(int(max_samples), min_samples)
        if curves.ndim != 3 or curves.shape[0] == 0 or curves.shape[1] < 2:
            return min_samples
        max_total_samples = max(int(max_total_samples), min_samples)
        max_samples = min(max_samples, max(min_samples, max_total_samples // max(int(curves.shape[0]), 1)))
        deltas = curves[:, 1:] - curves[:, :-1]
        segment_lengths = torch.linalg.vector_norm(deltas, dim=-1)
        max_arc_length = segment_lengths.sum(dim=1).amax()
        spacing_t = torch.as_tensor(target_spacing, dtype=curves.dtype, device=curves.device)
        spacing_t = spacing_t.clamp_min(self.eps)
        length_samples = torch.ceil(max_arc_length / spacing_t).to(torch.long) + 1
        if deltas.shape[1] > 1:
            unit = deltas / segment_lengths.unsqueeze(-1).clamp_min(self.eps)
            cos_turn = (unit[:, 1:] * unit[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
            max_total_turn = torch.acos(cos_turn).sum(dim=1).amax()
            turn_samples = torch.ceil(max_total_turn / max(float(turn_angle_step), self.eps)).to(torch.long) + 2
            needed = torch.maximum(length_samples, turn_samples)
        else:
            needed = length_samples
        needed_int = int(needed.detach().cpu().item())
        return min(max(needed_int, min_samples), max_samples)

    def adaptive_graph_curve_sample_count_uv(
        self,
        seeds_uv: torch.Tensor,
        graph: dict[str, torch.Tensor],
        min_samples: int,
        target_spacing: float | torch.Tensor,
        max_total_samples: int=65536,
        u_periodic: bool=False,
        v_periodic: bool=False,
    ) -> int:
        """Estimate an edge-curve count from a coarse UV pass."""
        min_samples = max(int(min_samples), 2)
        edge_index = graph.get('edge_index')
        if edge_index is None or edge_index.numel() == 0:
            return min_samples
        coarse = self.sample_graph_edge_curves_uv(
            seeds_uv=seeds_uv,
            graph=graph,
            n_samples=min_samples,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        return self.adaptive_sample_count_from_curves(
            coarse,
            min_samples=min_samples,
            target_spacing=target_spacing,
            max_total_samples=max_total_samples,
        )

    def soft_lift_uv_to_xyz(self, query_uv: torch.Tensor, support_uv: torch.Tensor, support_xyz: torch.Tensor, tau: float | None=None, u_periodic: bool=False, v_periodic: bool=False) -> torch.Tensor:
        """Differentiably lift UV samples to XYZ by soft interpolation from face samples."""
        if query_uv.ndim < 2 or query_uv.shape[-1] != 2:
            raise ValueError('query_uv must have shape [..., 2].')
        if support_uv.ndim != 2 or support_uv.shape[-1] != 2:
            raise ValueError('support_uv must have shape [N, 2].')
        if support_xyz.ndim != 2 or support_xyz.shape[-1] != 3:
            raise ValueError('support_xyz must have shape [N, 3].')
        if support_uv.shape[0] != support_xyz.shape[0]:
            raise ValueError('support_uv and support_xyz must contain the same number of points.')
        original_shape = query_uv.shape[:-1]
        flat_uv = query_uv.reshape(-1, 2)
        diff = flat_uv[:, None, :] - support_uv[None, :, :]
        if u_periodic:
            diff_u = diff[..., 0] - torch.round(diff[..., 0])
            diff = torch.cat((diff_u.unsqueeze(-1), diff[..., 1:2]), dim=-1)
        if v_periodic:
            diff_v = diff[..., 1] - torch.round(diff[..., 1])
            diff = torch.cat((diff[..., 0:1], diff_v.unsqueeze(-1)), dim=-1)
        dist = torch.linalg.vector_norm(diff, dim=-1)
        tau_t = flat_uv.new_tensor(self.tube_lift_tau if tau is None else float(tau)).clamp_min(self.eps)
        weights = torch.softmax(-dist / tau_t, dim=1)
        xyz = weights @ support_xyz
        return xyz.reshape(*original_shape, 3)

    def width(self, w_raw: torch.Tensor, seeds: torch.Tensor | None=None, **_: Any) -> torch.Tensor:
        """Map raw width to a non-negative UV radius.

        A raw value of zero means core curves only. Positive raw values grow
        thickness through a temperature-smoothed positive transform. Minimum
        printable feature constraints can be applied by training code later.
        """
        if w_raw.ndim != 2 or w_raw.shape[0] != w_raw.shape[1]:
            raise ValueError(f'w_raw must be square [S,S], got {tuple(w_raw.shape)}.')
        if seeds is not None and (seeds.ndim != 2 or seeds.shape[-1] != 2):
            raise ValueError('seeds must have shape [S, 2].')
        if w_raw.shape[0] > 1:
            pair_mask = torch.triu(torch.ones_like(w_raw, dtype=torch.bool), diagonal=1)
            width_raw_global = w_raw[pair_mask].mean()
        else:
            width_raw_global = w_raw.mean()
        temp = w_raw.new_tensor(max(float(self.raw_temp), self.eps))
        zero = width_raw_global.new_tensor(0.0)
        soft_zero = width_raw_global.new_tensor(np.log(2.0))
        positive_width = temp * (F.softplus(width_raw_global / temp) - soft_zero)
        w_geo = torch.where(width_raw_global > 0.0, positive_width, zero)
        return w_geo.expand_as(w_raw)

    def _local_uv_to_xyz_scale(self, Xu: torch.Tensor | None, Xv: torch.Tensor | None, ref_xyz: torch.Tensor) -> torch.Tensor:
        if Xu is None or Xv is None:
            return ref_xyz.new_tensor(1.0)
        Xu_norm = torch.linalg.vector_norm(Xu.to(dtype=ref_xyz.dtype, device=ref_xyz.device), dim=-1)
        Xv_norm = torch.linalg.vector_norm(Xv.to(dtype=ref_xyz.dtype, device=ref_xyz.device), dim=-1)
        scale = torch.minimum(Xu_norm, Xv_norm)
        scale = scale[torch.isfinite(scale) & (scale > self.eps)]
        if scale.numel() == 0:
            return ref_xyz.new_tensor(1.0)
        return scale.median()

    def _pair_upper_mean(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2 and value.shape[0] == value.shape[1] and value.shape[0] > 1:
            mask = torch.triu(torch.ones_like(value, dtype=torch.bool), diagonal=1)
            return value[mask].mean()
        return value.reshape(-1).mean()

    def build_swept_tube_fields(
        self,
        points_uv: torch.Tensor,
        points_3d: torch.Tensor,
        seeds_uv: torch.Tensor,
        w_raw: torch.Tensor,
        Xu: torch.Tensor | None=None,
        Xv: torch.Tensor | None=None,
        cad_domain: Any | None=None,
        u_periodic: bool=False,
        v_periodic: bool=False,
        return_xyz: bool=True,
    ) -> dict[str, Any]:
        topo_out = self.forward_scipy_topology(
            seeds_uv=seeds_uv,
            cad_domain=cad_domain,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
            return_xyz=return_xyz,
            keep_isolated_vertices=False,
        )
        w_geo = self.width(w_raw, seeds=seeds_uv)
        width_uv = self._pair_upper_mean(w_geo)
        local_scale = self._local_uv_to_xyz_scale(Xu, Xv, points_3d)
        radius_3d = (width_uv * local_scale).clamp_min(0.0)
        tau_distance = max(float(self.tube_distance_tau), self.eps) * local_scale.clamp_min(self.eps)
        tau_density = max(float(self.tube_density_tau), self.eps) * local_scale.clamp_min(self.eps)
        tau_fiber = max(float(self.tube_fiber_tau), self.eps) * local_scale.clamp_min(self.eps)
        min_tube_samples = max(int(self.tube_curve_samples), 2)
        curves_uv = topo_out.get('edge_curves_uv')
        if curves_uv is None:
            curves_uv = points_uv.new_empty((0, min_tube_samples, 2))
        topology_seeds_uv = topo_out.get('topology_seeds_uv', seeds_uv)
        use_torch_cad = cad_domain is not None and callable(getattr(cad_domain, 'eval_uv_norm_batch_torch', None))
        if curves_uv.shape[0] > 0:
            if curves_uv.shape[1] != min_tube_samples:
                curves_uv = self.sample_graph_edge_curves_uv(
                    seeds_uv=topology_seeds_uv,
                    graph=topo_out['graph'],
                    n_samples=min_tube_samples,
                    u_periodic=u_periodic,
                    v_periodic=v_periodic,
                )
            if use_torch_cad:
                coarse_curves_xyz = self.sample_smooth_edge_curves_xyz(cad_domain, curves_uv)
            else:
                coarse_curves_xyz = self.soft_lift_uv_to_xyz(
                    curves_uv,
                    points_uv,
                    points_3d,
                    u_periodic=u_periodic,
                    v_periodic=v_periodic,
                )
            target_spacing = torch.maximum(
                float(self.tube_target_spacing_ratio) * radius_3d,
                points_3d.new_tensor(max(float(self.min_tube_spacing), self.eps)),
            )
            tube_samples = self.adaptive_sample_count_from_curves(
                coarse_curves_xyz,
                min_samples=min_tube_samples,
                target_spacing=target_spacing,
            )
            if tube_samples != curves_uv.shape[1]:
                curves_uv = self.sample_graph_edge_curves_uv(
                    seeds_uv=topology_seeds_uv,
                    graph=topo_out['graph'],
                    n_samples=tube_samples,
                    u_periodic=u_periodic,
                    v_periodic=v_periodic,
                )
                curves_xyz = (
                    self.sample_smooth_edge_curves_xyz(cad_domain, curves_uv)
                    if use_torch_cad else
                    self.soft_lift_uv_to_xyz(curves_uv, points_uv, points_3d, u_periodic=u_periodic, v_periodic=v_periodic)
                )
            else:
                curves_xyz = coarse_curves_xyz
        else:
            curves_xyz = points_3d.new_empty((0, min_tube_samples, 3))
        if use_torch_cad:
            seeds_xyz = self.sample_smooth_edge_curves_xyz(cad_domain, seeds_uv.reshape(-1, 1, 2)).reshape(-1, 3)
        else:
            seeds_xyz = self.soft_lift_uv_to_xyz(
                seeds_uv,
                points_uv,
                points_3d,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
        if curves_xyz.shape[0] == 0:
            rho = points_3d.new_zeros((points_3d.shape[0],))
            fallback = points_3d.new_tensor([1.0, 0.0, 0.0]).expand(points_3d.shape[0], 3)
            if Xu is not None:
                fallback = F.normalize(Xu.to(dtype=points_3d.dtype, device=points_3d.device), dim=-1, eps=self.eps)
            field = {'density': rho, 'fiber': fallback, 'distance': points_3d.new_full((points_3d.shape[0],), float('inf'))}
        else:
            field = self.soft_tube_density_and_fiber_to_elements(
                elem_centers_xyz=points_3d,
                curves_xyz=curves_xyz,
                radius=radius_3d,
                tau_distance=float(tau_distance.detach().item()),
                tau_density=float(tau_density.detach().item()),
                tau_fiber=float(tau_fiber.detach().item()),
                rho_min=float(self.rho_min),
                fallback_fiber=Xu,
            )
        out = dict(topo_out)
        out.update({
            'seeds': seeds_uv,
            'seeds_uv': seeds_uv,
            'seeds_xyz': seeds_xyz,
            'rho': field['density'],
            'density': field['density'],
            'fiber3d': field['fiber'],
            'tube_distance': field['distance'],
            'w_geo': w_geo,
            'centerline_radius': radius_3d,
            'edge_curves_uv': curves_uv,
            'edge_curves_xyz': curves_xyz,
        })
        return out

    def softmin_distance_to_curves(self, query_xyz: torch.Tensor, curves_xyz: torch.Tensor, tau: float=0.01) -> torch.Tensor:
        """Return each query point's soft-min distance to all curve samples."""
        if query_xyz.ndim != 2 or query_xyz.shape[-1] != 3:
            raise ValueError('query_xyz must have shape [M, 3].')
        if curves_xyz.ndim != 3 or curves_xyz.shape[-1] != 3:
            raise ValueError('curves_xyz must have shape [E, K, 3].')
        if query_xyz.device != curves_xyz.device:
            raise ValueError('query_xyz and curves_xyz must share a device.')
        if query_xyz.dtype != curves_xyz.dtype:
            raise ValueError('query_xyz and curves_xyz must share a dtype.')
        if not query_xyz.is_floating_point() or not curves_xyz.is_floating_point():
            raise TypeError('query_xyz and curves_xyz must be floating point tensors.')
        if tau <= 0.0:
            raise ValueError('tau must be greater than zero.')
        curve_points = curves_xyz.reshape(-1, 3)
        if curve_points.shape[0] == 0:
            raise ValueError('curves_xyz must contain at least one curve sample.')
        distances = torch.cdist(query_xyz, curve_points)
        tau_t = query_xyz.new_tensor(float(tau))
        sample_count = max(int(curve_points.shape[0]), 1)
        return -tau_t * (torch.logsumexp(-distances / tau_t, dim=1) - np.log(sample_count))

    def soft_tube_occupancy(self, query_xyz: torch.Tensor, curves_xyz: torch.Tensor, radius: torch.Tensor | float, tau_distance: float=0.01, tau_occupancy: float=0.01) -> dict[str, torch.Tensor]:
        """Build a differentiable swept-sphere occupancy field around curves.

                ``radius`` is the physical positive radius. For a learnable
                unconstrained parameter, apply ``F.softplus`` before calling this
                method (as done with :meth:`make_learnable_radius`).
                """
        if tau_occupancy <= 0.0:
            raise ValueError('tau_occupancy must be greater than zero.')
        radius_tensor = torch.as_tensor(radius, dtype=query_xyz.dtype, device=query_xyz.device).clamp_min(self.eps)
        curve_points = curves_xyz.reshape(-1, 3)
        d_soft = torch.cdist(query_xyz, curve_points).min(dim=1).values
        tau_occupancy_t = query_xyz.new_tensor(float(tau_occupancy))
        occupancy = torch.sigmoid((radius_tensor - d_soft) / tau_occupancy_t)
        return {'distance': d_soft, 'occupancy': occupancy, 'radius': radius_tensor}

    def make_learnable_radius(self, initial_radius: float=0.02) -> nn.Parameter:
        """Return a parameter whose softplus is ``initial_radius``."""
        if initial_radius <= 0.0:
            raise ValueError('initial_radius must be greater than zero.')
        initial = torch.tensor(float(initial_radius), dtype=torch.get_default_dtype())
        unconstrained = torch.log(torch.expm1(initial))
        return nn.Parameter(unconstrained)

    def curve_points_and_tangents_xyz(self, curves_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten curve samples and their normalized finite-difference tangents."""
        if curves_xyz.ndim != 3 or curves_xyz.shape[-1] != 3:
            raise ValueError('curves_xyz must have shape [E, K, 3].')
        if not curves_xyz.is_floating_point():
            raise TypeError('curves_xyz must be a floating point tensor.')
        if curves_xyz.shape[0] == 0:
            raise ValueError('curves_xyz must contain at least one edge.')
        if curves_xyz.shape[1] < 2:
            raise ValueError('Each curve must contain at least two samples.')
        first = curves_xyz[:, 1] - curves_xyz[:, 0]
        middle = curves_xyz[:, 2:] - curves_xyz[:, :-2]
        last = curves_xyz[:, -1] - curves_xyz[:, -2]
        tangents = torch.cat((first[:, None, :], middle, last[:, None, :]), dim=1)
        tangents = tangents / torch.linalg.vector_norm(tangents, dim=-1, keepdim=True).clamp_min(self.eps)
        return (curves_xyz.reshape(-1, 3), tangents.reshape(-1, 3))

    def curve_segments_and_tangents_xyz(self, curves_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return segment endpoints, normalized tangents, and segment AABBs."""
        if curves_xyz.ndim != 3 or curves_xyz.shape[-1] != 3:
            raise ValueError('curves_xyz must have shape [E, K, 3].')
        if not curves_xyz.is_floating_point():
            raise TypeError('curves_xyz must be a floating point tensor.')
        if curves_xyz.shape[0] == 0 or curves_xyz.shape[1] < 2:
            empty = curves_xyz.new_empty((0, 3))
            return (empty, empty, empty, empty, empty)
        seg_a = curves_xyz[:, :-1, :].reshape(-1, 3)
        seg_b = curves_xyz[:, 1:, :].reshape(-1, 3)
        delta = seg_b - seg_a
        length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        fallback = seg_a.new_tensor([1.0, 0.0, 0.0]).expand_as(delta)
        tangents = torch.where(length > self.eps, delta / length.clamp_min(self.eps), fallback)
        aabb_min = torch.minimum(seg_a, seg_b)
        aabb_max = torch.maximum(seg_a, seg_b)
        return (seg_a, seg_b, tangents, aabb_min, aabb_max)

    def point_to_segments_distance(self, query_xyz: torch.Tensor, seg_a: torch.Tensor, seg_b: torch.Tensor) -> torch.Tensor:
        """Return point-to-segment distances for every query/segment pair."""
        if query_xyz.ndim != 2 or query_xyz.shape[-1] != 3:
            raise ValueError('query_xyz must have shape [M, 3].')
        if seg_a.ndim != 2 or seg_a.shape[-1] != 3 or seg_b.shape != seg_a.shape:
            raise ValueError('seg_a and seg_b must have shape [G, 3].')
        if query_xyz.device != seg_a.device or seg_b.device != seg_a.device:
            raise ValueError('query_xyz, seg_a, and seg_b must share a device.')
        if query_xyz.dtype != seg_a.dtype or seg_b.dtype != seg_a.dtype:
            raise ValueError('query_xyz, seg_a, and seg_b must share a dtype.')
        ab = seg_b - seg_a
        aq = query_xyz[:, None, :] - seg_a[None, :, :]
        denom = (ab * ab).sum(dim=-1).clamp_min(self.eps)
        t = (aq * ab[None, :, :]).sum(dim=-1) / denom[None, :]
        t = t.clamp(0.0, 1.0)
        closest = seg_a[None, :, :] + t[..., None] * ab[None, :, :]
        return torch.linalg.vector_norm(query_xyz[:, None, :] - closest, dim=-1)

    def _safe_normalize_fiber(self, fiber: torch.Tensor) -> torch.Tensor:
        default = fiber.new_tensor([1.0, 0.0, 0.0]).expand_as(fiber)
        norm = torch.linalg.vector_norm(fiber, dim=-1, keepdim=True)
        return torch.where(norm > self.eps, fiber / norm.clamp_min(self.eps), default)

    def _fallback_fiber_field(self, elem_centers_xyz: torch.Tensor, fallback_fiber: torch.Tensor | None=None) -> torch.Tensor:
        if fallback_fiber is None:
            fiber = elem_centers_xyz.new_tensor([1.0, 0.0, 0.0]).expand(elem_centers_xyz.shape[0], 3)
        else:
            fiber = torch.as_tensor(fallback_fiber, dtype=elem_centers_xyz.dtype, device=elem_centers_xyz.device)
            if fiber.ndim == 1:
                if fiber.shape[0] != 3:
                    raise ValueError('fallback_fiber must have shape [3] or [numElems, 3].')
                fiber = fiber.expand(elem_centers_xyz.shape[0], 3)
            elif fiber.shape != elem_centers_xyz.shape:
                raise ValueError('fallback_fiber must have shape [3] or [numElems, 3].')
        return self._safe_normalize_fiber(fiber)

    def soft_tube_density_and_fiber_to_elements_sampled(self, elem_centers_xyz: torch.Tensor, curves_xyz: torch.Tensor, radius: torch.Tensor | float, tau_distance: float=0.01, tau_density: float=0.01, tau_fiber: float=0.01, rho_min: float=0.001, fallback_fiber: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        """Legacy sampled-point tube field used when segment distances are disabled."""
        if curves_xyz.shape[0] == 0 or curves_xyz.shape[1] == 0:
            fiber = self._fallback_fiber_field(elem_centers_xyz, fallback_fiber)
            density = elem_centers_xyz.new_full((elem_centers_xyz.shape[0],), float(rho_min))
            distance = elem_centers_xyz.new_full((elem_centers_xyz.shape[0],), float('inf'))
            ax, ay, az = fiber.unbind(dim=1)
            phi = torch.atan2(ay, ax)
            theta = torch.acos(az.clamp(-1.0 + 1e-06, 1.0 - 1e-06))
            return {'density': density, 'fiber': fiber, 'phi': phi, 'theta': theta, 'distance': distance}
        curve_points, curve_tangents = self.curve_points_and_tangents_xyz(curves_xyz)
        radius_tensor = torch.as_tensor(radius, dtype=elem_centers_xyz.dtype, device=elem_centers_xyz.device).clamp_min(self.eps)
        tau_density_t = elem_centers_xyz.new_tensor(float(tau_density))
        tau_fiber_t = elem_centers_xyz.new_tensor(float(tau_fiber))
        max_cdist_values = 16000000
        chunk_size = max(1, min(int(elem_centers_xyz.shape[0]), max_cdist_values // max(int(curve_points.shape[0]), 1)))
        distance_chunks = []
        fiber_chunks = []
        nearest_k = min(int(self.nearest_segment_k), int(curve_points.shape[0]))
        for start in range(0, int(elem_centers_xyz.shape[0]), chunk_size):
            end = min(start + chunk_size, int(elem_centers_xyz.shape[0]))
            distances = torch.cdist(elem_centers_xyz[start:end], curve_points)
            nearest_distances, nearest_ids = torch.topk(distances, k=nearest_k, dim=1, largest=False)
            distance_chunks.append(nearest_distances[:, 0])
            fiber_weights = torch.softmax(-nearest_distances / tau_fiber_t, dim=1)
            nearest_tangents = curve_tangents[nearest_ids]
            fiber_chunks.append((fiber_weights.unsqueeze(-1) * nearest_tangents).sum(dim=1))
        d_soft = torch.cat(distance_chunks, dim=0)
        occupancy = torch.sigmoid((radius_tensor - d_soft) / tau_density_t)
        density = float(rho_min) + (1.0 - float(rho_min)) * occupancy
        fiber = torch.cat(fiber_chunks, dim=0)
        fiber = self._safe_normalize_fiber(fiber)
        ax, ay, az = fiber.unbind(dim=1)
        phi = torch.atan2(ay, ax)
        theta = torch.acos(az.clamp(-1.0 + 1e-06, 1.0 - 1e-06))
        return {'density': density, 'fiber': fiber, 'phi': phi, 'theta': theta, 'distance': d_soft}

    def soft_tube_density_and_fiber_to_elements(self, elem_centers_xyz: torch.Tensor, curves_xyz: torch.Tensor, radius: torch.Tensor | float, tau_distance: float=0.01, tau_density: float=0.01, tau_fiber: float=0.01, rho_min: float=0.001, fallback_fiber: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        """Map swept graph tubes to structured-grid density and fiber fields."""
        if elem_centers_xyz.ndim != 2 or elem_centers_xyz.shape[-1] != 3:
            raise ValueError('elem_centers_xyz must have shape [numElems, 3].')
        if not elem_centers_xyz.is_floating_point():
            raise TypeError('elem_centers_xyz must be a floating point tensor.')
        if elem_centers_xyz.device != curves_xyz.device:
            raise ValueError('elem_centers_xyz and curves_xyz must share a device.')
        if elem_centers_xyz.dtype != curves_xyz.dtype:
            raise ValueError('elem_centers_xyz and curves_xyz must share a dtype.')
        if tau_distance <= 0.0 or tau_density <= 0.0 or tau_fiber <= 0.0:
            raise ValueError('All distance, density, and fiber temperatures must be positive.')
        if not 0.0 <= rho_min < 1.0:
            raise ValueError('rho_min must satisfy 0 <= rho_min < 1.')
        if not self.use_segment_distance:
            return self.soft_tube_density_and_fiber_to_elements_sampled(
                elem_centers_xyz=elem_centers_xyz,
                curves_xyz=curves_xyz,
                radius=radius,
                tau_distance=tau_distance,
                tau_density=tau_density,
                tau_fiber=tau_fiber,
                rho_min=rho_min,
                fallback_fiber=fallback_fiber,
            )
        seg_a, seg_b, seg_tangents, aabb_min, aabb_max = self.curve_segments_and_tangents_xyz(curves_xyz)
        fallback = self._fallback_fiber_field(elem_centers_xyz, fallback_fiber)
        if seg_a.shape[0] == 0:
            density = elem_centers_xyz.new_full((elem_centers_xyz.shape[0],), float(rho_min))
            distance = elem_centers_xyz.new_full((elem_centers_xyz.shape[0],), float('inf'))
            ax, ay, az = fallback.unbind(dim=1)
            phi = torch.atan2(ay, ax)
            theta = torch.acos(az.clamp(-1.0 + 1e-06, 1.0 - 1e-06))
            return {'density': density, 'fiber': fallback, 'phi': phi, 'theta': theta, 'distance': distance}
        radius_tensor = torch.as_tensor(radius, dtype=elem_centers_xyz.dtype, device=elem_centers_xyz.device).clamp_min(self.eps)
        tau_density_t = elem_centers_xyz.new_tensor(float(tau_density))
        tau_fiber_t = elem_centers_xyz.new_tensor(float(tau_fiber))
        active_band = radius_tensor + 3.0 * tau_density_t
        max_cdist_values = 16000000
        chunk_size = max(1, min(int(elem_centers_xyz.shape[0]), max_cdist_values // max(int(seg_a.shape[0]), 1)))
        distance_chunks = []
        fiber_chunks = []
        nearest_k = min(int(self.nearest_segment_k), int(seg_a.shape[0]))
        for start in range(0, int(elem_centers_xyz.shape[0]), chunk_size):
            end = min(start + chunk_size, int(elem_centers_xyz.shape[0]))
            query = elem_centers_xyz[start:end]
            active = torch.ones((query.shape[0],), dtype=torch.bool, device=query.device)
            if self.use_spatial_pruning:
                below = (aabb_min[None, :, :] - active_band) - query[:, None, :]
                above = query[:, None, :] - (aabb_max[None, :, :] + active_band)
                outside_delta = torch.clamp(torch.maximum(below, above), min=0.0)
                aabb_dist = torch.linalg.vector_norm(outside_delta, dim=-1)
                active = (aabb_dist <= self.eps).any(dim=1)
            chunk_distance = elem_centers_xyz.new_full((query.shape[0],), float('inf'))
            chunk_fiber = fallback[start:end].clone()
            if bool(active.any()):
                active_query = query[active]
                distances = self.point_to_segments_distance(active_query, seg_a, seg_b)
                nearest_distances, nearest_ids = torch.topk(distances, k=nearest_k, dim=1, largest=False)
                chunk_distance[active] = nearest_distances[:, 0]
                fiber_weights = torch.softmax(-nearest_distances / tau_fiber_t, dim=1)
                nearest_tangents = seg_tangents[nearest_ids]
                chunk_fiber[active] = (fiber_weights.unsqueeze(-1) * nearest_tangents).sum(dim=1)
            distance_chunks.append(chunk_distance)
            fiber_chunks.append(chunk_fiber)
        d_soft = torch.cat(distance_chunks, dim=0)
        occupancy = torch.where(
            torch.isfinite(d_soft),
            torch.sigmoid((radius_tensor - d_soft) / tau_density_t),
            torch.zeros_like(d_soft),
        )
        density = float(rho_min) + (1.0 - float(rho_min)) * occupancy
        fiber = torch.cat(fiber_chunks, dim=0)
        fiber = self._safe_normalize_fiber(fiber)
        ax, ay, az = fiber.unbind(dim=1)
        phi = torch.atan2(ay, ax)
        theta = torch.acos(az.clamp(-1.0 + 1e-06, 1.0 - 1e-06))
        return {'density': density, 'fiber': fiber, 'phi': phi, 'theta': theta, 'distance': d_soft}

    def unwrap_triple_seeds(self, seeds_uv: torch.Tensor, triples: torch.Tensor, u_periodic: bool, v_periodic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qi = seeds_uv[triples[:, 0]]
        sj = seeds_uv[triples[:, 1]]
        sk = seeds_uv[triples[:, 2]]
        qj = qi + self.periodic_difference(sj, qi, u_periodic, v_periodic)
        qk = qi + self.periodic_difference(sk, qi, u_periodic, v_periodic)
        return (qi, qj, qk)

    def wrap_uv(self, P: torch.Tensor, u_periodic: bool, v_periodic: bool) -> torch.Tensor:
        if not (u_periodic or v_periodic):
            return P
        u = P[..., 0:1]
        v = P[..., 1:2]
        if u_periodic:
            u = torch.remainder(u, 1.0)
        if v_periodic:
            v = torch.remainder(v, 1.0)
        return torch.cat((u, v), dim=-1)

    def circumcenters_from_triples(self, seeds_uv: torch.Tensor, triples: torch.Tensor, u_periodic: bool, v_periodic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        qi, qj, qk = self.unwrap_triple_seeds(seeds_uv, triples, u_periodic, v_periodic)
        row_j = 2.0 * (qj - qi)
        row_k = 2.0 * (qk - qi)
        A = torch.stack((row_j, row_k), dim=-2)
        qi2 = (qi * qi).sum(dim=-1)
        b = torch.stack(((qj * qj).sum(dim=-1) - qi2, (qk * qk).sum(dim=-1) - qi2), dim=-1)
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
        pair_dists = {'dij': self.periodic_distance(qi, qj, False, False), 'dik': self.periodic_distance(qi, qk, False, False), 'djk': self.periodic_distance(qj, qk, False, False)}
        return (P_unwrapped, P_uv, area2, pair_dists)

    def _cross2(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

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
        if hasattr(cad_domain, 'sample_trim_sdf'):
            sdf = cad_domain.sample_trim_sdf(P_uv)
            sdf = torch.as_tensor(sdf, dtype=P_uv.dtype, device=P_uv.device)
            return torch.sigmoid((sdf + self.vertex_boundary_margin) / self._tau_tensor(self.tau_trim, P_uv))
        g_trim = cad_domain.smooth_inside_activity(P_uv, tau=self.tau_trim)
        return torch.as_tensor(g_trim, dtype=P_uv.dtype, device=P_uv.device)

    def edge_trim_gate(self, edge_curves_uv: torch.Tensor, cad_domain: Any | None, tau_trim: float | None=None, margin: float | None=None, reduction: str='softmin') -> torch.Tensor:
        """
                Return per-edge trim validity in [0, 1].

                edge_curves_uv: [E, K, 2]
                """
        if edge_curves_uv.ndim != 3 or edge_curves_uv.shape[-1] != 2:
            raise ValueError('edge_curves_uv must have shape [E, K, 2].')
        if not edge_curves_uv.is_floating_point():
            raise TypeError('edge_curves_uv must be a floating point tensor.')
        num_edges = edge_curves_uv.shape[0]
        if num_edges == 0:
            return edge_curves_uv.new_empty((0,))
        if cad_domain is None or not self.use_trim_activity or (not self.use_edge_trim_gate):
            return torch.ones((num_edges,), dtype=edge_curves_uv.dtype, device=edge_curves_uv.device)
        if not hasattr(cad_domain, 'sample_trim_sdf'):
            return torch.ones((num_edges,), dtype=edge_curves_uv.dtype, device=edge_curves_uv.device)
        tau_trim_t = self._tau_tensor(self.tau_trim if tau_trim is None else float(tau_trim), edge_curves_uv)
        margin_t = torch.as_tensor(self.vertex_boundary_margin if margin is None else float(margin), dtype=edge_curves_uv.dtype, device=edge_curves_uv.device)
        flat_uv = edge_curves_uv.reshape(-1, 2)
        sdf = cad_domain.sample_trim_sdf(flat_uv)
        sdf = torch.as_tensor(sdf, dtype=edge_curves_uv.dtype, device=edge_curves_uv.device).reshape(edge_curves_uv.shape[0], edge_curves_uv.shape[1])
        activity = torch.sigmoid((sdf + margin_t) / tau_trim_t)
        reduction = str(reduction)
        if reduction == 'min':
            edge_gate = activity.amin(dim=1)
        elif reduction == 'mean':
            edge_gate = activity.mean(dim=1)
        elif reduction == 'softmin':
            tau_reduce = torch.as_tensor(max(float(self.edge_trim_reduce_tau), self.eps), dtype=edge_curves_uv.dtype, device=edge_curves_uv.device)
            edge_gate = -tau_reduce * torch.logsumexp(-activity / tau_reduce, dim=1) + tau_reduce * np.log(activity.shape[1])
        else:
            raise ValueError(f"edge trim reduction must be 'softmin', 'min', or 'mean', got {reduction!r}.")
        return torch.nan_to_num(edge_gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def _safe_div(self, numerator, denominator, eps):
        sign = torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
        denom = torch.where(denominator.abs() < eps, sign * eps, denominator)
        return numerator / denom

    def ray_box_intersection_uv(self, origin: torch.Tensor, direction: torch.Tensor, u_periodic: bool=False, v_periodic: bool=False) -> torch.Tensor:
        """
                Intersect ray origin + t direction, t > 0, with normalized UV box [0,1]^2.
                Returns boundary point [2].
                """
        hit, _, valid = self.ray_box_hit_torch(origin, direction, u_periodic=u_periodic, v_periodic=v_periodic)
        return torch.where(valid, hit, origin)

    @staticmethod
    def point_inside_box_np(p: np.ndarray, tol: float=1e-09) -> bool:
        """Hard topology test for the normalized UV box."""
        return bool(-tol <= float(p[0]) <= 1.0 + tol and -tol <= float(p[1]) <= 1.0 + tol)

    @staticmethod
    def segment_box_clip_np(p0: np.ndarray, p1: np.ndarray, bounds: tuple[float, float, float, float]=(0.0, 1.0, 0.0, 1.0), tol: float=1e-12) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """Liang--Barsky clip of a 2-D segment, including entry/exit parameters."""
        xmin, xmax, ymin, ymax = bounds
        delta = p1 - p0
        t_enter, t_exit = (0.0, 1.0)
        for p, q in ((-delta[0], p0[0] - xmin), (delta[0], xmax - p0[0]), (-delta[1], p0[1] - ymin), (delta[1], ymax - p0[1])):
            if abs(float(p)) <= tol:
                if float(q) < -tol:
                    return None
                continue
            ratio = float(q / p)
            if p < 0.0:
                t_enter = max(t_enter, ratio)
            else:
                t_exit = min(t_exit, ratio)
            if t_enter > t_exit + tol:
                return None
        q0 = np.clip(p0 + t_enter * delta, (xmin, ymin), (xmax, ymax))
        q1 = np.clip(p0 + t_exit * delta, (xmin, ymin), (xmax, ymax))
        return (q0, q1, t_enter, t_exit)

    def ray_box_hit_torch(self, origin: torch.Tensor, direction: torch.Tensor, u_periodic: bool=False, v_periodic: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the nearest positive box hit as ``(point, t, valid)``."""
        candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        eps = torch.as_tensor(self.eps, dtype=origin.dtype, device=origin.device)
        ox, oy = origin.unbind()
        dx, dy = direction.unbind()
        if not u_periodic:
            for value in (0.0, 1.0):
                u = torch.as_tensor(value, dtype=origin.dtype, device=origin.device)
                t = self._safe_div(u - ox, dx, eps)
                y = oy + t * dy
                valid = (t > eps) & (y >= -eps) & (y <= 1.0 + eps)
                candidates.append((t, torch.stack((u, y)), valid))
        if not v_periodic:
            for value in (0.0, 1.0):
                v = torch.as_tensor(value, dtype=origin.dtype, device=origin.device)
                t = self._safe_div(v - oy, dy, eps)
                x = ox + t * dx
                valid = (t > eps) & (x >= -eps) & (x <= 1.0 + eps)
                candidates.append((t, torch.stack((x, v)), valid))
        if not candidates:
            return (origin, torch.zeros_like(ox), torch.zeros((), dtype=torch.bool, device=origin.device))
        big = torch.as_tensor(float('inf'), dtype=origin.dtype, device=origin.device)
        ts = torch.stack([torch.where(valid, t, big) for t, _, valid in candidates])
        points = torch.stack([point for _, point, _ in candidates])
        index = torch.argmin(ts)
        valid = torch.isfinite(ts[index])
        point = points[index]
        hit_u = torch.remainder(point[0], 1.0) if u_periodic else point[0].clamp(0.0, 1.0)
        hit_v = torch.remainder(point[1], 1.0) if v_periodic else point[1].clamp(0.0, 1.0)
        hit_candidate = torch.stack((hit_u, hit_v))
        hit = torch.where(valid, hit_candidate, origin)
        return (hit, ts[index], valid)

    def snap_near_box_boundary_uv(
        self,
        p: torch.Tensor,
        tol: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        If p is within tol of the UV box boundary [0,1]^2, snap it to the nearest
        boundary side.

        Returns:
            snapped: [2] tensor
            did_snap: bool tensor scalar
        """
        if p.shape != (2,):
            raise ValueError('p must have shape [2].')
        tol_t = torch.as_tensor(
            self.boundary_snap_tol if tol is None else float(tol),
            dtype=p.dtype,
            device=p.device,
        )
        zero = p.new_tensor(0.0)
        one = p.new_tensor(1.0)

        dists = torch.stack([
            torch.abs(p[0] - zero),
            torch.abs(p[0] - one),
            torch.abs(p[1] - zero),
            torch.abs(p[1] - one),
        ])
        side = torch.argmin(dists)
        did_snap = dists[side] <= tol_t

        p_clamped = p.clamp(0.0, 1.0)
        left = torch.stack((zero, p_clamped[1]))
        right = torch.stack((one, p_clamped[1]))
        bottom = torch.stack((p_clamped[0], zero))
        top = torch.stack((p_clamped[0], one))
        candidates = torch.stack((left, right, bottom, top), dim=0)
        snapped_candidate = candidates[side]
        snapped = torch.where(did_snap, snapped_candidate, p)
        return snapped, did_snap

    def choose_valid_boundary_ray_direction(self, origin: torch.Tensor, seed_i: torch.Tensor, seed_j: torch.Tensor, cad_domain: Any | None=None, u_periodic: bool=False, v_periodic: bool=False, all_seeds: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Validate both perpendicular directions and return the shortest valid hit."""
        snapped_origin, did_snap = self.snap_near_box_boundary_uv(origin)
        if bool(did_snap.detach().cpu().item()):
            direction = snapped_origin - origin
            norm = torch.linalg.vector_norm(direction)
            if bool((norm <= self.eps).detach().cpu().item()):
                direction = origin.new_zeros((2,))
            else:
                direction = direction / norm.clamp_min(self.eps)
            return (direction, snapped_origin, origin.new_tensor(0.0))

        tangent = self.periodic_difference(seed_j, seed_i, u_periodic, v_periodic)
        normal = torch.stack((-tangent[1], tangent[0]))
        normal = normal / torch.sqrt((normal * normal).sum() + self.eps)
        valid_candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        for direction in (normal, -normal):
            hit, t, valid = self.ray_box_hit_torch(origin, direction, u_periodic=u_periodic, v_periodic=v_periodic)
            midpoint = 0.5 * (origin + hit)
            boundary_distance = torch.minimum(torch.minimum(hit[0], 1.0 - hit[0]), torch.minimum(hit[1], 1.0 - hit[1])).abs()
            boundary_ok = boundary_distance <= 0.0001
            if cad_domain is not None and self.use_trim_activity:
                midpoint_ok = self.trim_gate(midpoint.unsqueeze(0), cad_domain)[0] > 0.5
            else:
                midpoint_ok = self.point_inside_box_np(midpoint.detach().cpu().numpy(), tol=1e-06)
            if bool((valid & boundary_ok & midpoint_ok).detach().cpu().item()):
                voronoi_margin = 0.0
                if all_seeds is not None and all_seeds.shape[0] > 2:
                    probe = origin + 0.0001 * direction
                    distances = self.periodic_distance(probe.unsqueeze(0), all_seeds, u_periodic=u_periodic, v_periodic=v_periodic)
                    pair_distance = 0.5 * (self.periodic_distance(probe, seed_i, u_periodic, v_periodic) + self.periodic_distance(probe, seed_j, u_periodic, v_periodic))
                    pair_mask = torch.isclose(distances, pair_distance, atol=1e-05, rtol=1e-05)
                    other_distances = distances.masked_fill(pair_mask, float('inf'))
                    voronoi_margin = float((other_distances.min() - pair_distance).detach().cpu().item())
                valid_candidates.append((t, direction, hit, voronoi_margin))
        if not valid_candidates:
            return None
        best_margin = max((item[3] for item in valid_candidates))
        best = [item for item in valid_candidates if item[3] >= best_margin - 1e-08]
        t, direction, hit, _ = min(best, key=lambda item: float(item[0].detach().cpu().item()))
        return (direction, hit, t)

    def build_scipy_voronoi_topology(self, seeds_uv: torch.Tensor, cad_domain: Any | None=None, u_periodic: bool=False, v_periodic: bool=False) -> dict[str, Any]:
        """
                seeds_uv: torch.Tensor[S,2]
                Returns topology dict built with scipy.spatial.Voronoi using detached NumPy seeds.
                """
        if not isinstance(seeds_uv, torch.Tensor):
            raise TypeError('seeds_uv must be a torch.Tensor.')
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError(f'seeds_uv must have shape [S, 2], got {tuple(seeds_uv.shape)}.')
        device = seeds_uv.device
        dtype = seeds_uv.dtype
        points_np = seeds_uv.detach().cpu().numpy()
        empty_long_2 = lambda: torch.empty((0, 2), dtype=torch.long, device=device)
        empty_long_3 = lambda: torch.empty((0, 3), dtype=torch.long, device=device)
        empty_float_2 = lambda: torch.empty((0, 2), dtype=dtype, device=device)

        def empty_topology() -> dict[str, Any]:
            return {'triples': empty_long_3(), 'vertex_type': torch.empty((0,), dtype=torch.long, device=device), 'vertex_seed_triples': empty_long_3(), 'boundary_origin_vertex': torch.empty((0,), dtype=torch.long, device=device), 'boundary_target_vertex': torch.empty((0,), dtype=torch.long, device=device), 'boundary_seed_pair': empty_long_2(), 'boundary_ray_dir': empty_float_2(), 'boundary_source_type': torch.empty((0,), dtype=torch.long, device=device), 'edges': empty_long_2(), 'edge_seed_pairs': empty_long_2(), 'edge_type': torch.empty((0,), dtype=torch.long, device=device), 'boundary_rays': empty_long_3(), 'boundary_ray_dirs': empty_float_2(), 'scipy_vertices_np': np.empty((0, 2), dtype=points_np.dtype), 'isolated_vertices': torch.empty((0,), dtype=torch.long, device=device), 'delaunay_triples_np': np.empty((0, 3), dtype=np.int64), 'diagnostics': {'num_finite_edges_inside': 0, 'num_finite_edges_clipped_once': 0, 'num_finite_edges_clipped_twice': 0, 'num_infinite_rays_clipped': 0, 'num_boundary_snapped_rays': 0, 'num_discarded_rays': 0, 'num_raw_scipy_vertices': 0, 'num_raw_boundary_vertices': 0, 'num_pruned_vertices': 0, 'num_final_vertices': 0, 'num_final_interior_vertices': 0, 'num_final_boundary_vertices': 0}}
        if points_np.shape[0] < 3:
            return empty_topology()
        try:
            vor = Voronoi(points_np)
        except Exception:
            return empty_topology()
        try:
            delaunay = Delaunay(points_np)
            delaunay_triples_np = delaunay.simplices
        except Exception:
            delaunay_triples_np = np.empty((0, 3), dtype=np.int64)
        scipy_vertices_np = vor.vertices
        if scipy_vertices_np.shape[0] == 0:
            triples = torch.empty((0, 3), dtype=torch.long, device=device)
        else:
            tree = cKDTree(points_np)
            _, triple_idx_np = tree.query(scipy_vertices_np, k=3)
            triples = torch.as_tensor(triple_idx_np, dtype=torch.long, device=device)
            if triples.ndim == 1:
                triples = triples.reshape(1, 3)
        num_raw_scipy_vertices = int(scipy_vertices_np.shape[0])
        vertex_seed_triples = triples.detach().cpu().tolist()
        vertex_type = [0] * num_raw_scipy_vertices
        boundary_origin_vertex = [-1] * num_raw_scipy_vertices
        boundary_target_vertex = [-1] * num_raw_scipy_vertices
        boundary_seed_pair = [[-1, -1] for _ in range(num_raw_scipy_vertices)]
        boundary_ray_dir = [[0.0, 0.0] for _ in range(num_raw_scipy_vertices)]
        boundary_source_type = [0] * num_raw_scipy_vertices
        edges: list[list[int]] = []
        edge_seed_pairs: list[list[int]] = []
        edge_types: list[int] = []
        boundary_rays: list[list[int]] = []
        boundary_ray_dirs: list[list[float]] = []
        diagnostics = {'num_finite_edges_inside': 0, 'num_finite_edges_clipped_once': 0, 'num_finite_edges_clipped_twice': 0, 'num_infinite_rays_clipped': 0, 'num_boundary_snapped_rays': 0, 'num_discarded_rays': 0, 'num_raw_scipy_vertices': num_raw_scipy_vertices}

        def add_boundary_vertex(origin_vertex: int, target_vertex: int, seed_i: int, seed_j: int, direction: np.ndarray, source_type: int) -> int:
            boundary_id = len(vertex_type)
            vertex_type.append(1)
            vertex_seed_triples.append([seed_i, seed_j, -1])
            boundary_origin_vertex.append(origin_vertex)
            boundary_target_vertex.append(target_vertex)
            boundary_seed_pair.append([seed_i, seed_j])
            boundary_ray_dir.append([float(direction[0]), float(direction[1])])
            boundary_source_type.append(source_type)
            return boundary_id

        # Graph edges must come from Voronoi ridges. Delaunay simplices are
        # exposed only as diagnostics and must not add simplex-adjacency edges.
        for seed_pair, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
            finite_vertices = [int(v) for v in ridge_vertices if int(v) >= 0]
            seed_i = int(seed_pair[0])
            seed_j = int(seed_pair[1])
            if len(finite_vertices) == 2:
                a, b = finite_vertices
                pa, pb = (scipy_vertices_np[a], scipy_vertices_np[b])
                clipped = self.segment_box_clip_np(pa, pb)
                if clipped is None:
                    continue
                _, _, t_enter, t_exit = clipped
                a_inside = self.point_inside_box_np(pa)
                b_inside = self.point_inside_box_np(pb)
                if a_inside and b_inside:
                    edges.append([a, b])
                    edge_seed_pairs.append([seed_i, seed_j])
                    edge_types.append(0)
                    diagnostics['num_finite_edges_inside'] += 1
                elif a_inside != b_inside:
                    inside_id, outside_id = (a, b) if a_inside else (b, a)
                    direction = scipy_vertices_np[outside_id] - scipy_vertices_np[inside_id]
                    direction /= np.linalg.norm(direction) + 1e-12
                    boundary_id = add_boundary_vertex(inside_id, outside_id, seed_i, seed_j, direction, 2)
                    edges.append([inside_id, boundary_id])
                    edge_seed_pairs.append([seed_i, seed_j])
                    edge_types.append(1)
                    diagnostics['num_finite_edges_clipped_once'] += 1
                elif t_exit - t_enter > 1e-12:
                    direction_ab = pb - pa
                    direction_ab /= np.linalg.norm(direction_ab) + 1e-12
                    entry_id = add_boundary_vertex(a, b, seed_i, seed_j, direction_ab, 2)
                    exit_id = add_boundary_vertex(b, a, seed_i, seed_j, -direction_ab, 2)
                    edges.append([entry_id, exit_id])
                    edge_seed_pairs.append([seed_i, seed_j])
                    edge_types.append(3)
                    diagnostics['num_finite_edges_clipped_twice'] += 1
            elif len(finite_vertices) == 1 and any((int(v) == -1 for v in ridge_vertices)):
                finite_v = finite_vertices[0]
                origin = torch.as_tensor(scipy_vertices_np[finite_v], dtype=dtype, device=device)
                selected = self.choose_valid_boundary_ray_direction(origin, seeds_uv[seed_i].detach(), seeds_uv[seed_j].detach(), cad_domain=cad_domain, u_periodic=u_periodic, v_periodic=v_periodic, all_seeds=seeds_uv.detach())
                if selected is None:
                    diagnostics['num_discarded_rays'] += 1
                    continue
                direction_t, _, ray_t = selected
                source_type = 1
                if bool((ray_t <= self.eps).detach().cpu().item()):
                    diagnostics['num_boundary_snapped_rays'] += 1
                    source_type = 5
                direction = direction_t.detach().cpu().numpy()
                boundary_id = add_boundary_vertex(finite_v, -1, seed_i, seed_j, direction, source_type)
                edges.append([finite_v, boundary_id])
                edge_seed_pairs.append([seed_i, seed_j])
                edge_types.append(1)
                boundary_rays.append([finite_v, seed_i, seed_j])
                boundary_ray_dirs.append([float(direction[0]), float(direction[1])])
                diagnostics['num_infinite_rays_clipped'] += 1
        referenced = set()
        for e in edges:
            referenced.add(e[0])
            referenced.add(e[1])
        for r in boundary_rays:
            referenced.add(r[0])
        all_ids = set(range(num_raw_scipy_vertices))
        isolated = sorted(list(all_ids - referenced))
        isolated_t = torch.as_tensor(isolated, dtype=torch.long, device=device)
        edges_t = torch.as_tensor(edges, dtype=torch.long, device=device)
        if edges_t.numel() == 0:
            edges_t = torch.empty((0, 2), dtype=torch.long, device=device)
        else:
            edges_t = edges_t.reshape(-1, 2)
        edge_seed_pairs_t = torch.as_tensor(edge_seed_pairs, dtype=torch.long, device=device)
        if edge_seed_pairs_t.numel() == 0:
            edge_seed_pairs_t = torch.empty((0, 2), dtype=torch.long, device=device)
        else:
            edge_seed_pairs_t = edge_seed_pairs_t.reshape(-1, 2)
        boundary_rays_t = torch.as_tensor(boundary_rays, dtype=torch.long, device=device)
        if boundary_rays_t.numel() == 0:
            boundary_rays_t = torch.empty((0, 3), dtype=torch.long, device=device)
        else:
            boundary_rays_t = boundary_rays_t.reshape(-1, 3)
        boundary_ray_dirs_t = torch.as_tensor(boundary_ray_dirs, dtype=seeds_uv.dtype, device=device)
        if boundary_ray_dirs_t.numel() == 0:
            boundary_ray_dirs_t = torch.empty((0, 2), dtype=seeds_uv.dtype, device=device)
        else:
            boundary_ray_dirs_t = boundary_ray_dirs_t.reshape(-1, 2)
        vertex_seed_triples_t = torch.as_tensor(vertex_seed_triples, dtype=torch.long, device=device).reshape(-1, 3)
        vertex_type_t = torch.as_tensor(vertex_type, dtype=torch.long, device=device)
        boundary_origin_vertex_t = torch.as_tensor(boundary_origin_vertex, dtype=torch.long, device=device)
        boundary_target_vertex_t = torch.as_tensor(boundary_target_vertex, dtype=torch.long, device=device)
        boundary_seed_pair_t = torch.as_tensor(boundary_seed_pair, dtype=torch.long, device=device).reshape(-1, 2)
        boundary_ray_dir_t = torch.as_tensor(boundary_ray_dir, dtype=dtype, device=device).reshape(-1, 2)
        boundary_source_type_t = torch.as_tensor(boundary_source_type, dtype=torch.long, device=device)
        diagnostics['num_raw_boundary_vertices'] = len(vertex_type) - num_raw_scipy_vertices
        return {'triples': vertex_seed_triples_t, 'vertex_type': vertex_type_t, 'vertex_seed_triples': vertex_seed_triples_t, 'boundary_origin_vertex': boundary_origin_vertex_t, 'boundary_target_vertex': boundary_target_vertex_t, 'boundary_seed_pair': boundary_seed_pair_t, 'boundary_ray_dir': boundary_ray_dir_t, 'boundary_source_type': boundary_source_type_t, 'edges': edges_t, 'edge_seed_pairs': edge_seed_pairs_t, 'edge_type': torch.as_tensor(edge_types, dtype=torch.long, device=device), 'boundary_rays': boundary_rays_t, 'boundary_ray_dirs': boundary_ray_dirs_t, 'scipy_vertices_np': scipy_vertices_np, 'isolated_vertices': isolated_t, 'delaunay_triples_np': delaunay_triples_np, 'diagnostics': diagnostics}

    def prune_graph_vertices(self, nodes_uv: torch.Tensor, vertex_type: torch.Tensor, vertex_seed_triples: torch.Tensor, boundary_origin_vertex: torch.Tensor, boundary_target_vertex: torch.Tensor, boundary_seed_pair: torch.Tensor, boundary_ray_dir: torch.Tensor, boundary_source_type: torch.Tensor, edges: torch.Tensor, edge_seed_pairs: torch.Tensor, edge_type: torch.Tensor, alpha: torch.Tensor | None=None, keep_isolated_vertices: bool=False) -> dict[str, torch.Tensor | int | None]:
        """Compact topology to vertices participating in the final edge graph."""
        num_vertices = int(nodes_uv.shape[0])
        device = nodes_uv.device
        active_mask = torch.zeros((num_vertices,), dtype=torch.bool, device=device)
        if edges.numel() > 0:
            active_mask[edges.reshape(-1)] = True
        if keep_isolated_vertices:
            active_mask[:] = True
        active_ids = torch.nonzero(active_mask, as_tuple=False).flatten()
        old_to_new = torch.full((num_vertices,), -1, dtype=torch.long, device=device)
        old_to_new[active_ids] = torch.arange(active_ids.numel(), device=device)
        compact_edges = old_to_new[edges] if edges.numel() > 0 else edges.reshape(0, 2)

        def remap_reference(values: torch.Tensor) -> torch.Tensor:
            compact = values[active_ids].clone()
            valid = (compact >= 0) & (compact < num_vertices)
            compact[valid] = old_to_new[compact[valid]]
            compact[~valid] = -1
            return compact
        compact_type = vertex_type[active_ids]
        return {'nodes_uv': nodes_uv[active_ids], 'vertex_type': compact_type, 'vertex_seed_triples': vertex_seed_triples[active_ids], 'boundary_origin_vertex': remap_reference(boundary_origin_vertex), 'boundary_target_vertex': remap_reference(boundary_target_vertex), 'boundary_seed_pair': boundary_seed_pair[active_ids], 'boundary_ray_dir': boundary_ray_dir[active_ids], 'boundary_source_type': boundary_source_type[active_ids], 'edges': compact_edges, 'edge_seed_pairs': edge_seed_pairs, 'edge_type': edge_type, 'alpha': None if alpha is None else alpha[active_ids], 'old_to_new': old_to_new, 'active_vertex_ids': active_ids, 'num_pruned_vertices': num_vertices - int(active_ids.numel())}

    def differentiable_vertices_from_topology(self, seeds_uv: torch.Tensor, vertex_type: torch.Tensor, vertex_seed_triples: torch.Tensor, boundary_origin_vertex: torch.Tensor, boundary_seed_pair: torch.Tensor, boundary_ray_dir: torch.Tensor, u_periodic: bool=False, v_periodic: bool=False, cad_domain: Any | None=None, boundary_target_vertex: torch.Tensor | None=None, boundary_source_type: torch.Tensor | None=None) -> torch.Tensor:
        """Reconstruct unified SciPy topology with differentiable coordinates."""
        num_vertices = vertex_type.shape[0]
        if num_vertices == 0:
            return torch.empty((0, 2), dtype=seeds_uv.dtype, device=seeds_uv.device)
        zero_node = torch.zeros((2,), dtype=seeds_uv.dtype, device=seeds_uv.device)
        node_values = [zero_node for _ in range(num_vertices)]
        if boundary_target_vertex is None:
            boundary_target_vertex = torch.full_like(boundary_origin_vertex, -1)
        if boundary_source_type is None:
            boundary_source_type = vertex_type
        interior_ids = torch.nonzero(vertex_type == 0, as_tuple=False).flatten()
        if interior_ids.numel() > 0:
            triples = vertex_seed_triples[interior_ids]
            interior_nodes = self.differentiable_vertices_from_triples(seeds_uv, triples, u_periodic, v_periodic)
            for local_id, vertex_id in enumerate(interior_ids.tolist()):
                node_values[vertex_id] = interior_nodes[local_id]
        for boundary_id in torch.nonzero(vertex_type == 1, as_tuple=False).flatten().tolist():
            origin_id = int(boundary_origin_vertex[boundary_id].item())
            if origin_id < 0 or origin_id >= num_vertices:
                continue
            pair = boundary_seed_pair[boundary_id]
            i, j = (int(pair[0].item()), int(pair[1].item()))
            stored_direction = boundary_ray_dir[boundary_id].to(dtype=seeds_uv.dtype)
            source_type = int(boundary_source_type[boundary_id].item())
            if source_type == 2:
                target_id = int(boundary_target_vertex[boundary_id].item())
                if target_id < 0 or target_id >= num_vertices:
                    continue
                direction = node_values[target_id] - node_values[origin_id]
                direction = direction / torch.sqrt((direction * direction).sum() + self.eps)
            elif source_type == 5:
                snapped, _ = self.snap_near_box_boundary_uv(node_values[origin_id])
                node_values[boundary_id] = snapped
                continue
            elif 0 <= i < seeds_uv.shape[0] and 0 <= j < seeds_uv.shape[0]:
                tangent = self.periodic_difference(seeds_uv[j], seeds_uv[i], u_periodic, v_periodic)
                direction = torch.stack((-tangent[1], tangent[0]))
                direction = direction / torch.sqrt((direction * direction).sum() + self.eps)
                sign = torch.where(torch.dot(direction, stored_direction) < 0, -torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device), torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device))
                direction = direction * sign
            else:
                direction = stored_direction
                direction = direction / torch.sqrt((direction * direction).sum() + self.eps)
            node_values[boundary_id] = self.ray_box_intersection_uv(node_values[origin_id], direction, u_periodic=u_periodic, v_periodic=v_periodic)
        return torch.stack(node_values, dim=0)

    def _box_bisector_intersections(self, seed_i: torch.Tensor, seed_j: torch.Tensor, tol: float=1e-07) -> list[torch.Tensor]:
        """Return the two intersections of a pair bisector with the UV box."""
        midpoint = 0.5 * (seed_i + seed_j)
        tangent = seed_j - seed_i
        direction = torch.stack((-tangent[1], tangent[0]))
        candidates: list[torch.Tensor] = []
        if bool((direction[0].abs() > tol).detach().cpu().item()):
            for u_value in (0.0, 1.0):
                u = torch.as_tensor(u_value, dtype=midpoint.dtype, device=midpoint.device)
                t = (u - midpoint[0]) / direction[0]
                v = midpoint[1] + t * direction[1]
                if bool(((v >= -tol) & (v <= 1.0 + tol)).detach().cpu().item()):
                    candidates.append(torch.stack((u, v.clamp(0.0, 1.0))))
        if bool((direction[1].abs() > tol).detach().cpu().item()):
            for v_value in (0.0, 1.0):
                v = torch.as_tensor(v_value, dtype=midpoint.dtype, device=midpoint.device)
                t = (v - midpoint[1]) / direction[1]
                u = midpoint[0] + t * direction[0]
                if bool(((u >= -tol) & (u <= 1.0 + tol)).detach().cpu().item()):
                    candidates.append(torch.stack((u.clamp(0.0, 1.0), v)))
        unique: list[torch.Tensor] = []
        for candidate in candidates:
            if not any((torch.linalg.vector_norm(candidate.detach() - other.detach()) <= tol for other in unique)):
                unique.append(candidate)
        return unique

    def _pair_boundary_candidate_alpha(self, seeds_uv: torch.Tensor, pair: torch.Tensor, candidate: torch.Tensor, seed_activity: torch.Tensor | None=None) -> torch.Tensor:
        """Soft nearest-pair validity at a bisector-boundary intersection."""
        i, j = (int(pair[0].item()), int(pair[1].item()))
        distances = self.periodic_distance(candidate.unsqueeze(0), seeds_uv)
        pair_distance = 0.5 * (distances[i] + distances[j])
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask[i] = False
        mask[j] = False
        if bool(mask.any().detach().cpu().item()):
            margin = distances[mask].min() - pair_distance
            nearest_gate = torch.sigmoid(margin / self._tau_tensor(self.tau_voronoi, seeds_uv))
        else:
            nearest_gate = torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device)
        equality_gate = torch.exp(-((distances[i] - distances[j]) / self._tau_tensor(self.tau_voronoi, seeds_uv)) ** 2)
        activity = torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device)
        if seed_activity is not None:
            activity = seed_activity[i] * seed_activity[j]
        return (activity * nearest_gate * equality_gate).clamp(0.0, 1.0)

    def _cad_bisector_intersections(self, seed_i: torch.Tensor, seed_j: torch.Tensor, cad_domain: Any, samples: int=129) -> list[torch.Tensor]:
        """Find trim-boundary crossings along the box-clipped pair bisector."""
        segment = self._box_bisector_intersections(seed_i, seed_j)
        if len(segment) != 2:
            return []
        t = torch.linspace(0.0, 1.0, samples, dtype=seed_i.dtype, device=seed_i.device)
        points = segment[0].unsqueeze(0) + t.unsqueeze(1) * (segment[1] - segment[0]).unsqueeze(0)
        if hasattr(cad_domain, 'sample_trim_sdf'):
            values = cad_domain.sample_trim_sdf(points)
            values = torch.as_tensor(values, dtype=seed_i.dtype, device=seed_i.device).reshape(-1)
        elif hasattr(cad_domain, 'smooth_inside_activity'):
            values = cad_domain.smooth_inside_activity(points, tau=self.tau_trim)
            values = torch.as_tensor(values, dtype=seed_i.dtype, device=seed_i.device).reshape(-1) - 0.5
        else:
            return []
        intersections: list[torch.Tensor] = []
        for index in range(samples - 1):
            a, b = (values[index], values[index + 1])
            crosses = (a == 0) | (b == 0) | ((a < 0) != (b < 0))
            if not bool(crosses.detach().cpu().item()):
                continue
            weight = (a / (a - b + self.eps)).clamp(0.0, 1.0)
            point = points[index] + weight * (points[index + 1] - points[index])
            if not intersections or torch.linalg.vector_norm(point.detach() - intersections[-1].detach()) > 0.0001:
                intersections.append(point)
        return intersections

    def add_pair_boundary_candidates(self, seeds_uv: torch.Tensor, nodes_uv: torch.Tensor, node_alpha: torch.Tensor, node_type: torch.Tensor, node_seed_triples: torch.Tensor, boundary_seed_pair: torch.Tensor, boundary_source_type: torch.Tensor, edges: torch.Tensor, edge_seed_pairs: torch.Tensor, edge_type: torch.Tensor, cad_domain: Any | None=None, seed_activity: torch.Tensor | None=None, hard_validity: bool=True, tol: float=0.0001) -> dict[str, torch.Tensor]:
        """Add missing valid pair-bisector intersections and their Voronoi edges."""
        device, dtype = (seeds_uv.device, seeds_uv.dtype)
        pairs = torch.combinations(torch.arange(seeds_uv.shape[0], device=device), r=2)
        added_nodes: list[torch.Tensor] = []
        added_alpha: list[torch.Tensor] = []
        added_pairs: list[list[int]] = []
        added_edges: list[list[int]] = []
        added_edge_pairs: list[list[int]] = []
        added_edge_types: list[int] = []
        for pair in pairs:
            i, j = (int(pair[0].item()), int(pair[1].item()))
            if cad_domain is None:
                candidates = self._box_bisector_intersections(seeds_uv[i], seeds_uv[j])
            elif hasattr(cad_domain, 'intersect_bisector_boundary'):
                raw = cad_domain.intersect_bisector_boundary(seeds_uv[i], seeds_uv[j])
                raw = torch.as_tensor(raw, dtype=dtype, device=device).reshape(-1, 2)
                candidates = list(raw.unbind(0))
            else:
                candidates = self._cad_bisector_intersections(seeds_uv[i], seeds_uv[j], cad_domain)
            pair_in_triple = (node_seed_triples == i).any(dim=1) & (node_seed_triples == j).any(dim=1) & (node_type == 0) if node_seed_triples.numel() > 0 else torch.zeros((nodes_uv.shape[0],), dtype=torch.bool, device=device)
            interior_ids = torch.nonzero(pair_in_triple, as_tuple=False).flatten()
            if interior_ids.numel() > 0:
                inside = (nodes_uv[interior_ids, 0] >= -tol) & (nodes_uv[interior_ids, 0] <= 1.0 + tol) & (nodes_uv[interior_ids, 1] >= -tol) & (nodes_uv[interior_ids, 1] <= 1.0 + tol) & (node_alpha[interior_ids] >= 0.5)
                interior_ids = interior_ids[inside]
            valid_for_pair: list[tuple[torch.Tensor, torch.Tensor]] = []
            for candidate in candidates:
                alpha = self._pair_boundary_candidate_alpha(seeds_uv, pair, candidate, seed_activity)
                if hard_validity and float(alpha.detach().cpu().item()) < 0.5:
                    continue
                existing_pair = (boundary_seed_pair[:, 0] == i) & (boundary_seed_pair[:, 1] == j) | (boundary_seed_pair[:, 0] == j) & (boundary_seed_pair[:, 1] == i) if boundary_seed_pair.numel() > 0 else torch.zeros((nodes_uv.shape[0],), dtype=torch.bool, device=device)
                existing_ids = torch.nonzero(existing_pair, as_tuple=False).flatten()
                if existing_ids.numel() > 0 and bool((torch.linalg.vector_norm(nodes_uv[existing_ids].detach() - candidate.detach(), dim=1).min() <= tol).detach().cpu().item()):
                    continue
                valid_for_pair.append((candidate, alpha))
            if interior_ids.numel() == 0 and len(valid_for_pair) == 2:
                midpoint = 0.5 * (valid_for_pair[0][0] + valid_for_pair[1][0])
                midpoint_alpha = self._pair_boundary_candidate_alpha(seeds_uv, pair, midpoint, seed_activity)
                domain_ok = self.trim_gate(midpoint.unsqueeze(0), cad_domain)[0] > 0.5 if cad_domain is not None else torch.ones((), dtype=torch.bool, device=device)
                if float(midpoint_alpha.detach().cpu().item()) < 0.5 or not bool(domain_ok.detach().cpu().item()):
                    valid_for_pair = []
            new_ids: list[int] = []
            for candidate, alpha in valid_for_pair:
                target = None
                if interior_ids.numel() > 0:
                    distances = torch.linalg.vector_norm(nodes_uv[interior_ids].detach() - candidate.detach(), dim=1)
                    target = int(interior_ids[torch.argmin(distances)].item())
                    segment_midpoint = 0.5 * (candidate + nodes_uv[target])
                    midpoint_alpha = self._pair_boundary_candidate_alpha(seeds_uv, pair, segment_midpoint, seed_activity)
                    domain_ok = self.trim_gate(segment_midpoint.unsqueeze(0), cad_domain)[0] > 0.5 if cad_domain is not None else torch.ones((), dtype=torch.bool, device=device)
                    if float(midpoint_alpha.detach().cpu().item()) < 0.5 or not bool(domain_ok.detach().cpu().item()):
                        continue
                new_id = nodes_uv.shape[0] + len(added_nodes)
                added_nodes.append(candidate)
                added_alpha.append(alpha)
                added_pairs.append([i, j])
                new_ids.append(new_id)
                if target is not None:
                    added_edges.append([target, new_id])
                    added_edge_pairs.append([i, j])
                    added_edge_types.append(1)
            if interior_ids.numel() == 0 and len(new_ids) == 2:
                midpoint = 0.5 * (valid_for_pair[0][0] + valid_for_pair[1][0])
                midpoint_alpha = self._pair_boundary_candidate_alpha(seeds_uv, pair, midpoint, seed_activity)
                if float(midpoint_alpha.detach().cpu().item()) >= 0.5:
                    added_edges.append(new_ids)
                    added_edge_pairs.append([i, j])
                    added_edge_types.append(3)
        if not added_nodes:
            return {'nodes_uv': nodes_uv, 'node_alpha': node_alpha, 'node_type': node_type, 'node_seed_triples': node_seed_triples, 'boundary_seed_pair': boundary_seed_pair, 'boundary_source_type': boundary_source_type, 'edges': edges, 'edge_seed_pairs': edge_seed_pairs, 'edge_type': edge_type}
        added_nodes_t = torch.stack(added_nodes)
        added_pairs_t = torch.as_tensor(added_pairs, dtype=torch.long, device=device)
        added_count = added_nodes_t.shape[0]
        nodes_uv = torch.cat((nodes_uv, added_nodes_t), dim=0)
        node_alpha = torch.cat((node_alpha, torch.stack(added_alpha)), dim=0)
        node_type = torch.cat((node_type, torch.ones(added_count, dtype=torch.long, device=device)))
        node_seed_triples = torch.cat((node_seed_triples, torch.cat((added_pairs_t, -torch.ones((added_count, 1), dtype=torch.long, device=device)), dim=1)), dim=0)
        boundary_seed_pair = torch.cat((boundary_seed_pair, added_pairs_t), dim=0)
        boundary_source_type = torch.cat((boundary_source_type, torch.full((added_count,), 3, dtype=torch.long, device=device)))
        if added_edges:
            edges = torch.cat((edges, torch.as_tensor(added_edges, dtype=torch.long, device=device)), dim=0)
            edge_seed_pairs = torch.cat((edge_seed_pairs, torch.as_tensor(added_edge_pairs, dtype=torch.long, device=device)), dim=0)
            edge_type = torch.cat((edge_type, torch.as_tensor(added_edge_types, dtype=torch.long, device=device)))
        return {'nodes_uv': nodes_uv, 'node_alpha': node_alpha, 'node_type': node_type, 'node_seed_triples': node_seed_triples, 'boundary_seed_pair': boundary_seed_pair, 'boundary_source_type': boundary_source_type, 'edges': edges, 'edge_seed_pairs': edge_seed_pairs, 'edge_type': edge_type}

    def add_box_shell_corners(self, nodes_uv: torch.Tensor, node_alpha: torch.Tensor, node_type: torch.Tensor, node_seed_triples: torch.Tensor, boundary_seed_pair: torch.Tensor, boundary_source_type: torch.Tensor, tol: float=0.0001) -> dict[str, torch.Tensor]:
        """Append missing UV-box corners as boundary shell nodes (source type 4)."""
        device, dtype = (nodes_uv.device, nodes_uv.dtype)
        corners = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=dtype, device=device)
        missing = []
        for corner in corners:
            if nodes_uv.numel() == 0 or not bool((torch.linalg.vector_norm(nodes_uv.detach() - corner, dim=1) <= tol).any().item()):
                missing.append(corner)
        if not missing:
            return {'nodes_uv': nodes_uv, 'node_alpha': node_alpha, 'node_type': node_type, 'node_seed_triples': node_seed_triples, 'boundary_seed_pair': boundary_seed_pair, 'boundary_source_type': boundary_source_type}
        added = torch.stack(missing)
        count = added.shape[0]
        return {'nodes_uv': torch.cat((nodes_uv, added), dim=0), 'node_alpha': torch.cat((node_alpha, torch.ones(count, dtype=dtype, device=device))), 'node_type': torch.cat((node_type, torch.ones(count, dtype=torch.long, device=device))), 'node_seed_triples': torch.cat((node_seed_triples, torch.full((count, 3), -1, dtype=torch.long, device=device))), 'boundary_seed_pair': torch.cat((boundary_seed_pair, torch.full((count, 2), -1, dtype=torch.long, device=device))), 'boundary_source_type': torch.cat((boundary_source_type, torch.full((count,), 4, dtype=torch.long, device=device)))}

    def build_boundary_loop_edges(self, nodes_uv: torch.Tensor, vertex_type: torch.Tensor, cad_domain: Any | None=None, tol: float=0.0001) -> tuple[torch.Tensor, torch.Tensor]:
        """Connect boundary nodes cyclically in shell-parameter order."""
        device = nodes_uv.device
        boundary_ids = torch.nonzero(vertex_type == 1, as_tuple=False).flatten()
        if boundary_ids.numel() < 2:
            return (torch.empty((0, 2), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device))
        p = nodes_uv.detach()[boundary_ids]
        loops: list[torch.Tensor] = []
        if cad_domain is None:
            u, v = (p[:, 0], p[:, 1])
            parameter = torch.empty_like(u)
            bottom = torch.abs(v) <= tol
            right = ~bottom & (torch.abs(u - 1.0) <= tol)
            top = ~bottom & ~right & (torch.abs(v - 1.0) <= tol)
            left = ~(bottom | right | top)
            parameter[bottom] = u[bottom]
            parameter[right] = 1.0 + v[right]
            parameter[top] = 3.0 - u[top]
            parameter[left] = 4.0 - v[left]
            loops = [boundary_ids[torch.argsort(parameter)]]
        else:
            used_projection_hook = False
            if hasattr(cad_domain, 'boundary_parameter'):
                result = cad_domain.boundary_parameter(nodes_uv[boundary_ids])
            elif hasattr(cad_domain, 'project_to_boundary_with_parameter'):
                used_projection_hook = True
                result = cad_domain.project_to_boundary_with_parameter(nodes_uv[boundary_ids])
            else:
                raise AttributeError('CAD shell completion requires cad_domain.boundary_parameter(P_uv) or cad_domain.project_to_boundary_with_parameter(P_uv).')
            if isinstance(result, dict):
                loop_id = torch.as_tensor(result.get('loop_id', 0), device=device).reshape(-1)
                parameter_value = result.get('parameter', result.get('s'))
                if parameter_value is None:
                    raise KeyError("CAD boundary parameter result must contain 'parameter' or 's'.")
                parameter = torch.as_tensor(parameter_value, device=device).reshape(-1)
            elif isinstance(result, (tuple, list)) and len(result) >= 2:
                if used_projection_hook and len(result) == 2:
                    loop_id = torch.zeros(boundary_ids.shape[0], dtype=torch.long, device=device)
                else:
                    loop_id = torch.as_tensor(result[-2], device=device).reshape(-1)
                parameter = torch.as_tensor(result[-1], device=device).reshape(-1)
            else:
                parameter = torch.as_tensor(result, device=device).reshape(-1)
                loop_id = torch.zeros_like(parameter, dtype=torch.long)
            for value in torch.unique(loop_id).tolist():
                ids = torch.nonzero(loop_id == value, as_tuple=False).flatten()
                loops.append(boundary_ids[ids[torch.argsort(parameter[ids])]])
        edges_list: list[list[int]] = []
        for ordered_ids in loops:
            if ordered_ids.numel() < 2:
                continue
            ids = ordered_ids.tolist()
            edges_list.extend([[int(a), int(b)] for a, b in zip(ids, ids[1:] + ids[:1]) if a != b])
        edge_index = torch.as_tensor(edges_list, dtype=torch.long, device=device).reshape(-1, 2)
        return (edge_index, torch.full((edge_index.shape[0],), 4, dtype=torch.long, device=device))

    def differentiable_vertices_from_triples(self, seeds_uv: torch.Tensor, triples: torch.Tensor, u_periodic: bool=False, v_periodic: bool=False) -> torch.Tensor:
        """
                Recompute SciPy-selected vertices using differentiable PyTorch circumcenter.
                """
        if triples.numel() == 0:
            return torch.empty((0, 2), dtype=seeds_uv.dtype, device=seeds_uv.device)
        _, P_uv, _, _ = self.circumcenters_from_triples(seeds_uv, triples, u_periodic, v_periodic)
        return P_uv

    def forward_scipy_topology(self, seeds_uv: torch.Tensor, cad_domain: Any | None=None, u_periodic: bool=False, v_periodic: bool=False, return_xyz: bool | None=None, keep_isolated_vertices: bool=False) -> dict[str, Any]:
        """
                - SciPy builds graph topology without gradients.
                - PyTorch recomputes vertex positions with gradients.
                - Therefore gradients flow through geometry, not topology.
                """
        if not isinstance(seeds_uv, torch.Tensor):
            raise TypeError('seeds_uv must be a torch.Tensor.')
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError(f'seeds_uv must have shape [S, 2], got {tuple(seeds_uv.shape)}.')
        if not seeds_uv.is_floating_point():
            raise TypeError('seeds_uv must be a floating point tensor.')
        want_xyz = self.return_xyz if return_xyz is None else bool(return_xyz)
        original_seeds_uv = seeds_uv
        seed_active_ids = torch.arange(seeds_uv.shape[0], dtype=torch.long, device=seeds_uv.device)
        seed_active_mask = torch.ones((seeds_uv.shape[0],), dtype=torch.bool, device=seeds_uv.device)
        seed_activity_weight = torch.ones((seeds_uv.shape[0],), dtype=seeds_uv.dtype, device=seeds_uv.device)

        if self.use_seed_activation:
            seed_domain_sdf = None
            seed_domain_mask = None
            if cad_domain is not None and self.use_trim_activity:
                if callable(getattr(cad_domain, 'sample_trim_sdf', None)):
                    seed_domain_sdf = cad_domain.sample_trim_sdf
                elif callable(getattr(cad_domain, 'smooth_inside_activity', None)):
                    seed_domain_mask = lambda points: cad_domain.smooth_inside_activity(points, tau=self.tau_trim)
            seed_active_ids, seed_active_mask, seed_activity_weight = self._seed_activation_state(
                seeds_uv,
                seed_domain_sdf=seed_domain_sdf,
                seed_domain_mask=seed_domain_mask,
                seed_domain_mask_threshold=self.seed_domain_mask_threshold,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
            if seed_active_ids.numel() >= self.min_active_seeds:
                seeds_uv = seeds_uv[seed_active_ids]
            else:
                seeds_uv = seeds_uv.new_empty((0, 2))

        with torch.no_grad():
            topo = self.build_scipy_voronoi_topology(seeds_uv, cad_domain=cad_domain, u_periodic=u_periodic, v_periodic=v_periodic)
        vertices_uv = self.differentiable_vertices_from_topology(seeds_uv=seeds_uv, vertex_type=topo['vertex_type'], vertex_seed_triples=topo['vertex_seed_triples'], boundary_origin_vertex=topo['boundary_origin_vertex'], boundary_seed_pair=topo['boundary_seed_pair'], boundary_ray_dir=topo['boundary_ray_dir'], u_periodic=u_periodic, v_periodic=v_periodic, cad_domain=cad_domain, boundary_target_vertex=topo['boundary_target_vertex'], boundary_source_type=topo['boundary_source_type'])
        alpha = torch.ones((vertices_uv.shape[0],), dtype=seeds_uv.dtype, device=seeds_uv.device)
        if cad_domain is not None and self.use_trim_activity:
            alpha = alpha * self.trim_gate(vertices_uv, cad_domain)
        else:
            alpha = alpha * self.box_gate(vertices_uv, u_periodic, v_periodic)
        if cad_domain is None:
            boundary_mask = topo['vertex_type'] == 1
            alpha = torch.where(boundary_mask, torch.ones_like(alpha), alpha)
        num_before_pair_completion = int(vertices_uv.shape[0])
        augmented = self.add_pair_boundary_candidates(seeds_uv=seeds_uv, nodes_uv=vertices_uv, node_alpha=alpha, node_type=topo['vertex_type'], node_seed_triples=topo['vertex_seed_triples'], boundary_seed_pair=topo['boundary_seed_pair'], boundary_source_type=topo['boundary_source_type'], edges=topo['edges'], edge_seed_pairs=topo['edge_seed_pairs'], edge_type=topo['edge_type'], cad_domain=cad_domain, hard_validity=True)
        num_pair_boundary_vertices = int(augmented['nodes_uv'].shape[0]) - num_before_pair_completion
        if cad_domain is None:
            augmented.update(self.add_box_shell_corners(nodes_uv=augmented['nodes_uv'], node_alpha=augmented['node_alpha'], node_type=augmented['node_type'], node_seed_triples=augmented['node_seed_triples'], boundary_seed_pair=augmented['boundary_seed_pair'], boundary_source_type=augmented['boundary_source_type']))
        num_corner_vertices = int((augmented['boundary_source_type'] == 4).sum().item()) - int((topo['boundary_source_type'] == 4).sum().item())
        total_added = int(augmented['nodes_uv'].shape[0]) - int(vertices_uv.shape[0])
        vertices_uv = augmented['nodes_uv']
        alpha = augmented['node_alpha']
        topo['vertex_type'] = augmented['node_type']
        topo['vertex_seed_triples'] = augmented['node_seed_triples']
        topo['boundary_seed_pair'] = augmented['boundary_seed_pair']
        topo['boundary_source_type'] = augmented['boundary_source_type']
        topo['edges'] = augmented['edges']
        topo['edge_seed_pairs'] = augmented['edge_seed_pairs']
        topo['edge_type'] = augmented['edge_type']
        if total_added > 0:
            topo['boundary_origin_vertex'] = torch.cat((topo['boundary_origin_vertex'], torch.full((total_added,), -1, dtype=torch.long, device=seeds_uv.device)))
            topo['boundary_target_vertex'] = torch.cat((topo['boundary_target_vertex'], torch.full((total_added,), -1, dtype=torch.long, device=seeds_uv.device)))
            topo['boundary_ray_dir'] = torch.cat((topo['boundary_ray_dir'], torch.zeros((total_added, 2), dtype=seeds_uv.dtype, device=seeds_uv.device)))
        loop_edges, loop_edge_type = self.build_boundary_loop_edges(vertices_uv, topo['vertex_type'], cad_domain=cad_domain)
        base_edges = topo['edges']
        edges = torch.cat((base_edges, loop_edges), dim=0)
        edge_type = torch.cat((topo['edge_type'], loop_edge_type), dim=0)
        loop_seed_pairs = torch.full((loop_edges.shape[0], 2), -1, dtype=torch.long, device=seeds_uv.device)
        edge_seed_pairs = torch.cat((topo['edge_seed_pairs'], loop_seed_pairs), dim=0)
        pruned = self.prune_graph_vertices(nodes_uv=vertices_uv, vertex_type=topo['vertex_type'], vertex_seed_triples=topo['vertex_seed_triples'], boundary_origin_vertex=topo['boundary_origin_vertex'], boundary_target_vertex=topo['boundary_target_vertex'], boundary_seed_pair=topo['boundary_seed_pair'], boundary_ray_dir=topo['boundary_ray_dir'], boundary_source_type=topo['boundary_source_type'], edges=edges, edge_seed_pairs=edge_seed_pairs, edge_type=edge_type, alpha=alpha, keep_isolated_vertices=keep_isolated_vertices)
        inactive_vertex_ids = torch.nonzero(pruned['old_to_new'] < 0, as_tuple=False).flatten()
        pruned_vertices_uv = vertices_uv[inactive_vertex_ids]
        pruned_vertex_type = topo['vertex_type'][inactive_vertex_ids]
        vertices_uv = pruned['nodes_uv']
        alpha = pruned['alpha']
        edges = pruned['edges']
        edge_seed_pairs = pruned['edge_seed_pairs']
        edge_type = pruned['edge_type']
        for key in ('vertex_type', 'vertex_seed_triples', 'boundary_origin_vertex', 'boundary_target_vertex', 'boundary_seed_pair', 'boundary_ray_dir', 'boundary_source_type'):
            topo[key] = pruned[key]
        old_to_new = pruned['old_to_new']
        boundary_rays = topo['boundary_rays'].clone()
        if boundary_rays.numel() > 0:
            mapped_ray_origins = old_to_new[boundary_rays[:, 0]]
            keep_rays = mapped_ray_origins >= 0
            boundary_rays = boundary_rays[keep_rays]
            boundary_rays[:, 0] = mapped_ray_origins[keep_rays]
            boundary_ray_dirs = topo['boundary_ray_dirs'][keep_rays]
        else:
            boundary_ray_dirs = topo['boundary_ray_dirs']
        diagnostics = dict(topo['diagnostics'])
        diagnostics['num_raw_boundary_vertices'] = diagnostics.get('num_raw_boundary_vertices', 0) + total_added
        diagnostics.update({'num_pair_boundary_vertices': num_pair_boundary_vertices, 'num_corner_shell_vertices': num_corner_vertices, 'num_pruned_vertices': pruned['num_pruned_vertices'], 'num_final_vertices': int(vertices_uv.shape[0]), 'num_final_interior_vertices': int((topo['vertex_type'] == 0).sum().item()), 'num_final_boundary_vertices': int((topo['vertex_type'] == 1).sum().item())})
        topo['diagnostics'] = diagnostics
        if edges.numel() == 0:
            edge_alpha = torch.empty((0,), dtype=seeds_uv.dtype, device=seeds_uv.device)
        else:
            edge_alpha = alpha[edges[:, 0]] * alpha[edges[:, 1]]
        vertex_degree = self.exact_vertex_degree(num_vertices=vertices_uv.shape[0], edge_index=edges, dtype=vertices_uv.dtype, device=vertices_uv.device)
        active_interior = topo['vertex_type'] == 0
        num_interior = int(active_interior.sum().item())
        num_boundary = int((topo['vertex_type'] == 1).sum().item())
        graph = {'nodes_uv': vertices_uv, 'node_alpha': alpha, 'node_type': topo['vertex_type'], 'node_degree': vertex_degree, 'edge_index': edges, 'edge_seed_pair': edge_seed_pairs, 'edge_alpha': edge_alpha, 'edge_type': edge_type, 'vertex_degree': vertex_degree, 'boundary_source_type': topo['boundary_source_type'], 'boundary_source_name': [{0: 'interior', 1: 'infinite_ray_clipping', 2: 'finite_edge_clipping', 3: 'pair_bisector_boundary', 4: 'corner_shell', 5: 'snapped_infinite_ray'}.get(int(value), 'unknown') for value in topo['boundary_source_type'].detach().cpu().tolist()], 'diagnostics': topo['diagnostics'], 'num_interior_nodes': num_interior, 'num_boundary_nodes': num_boundary}
        if 'edge_seed_pair' in graph:
            local_pairs = graph['edge_seed_pair']
            valid_pair_mask = local_pairs >= 0
            original_pairs = torch.full_like(local_pairs, -1)
            if bool(valid_pair_mask.any().detach().cpu().item()):
                original_pairs[valid_pair_mask] = seed_active_ids[local_pairs[valid_pair_mask]]
            graph['edge_seed_pair_original'] = original_pairs
        local_triples_for_original = topo['vertex_seed_triples']
        valid_triple_mask = local_triples_for_original >= 0
        vertex_seed_triples_original = torch.full_like(local_triples_for_original, -1)
        if bool(valid_triple_mask.any().detach().cpu().item()):
            vertex_seed_triples_original[valid_triple_mask] = seed_active_ids[
                local_triples_for_original[valid_triple_mask]
            ]
        min_edge_trim_samples = max(int(self.edge_trim_samples), 2)
        if edges.numel() > 0:
            trim_target_spacing = max(0.5 * min(float(self.tau_trim), float(self.vertex_boundary_margin)), self.eps)
            edge_trim_samples = self.adaptive_graph_curve_sample_count_uv(
                seeds_uv=seeds_uv,
                graph=graph,
                min_samples=min_edge_trim_samples,
                target_spacing=trim_target_spacing,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
            edge_curves_uv_for_trim = self.sample_graph_edge_curves_uv(seeds_uv=seeds_uv, graph=graph, n_samples=edge_trim_samples, u_periodic=u_periodic, v_periodic=v_periodic)
        else:
            edge_curves_uv_for_trim = vertices_uv.new_empty((0, min_edge_trim_samples, 2))
        edge_trim_alpha = self.edge_trim_gate(edge_curves_uv=edge_curves_uv_for_trim, cad_domain=cad_domain, reduction=self.edge_trim_reduction)
        graph['edge_alpha'] = graph['edge_alpha'] * edge_trim_alpha
        graph['edge_trim_alpha'] = edge_trim_alpha
        graph['edge_curves_uv_for_trim'] = edge_curves_uv_for_trim
        edge_alpha = graph['edge_alpha']
        out: dict[str, Any] = {'vertices_uv': vertices_uv, 'alpha': alpha, 'triple_idx': topo['vertex_seed_triples'], 'vertex_type': topo['vertex_type'], 'vertex_seed_triples': topo['vertex_seed_triples'], 'boundary_origin_vertex': topo['boundary_origin_vertex'], 'boundary_target_vertex': topo['boundary_target_vertex'], 'boundary_seed_pair': topo['boundary_seed_pair'], 'boundary_ray_dir': topo['boundary_ray_dir'], 'boundary_source_type': topo['boundary_source_type'], 'boundary_source_name': graph['boundary_source_name'], 'edges': {'edge_index': edges, 'edge_seed_pair': edge_seed_pairs, 'edge_alpha': edge_alpha, 'vertex_degree': vertex_degree, 'edge_type': edge_type, 'edge_trim_alpha': edge_trim_alpha}, 'boundary_rays': boundary_rays, 'boundary_ray_dirs': boundary_ray_dirs, 'scipy_vertices_np': topo['scipy_vertices_np'], 'pruned_vertices_uv': pruned_vertices_uv, 'pruned_vertex_type': pruned_vertex_type, 'isolated_vertices': torch.nonzero(vertex_degree == 0, as_tuple=False).flatten() if keep_isolated_vertices else torch.empty((0,), dtype=torch.long, device=seeds_uv.device), 'delaunay_triples_np': topo['delaunay_triples_np'], 'mode': 'scipy_topology', 'vertex_degree': vertex_degree, 'graph': graph, 'diagnostics': topo['diagnostics']}
        out['original_seeds_uv'] = original_seeds_uv
        out['active_seed_ids'] = seed_active_ids
        out['seed_active_mask'] = seed_active_mask
        out['seed_activity_weight'] = seed_activity_weight
        out['topology_seeds_uv'] = seeds_uv
        out['seed_activation_diagnostics'] = {
            'num_original_seeds': int(original_seeds_uv.shape[0]),
            'num_active_seeds': int(seed_active_ids.numel()),
            'num_removed_seeds': int(original_seeds_uv.shape[0] - seed_active_ids.numel()),
        }
        out['vertex_seed_triples_original'] = vertex_seed_triples_original
        out['edges']['edge_seed_pair_original'] = graph.get('edge_seed_pair_original')
        out.update(topo['diagnostics'])
        if edges.numel() > 0:
            edge_curve_samples = self.adaptive_graph_curve_sample_count_uv(
                seeds_uv=seeds_uv,
                graph=graph,
                min_samples=max(int(self.tube_curve_samples), 2),
                target_spacing=max(0.5 * min(float(self.centerline_softmin_tau), float(self.centerline_beta)), self.eps),
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
            out['edge_curves_uv'] = self.sample_graph_edge_curves_uv(seeds_uv=seeds_uv, graph=graph, n_samples=edge_curve_samples, u_periodic=u_periodic, v_periodic=v_periodic)
            if cad_domain is not None and want_xyz and callable(getattr(cad_domain, 'eval_uv_norm_batch_torch', None)):
                out['edge_curves_xyz'] = self.sample_smooth_edge_curves_xyz(cad_domain, out['edge_curves_uv'])
        if cad_domain is not None and want_xyz:
            xyz = cad_domain.eval_uv_norm_batch(vertices_uv, return_inside_mask=False)['xyz']
            out['vertices_xyz'] = torch.as_tensor(xyz, dtype=seeds_uv.dtype, device=seeds_uv.device)
        return out

    def evaluate_at_uv(
        self,
        points_uv: torch.Tensor,
        Xu: torch.Tensor | None=None,
        Xv: torch.Tensor | None=None,
        points_3d: torch.Tensor | None=None,
        points_xyz: torch.Tensor | None=None,
        tau: float | torch.Tensor | None=None,
        seeds_raw: torch.Tensor | None=None,
        w_raw: torch.Tensor | None=None,
        cad_domain: Any | None=None,
        u_periodic: bool | None=None,
        v_periodic: bool | None=None,
        **_: Any,
    ) -> dict[str, Any]:
        if seeds_raw is None:
            raise ValueError('seeds_raw must be provided for swept tube evaluation.')
        if w_raw is None:
            raise ValueError('w_raw must be provided for swept tube evaluation.')
        if points_3d is None:
            points_3d = points_xyz
        if points_3d is None:
            raise ValueError('points_3d must be provided for swept tube evaluation.')
        points_uv = torch.as_tensor(points_uv, dtype=seeds_raw.dtype, device=seeds_raw.device)
        points_3d = torch.as_tensor(points_3d, dtype=seeds_raw.dtype, device=seeds_raw.device)
        Xu_t = None if Xu is None else torch.as_tensor(Xu, dtype=seeds_raw.dtype, device=seeds_raw.device)
        Xv_t = None if Xv is None else torch.as_tensor(Xv, dtype=seeds_raw.dtype, device=seeds_raw.device)
        use_u_periodic = self._bool_value(self.face_u_periodic) if u_periodic is None else bool(u_periodic)
        use_v_periodic = self._bool_value(self.face_v_periodic) if v_periodic is None else bool(v_periodic)
        return self.build_swept_tube_fields(
            points_uv=points_uv,
            points_3d=points_3d,
            seeds_uv=seeds_raw,
            w_raw=w_raw,
            Xu=Xu_t,
            Xv=Xv_t,
            cad_domain=cad_domain,
            u_periodic=use_u_periodic,
            v_periodic=use_v_periodic,
            return_xyz=True,
        )

    def forward(
        self,
        seeds_uv: torch.Tensor | None=None,
        seed_activity: torch.Tensor | None=None,
        cad_domain: Any | None=None,
        u_periodic: bool | None=None,
        v_periodic: bool | None=None,
        return_xyz: bool | None=None,
        debug_compare_scipy: bool=False,
        keep_isolated_vertices: bool=False,
        points_uv: torch.Tensor | None=None,
        Xu: torch.Tensor | None=None,
        Xv: torch.Tensor | None=None,
        points_3d: torch.Tensor | None=None,
        points_xyz: torch.Tensor | None=None,
        tau: float | torch.Tensor | None=None,
        seeds_raw: torch.Tensor | None=None,
        w_raw: torch.Tensor | None=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if points_uv is not None or seeds_raw is not None or w_raw is not None:
            return self.evaluate_at_uv(
                points_uv=points_uv,
                Xu=Xu,
                Xv=Xv,
                points_3d=points_3d if points_3d is not None else points_xyz,
                tau=tau,
                seeds_raw=seeds_raw if seeds_raw is not None else seeds_uv,
                w_raw=w_raw,
                cad_domain=cad_domain,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
                **kwargs,
            )
        if not isinstance(seeds_uv, torch.Tensor):
            raise TypeError('seeds_uv must be a torch.Tensor.')
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError(f'seeds_uv must have shape [S, 2], got {tuple(seeds_uv.shape)}.')
        if not seeds_uv.is_floating_point():
            raise TypeError('seeds_uv must be a floating point tensor.')
        use_u_periodic = self._bool_value(self.face_u_periodic) if u_periodic is None else bool(u_periodic)
        use_v_periodic = self._bool_value(self.face_v_periodic) if v_periodic is None else bool(v_periodic)
        return self.forward_scipy_topology(seeds_uv=seeds_uv, cad_domain=cad_domain, u_periodic=use_u_periodic, v_periodic=use_v_periodic, return_xyz=return_xyz, keep_isolated_vertices=keep_isolated_vertices)

    def exact_vertex_degree(self, num_vertices, edge_index, dtype, device):
        degree = torch.zeros(num_vertices, dtype=dtype, device=device)
        if edge_index.numel() == 0:
            return degree
        one = torch.ones(edge_index.shape[0], dtype=dtype, device=device)
        degree = degree.scatter_add(0, edge_index[:, 0], one)
        degree = degree.scatter_add(0, edge_index[:, 1], one)
        return degree

    @staticmethod
    def _generated_graph_edge_style(edge_type: int) -> tuple[str, str, str]:
        styles = {0: ('black', '-', 'Interior Voronoi edge'), 1: ('tab:orange', '-', 'Clipped interior-boundary Voronoi edge'), 2: ('0.5', ':', 'Reserved edge type'), 3: ('tab:orange', '-', 'Clipped boundary-boundary Voronoi edge'), 4: ('tab:cyan', '--', 'Boundary shell edge')}
        return styles.get(int(edge_type), ('0.5', ':', f'Edge type {edge_type}'))

    def _draw_generated_graph(self, ax, seeds_uv, out, show_node_ids: bool=True, show_edge_ids: bool=False, node_id_fontsize: int=9, show_pruned_nodes: bool=False, color_by_edge_type: bool=True):
        """Draw only the compact graph represented by ``out['graph']``."""
        graph = out['graph']
        nodes = graph['nodes_uv'].detach().cpu().numpy()
        edges = graph['edge_index'].detach().cpu().numpy()
        node_types = graph['node_type'].detach().cpu().numpy()
        source_types = graph.get('boundary_source_type', graph['node_type'])
        source_types = source_types.detach().cpu().numpy()
        edge_types_t = graph.get('edge_type')
        if edge_types_t is None:
            edge_types = np.zeros((len(edges),), dtype=np.int64)
        else:
            edge_types = edge_types_t.detach().cpu().numpy()
        for edge_id, ((source, target), edge_type) in enumerate(zip(edges, edge_types)):
            source, target = (int(source), int(target))
            color, linestyle, _ = self._generated_graph_edge_style(int(edge_type)) if color_by_edge_type else ('black', '-', 'Graph edge')
            ax.plot([nodes[source, 0], nodes[target, 0]], [nodes[source, 1], nodes[target, 1]], color=color, linestyle=linestyle, linewidth=1.5, alpha=0.85, zorder=1)
            if show_edge_ids:
                midpoint = 0.5 * (nodes[source] + nodes[target])
                ax.text(midpoint[0], midpoint[1], f'e{edge_id}', fontsize=node_id_fontsize - 1, color=color, ha='center', va='center', zorder=5, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
        interior = node_types == 0
        coordinate_corners = (np.isclose(nodes[:, 0], 0.0, atol=0.0001) | np.isclose(nodes[:, 0], 1.0, atol=0.0001)) & (np.isclose(nodes[:, 1], 0.0, atol=0.0001) | np.isclose(nodes[:, 1], 1.0, atol=0.0001))
        corners = (node_types == 1) & ((source_types == 4) | coordinate_corners)
        boundary = (node_types == 1) & ~corners
        ax.scatter(nodes[interior, 0], nodes[interior, 1], c='orange', marker='o', edgecolors='black', s=80, label='Interior nodes', zorder=3)
        ax.scatter(nodes[boundary, 0], nodes[boundary, 1], c='tab:cyan', marker='D', edgecolors='black', s=70, label='Boundary nodes', zorder=3)
        ax.scatter(nodes[corners, 0], nodes[corners, 1], c='gold', marker='s', edgecolors='black', s=85, label='Corner shell nodes', zorder=3)
        seeds_np = seeds_uv.detach().cpu().numpy()
        ax.scatter(seeds_np[:, 0], seeds_np[:, 1], c='red', marker='o', s=45, label='Seeds', zorder=4)
        if show_node_ids:
            for node_id, (point, node_type) in enumerate(zip(nodes, node_types)):
                if int(node_type) == 0:
                    prefix = 'I'
                elif bool(corners[node_id]):
                    prefix = 'C'
                else:
                    prefix = 'B'
                ax.annotate(f'{prefix}{node_id}', xy=point, xytext=(5, 5), textcoords='offset points', fontsize=node_id_fontsize, color='black', zorder=6, bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.5))
        if show_pruned_nodes:
            pruned = out.get('pruned_vertices_uv')
            if pruned is not None and pruned.numel() > 0:
                pruned_np = pruned.detach().cpu().numpy()
                ax.scatter(pruned_np[:, 0], pruned_np[:, 1], marker='x', c='0.45', s=65, label='Pruned nodes', zorder=2)
        edge_handles = []
        for edge_type in range(5):
            color, linestyle, label = self._generated_graph_edge_style(edge_type)
            edge_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, label=label))
        handles, labels = ax.get_legend_handles_labels()
        num_nodes = len(nodes)
        num_interior = int(interior.sum())
        num_boundary = int((node_types == 1).sum())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Soft Geometry Graph \nnodes={num_nodes} (interior={num_interior}, boundary={num_boundary}), edges={len(edges)}\n')
        return ax

    @staticmethod
    def _print_generated_graph_tables(out) -> None:
        graph = out['graph']
        nodes = graph['nodes_uv'].detach().cpu().numpy()
        node_types = graph['node_type'].detach().cpu().numpy()
        degrees = graph['node_degree'].detach().cpu().numpy()
        alpha = graph['node_alpha'].detach().cpu().numpy()
        edges = graph['edge_index'].detach().cpu().numpy()
        edge_types = graph['edge_type'].detach().cpu().numpy()
        seed_pairs = graph['edge_seed_pair'].detach().cpu().numpy()
        print('\nNode table')
        print('node_id  node_type       u          v      degree      alpha')
        for node_id, (point, node_type, degree, activity) in enumerate(zip(nodes, node_types, degrees, alpha)):
            type_name = 'interior' if int(node_type) == 0 else 'boundary'
            print(f'{node_id:7d}  {type_name:9s}  {point[0]:9.6f}  {point[1]:9.6f}  {degree:8.3f}  {activity:9.6f}')
        print('\nEdge table')
        print('edge_id  source  target  edge_type  seed_pair')
        for edge_id, (edge, edge_type, seed_pair) in enumerate(zip(edges, edge_types, seed_pairs)):
            print(f'{edge_id:7d}  {int(edge[0]):6d}  {int(edge[1]):6d}  {int(edge_type):9d}  ({int(seed_pair[0])}, {int(seed_pair[1])})')

    def plot_graph_output(self, seeds_uv, out=None, cad_domain=None, show_node_ids: bool=True, show_edge_ids: bool=False, node_id_fontsize: int=9, print_node_table: bool=True, show_pruned_nodes: bool=False, color_by_edge_type: bool=True):
        if out is None:
            out = self(seeds_uv, cad_domain=cad_domain, return_xyz=False)
        fig, ax = plt.subplots(figsize=(8, 8))
        self._draw_generated_graph(ax, seeds_uv, out, show_node_ids, show_edge_ids, node_id_fontsize, show_pruned_nodes, color_by_edge_type)
        plt.show()
        if print_node_table:
            self._print_generated_graph_tables(out)
        return (fig, ax)

    def plot_generated_graph_debug(self, seeds_uv, out=None, cad_domain=None, show_node_ids: bool=True, show_edge_ids: bool=False, node_id_fontsize: int=9, print_node_table: bool=True, show_pruned_nodes: bool=False, color_by_edge_type: bool=True):
        """Plot the generated graph abstraction without a SciPy background."""
        return self.plot_graph_output(seeds_uv=seeds_uv, out=out, cad_domain=cad_domain, show_node_ids=show_node_ids, show_edge_ids=show_edge_ids, node_id_fontsize=node_id_fontsize, print_node_table=print_node_table, show_pruned_nodes=show_pruned_nodes, color_by_edge_type=color_by_edge_type)
    def plot_scipy_vs_generated_graph(
        self,
        seeds_uv,
        out=None,
        cad_domain=None,
        show_node_ids: bool = True,
        show_edge_ids: bool = False,
        node_id_fontsize: int = 9,
        print_node_table: bool = True,
        show_pruned_nodes: bool = False,
        color_by_edge_type: bool = True,
    ):
        if out is None:
            out = self(seeds_uv, cad_domain=cad_domain, return_xyz=False)

        original_seeds_uv = seeds_uv
        topology_seeds_uv = out.get("topology_seeds_uv", original_seeds_uv)

        original_np = original_seeds_uv.detach().cpu().numpy()
        topology_np = topology_seeds_uv.detach().cpu().numpy()

        # Robust active mask over original seeds.
        if "seed_active_mask" in out:
            active_mask = out["seed_active_mask"].detach().cpu().numpy().astype(bool)
        else:
            active_mask = np.ones((original_np.shape[0],), dtype=bool)

        # If mask is not aligned with original seeds, rebuild it from active_seed_ids.
        if active_mask.shape[0] != original_np.shape[0]:
            active_mask = np.zeros((original_np.shape[0],), dtype=bool)
            if "active_seed_ids" in out:
                active_ids = out["active_seed_ids"].detach().cpu().numpy().astype(int)
                active_ids = active_ids[(active_ids >= 0) & (active_ids < original_np.shape[0])]
                active_mask[active_ids] = True
            else:
                # Fallback: match topology seeds to original seeds by coordinate.
                for p in topology_np:
                    d = np.linalg.norm(original_np - p[None, :], axis=1)
                    active_mask[np.argmin(d)] = True

        active_np = original_np[active_mask]
        inactive_np = original_np[~active_mask]

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18, 8),
            constrained_layout=True,
        )
        left, middle = axes

        # Left: raw SciPy Voronoi from active/topology seeds only.
        try:
            if topology_np.shape[0] >= 3:
                raw_voronoi = Voronoi(topology_np)
                voronoi_plot_2d(
                    raw_voronoi,
                    ax=left,
                    show_vertices=False,
                    show_points=False,
                    line_colors="black",
                    line_width=1.0,
                    line_alpha=0.75,
                    point_size=0,
                )
                if raw_voronoi.vertices.size > 0:
                    left.scatter(
                        raw_voronoi.vertices[:, 0],
                        raw_voronoi.vertices[:, 1],
                        marker="x",
                        c="0.35",
                        s=55,
                        label="Raw SciPy vertices",
                        zorder=3,
                    )
            else:
                left.text(
                    0.5,
                    0.5,
                    f"SciPy Voronoi unavailable\nonly {topology_np.shape[0]} active seeds",
                    ha="center",
                    va="center",
                )
        except Exception as error:
            left.text(
                0.5,
                0.5,
                f"SciPy Voronoi unavailable\n{error}",
                ha="center",
                va="center",
            )

        # Show all original seeds on left.
        if active_np.shape[0] > 0:
            left.scatter(
                active_np[:, 0],
                active_np[:, 1],
                c="green",
                s=60,
                edgecolors="black",
                linewidths=0.6,
                label="Active seeds",
                zorder=6,
            )

        if inactive_np.shape[0] > 0:
            left.scatter(
                inactive_np[:, 0],
                inactive_np[:, 1],
                c="red",
                s=70,
                marker="x",
                linewidths=2.0,
                label="Inactive seeds",
                zorder=7,
            )

        left.set_xlim(0, 1)
        left.set_ylim(0, 1)
        left.set_aspect("equal")
        left.set_title(
            f"VD for {original_np.shape[0]} seeds "
            f"({topology_np.shape[0]} active)\n"
            "Raw SciPy Voronoi from active seeds"
        )
        left.legend()

        # Right: generated graph from active/topology seeds.
        self._draw_generated_graph(
            middle,
            topology_seeds_uv,
            out,
            show_node_ids,
            show_edge_ids,
            node_id_fontsize,
            show_pruned_nodes,
            color_by_edge_type,
        )

        # Overlay all original seeds on right.
        if active_np.shape[0] > 0:
            middle.scatter(
                active_np[:, 0],
                active_np[:, 1],
                c="green",
                s=50,
                edgecolors="black",
                linewidths=0.6,
                label="Active seeds",
                zorder=8,
            )

        if inactive_np.shape[0] > 0:
            middle.scatter(
                inactive_np[:, 0],
                inactive_np[:, 1],
                c="red",
                s=70,
                marker="x",
                linewidths=2.0,
                label="Inactive seeds",
                zorder=9,
            )

        middle.set_xlim(0, 1)
        middle.set_ylim(0, 1)
        middle.set_aspect("equal")
        middle.set_facecolor("none")
        middle.patch.set_alpha(0.0)

        handles, labels = middle.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            middle.legend(by_label.values(), by_label.keys())

        try:
            from IPython.display import display

            display(fig)
            plt.close(fig)
        except Exception:
            plt.show()

        if print_node_table:
            self._print_generated_graph_tables(out)

        return fig, axes
