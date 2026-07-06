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
    """
    Differentiable UV Voronoi decoder using one topology philosophy:

    SciPy/Qhull builds discrete topology from real seeds plus fixed guard seeds.
    Guard-related ridges are discarded, and real-real ridges are represented as
    finite segments clipped to the UV box or CAD trim curves. PyTorch then
    reconstructs those finite clipped segments differentiably. No infinite-ray
    boundary reconstruction is used.
    """

    def __init__(self,Cad_domain: any, face_mesh: torch.Tensor, eps: float=1e-08, solve_reg: float=1e-06, tau_voronoi: float=0.01, tau_box: float=0.01, tau_trim: float=0.01, use_trim_activity: bool=True, return_xyz: bool=True, vertex_boundary_margin: float=0.02, edge_trim_samples: int=32, edge_trim_reduction: str='softmin', edge_trim_reduce_tau: float=0.05, use_edge_trim_gate: bool=True, n_seeds: int | None=None, w_min: float=0.02, w_max_ratio: float=0.5, raw_temp: float=1.0, beta: float=0.02, centerline_softmin_tau: float=0.02, centerline_beta: float | None=None, tube_curve_samples: int=64, tube_lift_tau: float=0.02, tube_lift_max_values: int=4000000, tube_distance_tau: float | None=None, tube_density_tau: float | None=None, tube_fiber_tau: float | None=None, rho_min: float=0.0, face_u_periodic: Any=False, face_v_periodic: Any=False, nearest_segment_k: int=4, use_segment_distance: bool=True, use_spatial_pruning: bool=True, min_tube_spacing: float=1e-3, tube_target_spacing_ratio: float=0.75, use_seed_activation: bool=True, duplicate_merge_sigma: float=1e-4, duplicate_effect_temp_ratio: float=0.25, seed_domain_mask_threshold: float=0.5, min_active_seeds: int=3, **unused_kwargs: Any):
        super().__init__()
        self.Cad_domain = Cad_domain
        self.face_mesh = face_mesh
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
        self.tube_lift_max_values = max(1, int(tube_lift_max_values))
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
        self.use_guard_seeds = bool(unused_kwargs.pop("use_guard_seeds", True))
        self.guard_seed_margin = float(unused_kwargs.pop("guard_seed_margin", 1.0))
        self.guard_seed_per_side = int(unused_kwargs.pop("guard_seed_per_side", 5))
        self.guard_seed_max_attempts = int(unused_kwargs.pop("guard_seed_max_attempts", 4))
        self.guard_seed_expand_factor = float(unused_kwargs.pop("guard_seed_expand_factor", 2.0))
        self.strict_guard_topology = bool(unused_kwargs.pop("strict_guard_topology", False))
        self.clip_tol = float(unused_kwargs.pop("clip_tol", 1e-10))
        self.node_merge_tol = float(unused_kwargs.pop("node_merge_tol", 1e-8))
        self.points_uv = face_mesh["uv"]
        self.Xu = face_mesh["Xu"]
        self.Xv = face_mesh["Xv"]
        self.points_3d = face_mesh["points_xyz"]

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

                Type 4 follows CAD boundary polylines when they are available,
                otherwise it falls back to the UV-box boundary. Types 0, 1, and
                3 are differentiable straight Voronoi edge segments.

                Shell edges are retained because CAD/box boundary-loop sampling
                consumes them directly when building tube centerline curves.
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
                cad_curve = self.sample_cad_boundary_edge_uv(
                    nodes_uv[a],
                    nodes_uv[b],
                    graph=graph,
                    n_samples=n_samples,
                )
                if cad_curve is None:
                    cad_curve = self.sample_boundary_box_edge_uv(nodes_uv[a], nodes_uv[b], n_samples=n_samples)
                result_curves.append(cad_curve)
            else:
                result_curves.append(curves[edge_id])
        return torch.stack(result_curves, dim=0)

    def sample_cad_boundary_edge_uv(self, p0: torch.Tensor, p1: torch.Tensor, graph: dict[str, torch.Tensor], n_samples: int) -> torch.Tensor | None:
        """Sample a shell edge along packed CAD boundary polylines."""
        boundary_uv = graph.get('boundary_curve_uv')
        offsets = graph.get('boundary_curve_offsets')
        loop_id = graph.get('boundary_curve_loop_id')
        if boundary_uv is None or offsets is None:
            return None
        boundary_uv = torch.as_tensor(boundary_uv, dtype=p0.dtype, device=p0.device).reshape(-1, 2)
        offsets = torch.as_tensor(offsets, dtype=torch.long, device=p0.device).reshape(-1)
        if boundary_uv.numel() == 0 or offsets.numel() < 2:
            return None
        if loop_id is None:
            loop_id = torch.arange(offsets.numel() - 1, dtype=torch.long, device=p0.device)
        else:
            loop_id = torch.as_tensor(loop_id, dtype=torch.long, device=p0.device).reshape(-1)
        if loop_id.numel() != offsets.numel() - 1:
            return None

        p0_np = p0.detach().cpu().numpy()
        p1_np = p1.detach().cpu().numpy()
        uv_np = boundary_uv.detach().cpu().numpy()
        offsets_np = offsets.detach().cpu().numpy()
        loop_np = loop_id.detach().cpu().numpy()

        def project_point_to_polyline(point: np.ndarray, polyline: np.ndarray) -> tuple[float, float, np.ndarray]:
            starts = polyline[:-1]
            ends = polyline[1:]
            deltas = ends - starts
            lengths2 = np.sum(deltas * deltas, axis=1)
            valid = lengths2 > 1e-16
            if not np.any(valid):
                return (float('inf'), 0.0, polyline[0])
            starts_v = starts[valid]
            deltas_v = deltas[valid]
            lengths2_v = lengths2[valid]
            rel = point[None, :] - starts_v
            local_t = np.clip(np.sum(rel * deltas_v, axis=1) / lengths2_v, 0.0, 1.0)
            proj = starts_v + local_t[:, None] * deltas_v
            d2 = np.sum((proj - point[None, :]) ** 2, axis=1)
            best = int(np.argmin(d2))
            valid_ids = np.nonzero(valid)[0]
            seg_id = int(valid_ids[best])
            lengths = np.linalg.norm(deltas, axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
            s = float(cumulative[seg_id] + local_t[best] * lengths[seg_id])
            return (float(d2[best]), s, proj[best])

        best = None
        for loop_value in sorted(set(int(v) for v in loop_np.tolist())):
            piece_ids = np.nonzero(loop_np == loop_value)[0]
            if piece_ids.size == 0:
                continue
            parts = []
            for local_id, piece_id in enumerate(piece_ids.tolist()):
                start = int(offsets_np[piece_id])
                end = int(offsets_np[piece_id + 1])
                if end <= start:
                    continue
                pts = uv_np[start:end]
                parts.append(pts if local_id == 0 else pts[1:])
            if not parts:
                continue
            polyline = np.concatenate(parts, axis=0)
            if polyline.shape[0] < 2:
                continue
            closed = np.linalg.norm(polyline[0] - polyline[-1]) <= 1e-8
            if not closed:
                polyline = np.concatenate((polyline, polyline[:1]), axis=0)
            d0, s0, _ = project_point_to_polyline(p0_np, polyline)
            d1, s1, _ = project_point_to_polyline(p1_np, polyline)
            score = d0 + d1
            if best is None or score < best[0]:
                best = (score, polyline, s0, s1)
        if best is None:
            return None

        _, polyline, s0, s1 = best
        deltas = np.diff(polyline, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        total = float(np.sum(lengths))
        if total <= 1e-12:
            return None
        if s1 < s0:
            s1 += total
        if s1 - s0 > 0.5 * total:
            s0, s1 = s1, s0 + total
        sample_s = np.linspace(s0, s1, int(n_samples))
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        curve = []
        for value in sample_s:
            value_wrapped = value % total
            seg_id = int(np.searchsorted(cumulative, value_wrapped, side='right') - 1)
            seg_id = min(max(seg_id, 0), len(lengths) - 1)
            if lengths[seg_id] <= 1e-12:
                curve.append(polyline[seg_id])
                continue
            local_t = (value_wrapped - cumulative[seg_id]) / lengths[seg_id]
            curve.append(polyline[seg_id] + local_t * deltas[seg_id])
        return torch.as_tensor(np.asarray(curve), dtype=p0.dtype, device=p0.device)

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
        tau_t = flat_uv.new_tensor(self.tube_lift_tau if tau is None else float(tau)).clamp_min(self.eps)
        support_count = max(int(support_uv.shape[0]), 1)
        chunk_size = max(
            1,
            min(int(flat_uv.shape[0]), self.tube_lift_max_values // support_count),
        )
        xyz_chunks = []
        for start in range(0, int(flat_uv.shape[0]), chunk_size):
            end = min(start + chunk_size, int(flat_uv.shape[0]))
            diff = flat_uv[start:end, None, :] - support_uv[None, :, :]
            if u_periodic:
                diff_u = diff[..., 0] - torch.round(diff[..., 0])
                diff = torch.cat((diff_u.unsqueeze(-1), diff[..., 1:2]), dim=-1)
            if v_periodic:
                diff_v = diff[..., 1] - torch.round(diff[..., 1])
                diff = torch.cat((diff[..., 0:1], diff_v.unsqueeze(-1)), dim=-1)
            dist = torch.linalg.vector_norm(diff, dim=-1)
            weights = torch.softmax(-dist / tau_t, dim=1)
            xyz_chunks.append(weights @ support_xyz)
        xyz = torch.cat(xyz_chunks, dim=0) if xyz_chunks else support_xyz.new_empty((0, 3))
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

    @staticmethod
    def make_box_guard_seeds_np(
        margin: float = 0.25,
        per_side: int = 3,
        dtype=np.float64,
    ) -> np.ndarray:
        """
        Create fixed auxiliary seeds outside the unit square.

        These seeds are not part of the design variables. They only force Qhull
        to close the Voronoi diagram around the real seeds.
        """
        per_side = max(int(per_side), 2)
        margin = float(margin)
        t = np.linspace(0.0, 1.0, per_side, dtype=dtype)
        bottom = np.stack([t, np.full_like(t, -margin)], axis=1)
        top = np.stack([t, np.full_like(t, 1.0 + margin)], axis=1)
        left = np.stack([np.full_like(t, -margin), t], axis=1)
        right = np.stack([np.full_like(t, 1.0 + margin), t], axis=1)
        corners = np.array(
            [
                [-margin, -margin],
                [1.0 + margin, -margin],
                [1.0 + margin, 1.0 + margin],
                [-margin, 1.0 + margin],
            ],
            dtype=dtype,
        )
        guards = np.concatenate([bottom, right, top, left, corners], axis=0)
        guards = np.unique(np.round(guards, decimals=14), axis=0).astype(dtype, copy=False)
        return guards

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

    @staticmethod
    def segment_segment_intersection_np(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray, tol: float=1e-12) -> tuple[float, float, np.ndarray] | None:
        """Return segment parameters and point for a proper 2-D segment intersection."""
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)
        r = p1 - p0
        s = q1 - q0
        denom = float(r[0] * s[1] - r[1] * s[0])
        if abs(denom) <= tol:
            return None
        qp = q0 - p0
        t = float((qp[0] * s[1] - qp[1] * s[0]) / denom)
        u = float((qp[0] * r[1] - qp[1] * r[0]) / denom)
        if t < -tol or t > 1.0 + tol or u < -tol or u > 1.0 + tol:
            return None
        t = min(max(t, 0.0), 1.0)
        u = min(max(u, 0.0), 1.0)
        return (t, u, p0 + t * r)

    def build_scipy_voronoi_topology(self, seeds_uv: torch.Tensor, cad_domain: Any | None=None, u_periodic: bool=False, v_periodic: bool=False) -> dict[str, Any]:
        """
        Build a fixed discrete Voronoi graph with SciPy/Qhull.

        SciPy is used only for topology. Fixed guard seeds are appended outside
        [0, 1]^2 to make real-real ridges finite; guard-related ridges are
        discarded. Boundary nodes are produced by clipping those finite segments
        to the UV box or CAD trim curves. No infinite Voronoi ray is
        reconstructed as topology.
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

        def base_diagnostics() -> dict[str, int]:
            return {
                'num_guard_ridges_skipped': 0,
                'num_real_real_infinite_ridges_skipped': 0,
                'num_non_segment_ridges_skipped': 0,
                'num_ridges_outside_box_skipped': 0,
                'num_ridges_outside_domain_skipped': 0,
                'num_degenerate_edges_skipped': 0,
                'num_trim_intersections': 0,
                'num_trim_split_edges': 0,
                'num_final_nodes': 0,
                'num_final_edges': 0,
            }

        def empty_topology() -> dict[str, Any]:
            return {'vertices_uv': empty_float_2(), 'vertex_type': torch.empty((0,), dtype=torch.long, device=device), 'vertex_seed_triples': empty_long_3(), 'node_clip_source_vertices': empty_long_2(), 'node_trim_curve_piece': torch.empty((0,), dtype=torch.long, device=device), 'node_trim_curve_segment': torch.empty((0,), dtype=torch.long, device=device), 'node_trim_curve_fraction': torch.empty((0,), dtype=dtype, device=device), 'node_trim_segment_uv': torch.empty((0, 2, 2), dtype=dtype, device=device), 'scipy_vertex_aug_seed_triples': empty_long_3(), 'guard_seeds_uv': empty_float_2(), 'boundary_seed_pair': empty_long_2(), 'boundary_source_type': torch.empty((0,), dtype=torch.long, device=device), 'edges': empty_long_2(), 'edge_seed_pairs': empty_long_2(), 'edge_type': torch.empty((0,), dtype=torch.long, device=device), 'diagnostics': base_diagnostics()}
        num_real = int(points_np.shape[0])
        if num_real < 3:
            return empty_topology()
        #This function counts the number of infinite ridges between real seeds in the Voronoi diagram
        def count_real_real_infinite_ridges(vor_obj: Voronoi) -> int:
            count = 0
            for seed_pair, ridge_vertices in zip(vor_obj.ridge_points, vor_obj.ridge_vertices):
                seed_i = int(seed_pair[0])
                seed_j = int(seed_pair[1])
                if seed_i < num_real and seed_j < num_real and any(int(v) < 0 for v in ridge_vertices):
                    count += 1
            return count

        if self.use_guard_seeds:
            vor = None
            guard_np = np.empty((0, 2), dtype=points_np.dtype)
            points_for_voronoi_np = points_np
            max_attempts = max(int(self.guard_seed_max_attempts), 1)
            expand_factor = max(float(self.guard_seed_expand_factor), 1.0)
            for attempt in range(max_attempts):
                margin = float(self.guard_seed_margin) * (expand_factor ** attempt)
                # Generate guard seeds outside the unit square to help close the Voronoi diagram
                candidate_guard_np = self.make_box_guard_seeds_np(
                    margin=margin,
                    per_side=self.guard_seed_per_side,
                    dtype=points_np.dtype,
                )
                candidate_points_np = np.concatenate([points_np, candidate_guard_np], axis=0)
                try:
                    candidate_vor = Voronoi(candidate_points_np)
                except Exception:
                    continue
                vor = candidate_vor
                guard_np = candidate_guard_np
                points_for_voronoi_np = candidate_points_np
                if count_real_real_infinite_ridges(candidate_vor) == 0:
                    break
            if vor is None:
                return empty_topology()
        else:
            guard_np = np.empty((0, 2), dtype=points_np.dtype)
            points_for_voronoi_np = points_np
            try:
                vor = Voronoi(points_for_voronoi_np)
            except Exception:
                return empty_topology()
        scipy_vertices_np = vor.vertices
        num_raw_scipy_vertices = int(scipy_vertices_np.shape[0])
        diagnostics = base_diagnostics()
        vertex_seed_triples_by_scipy_id: list[list[int]] = []
        vertex_aug_seed_triples_by_scipy_id: list[list[int]] = []
        if num_raw_scipy_vertices > 0:
            tree = cKDTree(points_np)
            _, triple_idx_np = tree.query(scipy_vertices_np, k=min(3, num_real))
            triple_idx_np = np.asarray(triple_idx_np, dtype=np.int64)
            if triple_idx_np.ndim == 1:
                triple_idx_np = triple_idx_np.reshape(-1, 1)
            if triple_idx_np.shape[1] < 3:
                pad = np.full((triple_idx_np.shape[0], 3 - triple_idx_np.shape[1]), -1, dtype=np.int64)
                triple_idx_np = np.concatenate([triple_idx_np, pad], axis=1)
            vertex_seed_triples_by_scipy_id = triple_idx_np[:, :3].tolist()
            aug_tree = cKDTree(points_for_voronoi_np)
            _, aug_triple_idx_np = aug_tree.query(scipy_vertices_np, k=3)
            aug_triple_idx_np = np.asarray(aug_triple_idx_np, dtype=np.int64)
            if aug_triple_idx_np.ndim == 1:
                aug_triple_idx_np = aug_triple_idx_np.reshape(1, 3)
            vertex_aug_seed_triples_by_scipy_id = aug_triple_idx_np[:, :3].tolist()

        node_uv_list: list[np.ndarray] = []
        node_type_list: list[int] = []
        node_seed_triples_list: list[list[int]] = []
        node_clip_source_vertices_list: list[list[int]] = []
        node_trim_curve_piece_list: list[int] = []
        node_trim_curve_segment_list: list[int] = []
        node_trim_curve_fraction_list: list[float] = []
        node_trim_segment_uv_list: list[list[list[float]]] = []
        # boundary_seed_pair is retained as metadata; boundary node positions are reconstructed from finite clipped Voronoi segments.
        boundary_seed_pair_list: list[list[int]] = []
        # boundary_source_type is used only for shell/CAD bookkeeping.
        boundary_source_type_list: list[int] = []
        node_key_to_id: dict[tuple[int, int], int] = {}
        edges: list[list[int]] = []
        edge_seed_pairs: list[list[int]] = []
        edge_types: list[int] = []

        def as_numpy_maybe(value: Any) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        def cad_value(name: str) -> Any | None:
            if cad_domain is None:
                return None
            if isinstance(cad_domain, dict):
                return cad_domain.get(name)
            if hasattr(cad_domain, 'boundary_curve_tensors'):
                try:
                    boundary_data = cad_domain.boundary_curve_tensors(as_torch=True)
                    if isinstance(boundary_data, dict) and name in boundary_data:
                        return boundary_data[name]
                except Exception:
                    pass
            if hasattr(cad_domain, name):
                return getattr(cad_domain, name)
            private_name = f'_{name}'
            if hasattr(cad_domain, private_name):
                return getattr(cad_domain, private_name)
            return None

        boundary_curve_uv_np = cad_value('boundary_curve_uv')
        boundary_curve_offsets_np = cad_value('boundary_curve_offsets')
        if boundary_curve_uv_np is not None and boundary_curve_offsets_np is not None:
            boundary_curve_uv_np = as_numpy_maybe(boundary_curve_uv_np).astype(np.float64, copy=False)
            boundary_curve_offsets_np = as_numpy_maybe(boundary_curve_offsets_np).astype(np.int64, copy=False)
            if boundary_curve_uv_np.ndim != 2 or boundary_curve_uv_np.shape[-1] != 2 or boundary_curve_offsets_np.ndim != 1 or boundary_curve_offsets_np.size < 2:
                boundary_curve_uv_np = None
                boundary_curve_offsets_np = None
        trim_sdf_grid_np = cad_value('seed_domain_sdf_grid')
        if trim_sdf_grid_np is not None:
            trim_sdf_grid_np = as_numpy_maybe(trim_sdf_grid_np).astype(np.float64, copy=False)
            if trim_sdf_grid_np.ndim != 2:
                trim_sdf_grid_np = None
        has_trim_domain = boundary_curve_uv_np is not None and boundary_curve_offsets_np is not None and trim_sdf_grid_np is not None

        def sample_trim_sdf_np(p: np.ndarray) -> float:
            if trim_sdf_grid_np is None:
                return 1.0
            u = float(np.clip(p[0], 0.0, 1.0))
            v = float(np.clip(p[1], 0.0, 1.0))
            height, width = trim_sdf_grid_np.shape
            x = u * float(width - 1)
            y = v * float(height - 1)
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)
            sx = x - float(x0)
            sy = y - float(y0)
            v00 = trim_sdf_grid_np[y0, x0]
            v10 = trim_sdf_grid_np[y0, x1]
            v01 = trim_sdf_grid_np[y1, x0]
            v11 = trim_sdf_grid_np[y1, x1]
            return float((1.0 - sx) * (1.0 - sy) * v00 + sx * (1.0 - sy) * v10 + (1.0 - sx) * sy * v01 + sx * sy * v11)

        def trim_segment_intersections(q0: np.ndarray, q1: np.ndarray) -> list[dict[str, Any]]:
            if not has_trim_domain:
                return []
            hits: list[dict[str, Any]] = []
            for piece_id in range(int(boundary_curve_offsets_np.size - 1)):
                start = int(boundary_curve_offsets_np[piece_id])
                end = int(boundary_curve_offsets_np[piece_id + 1])
                points = boundary_curve_uv_np[start:end]
                if points.shape[0] < 2:
                    continue
                for local_seg_id in range(points.shape[0] - 1):
                    r0 = points[local_seg_id]
                    r1 = points[local_seg_id + 1]
                    hit = self.segment_segment_intersection_np(q0, q1, r0, r1, tol=max(self.clip_tol, 1e-12))
                    if hit is None:
                        continue
                    t, u, point = hit
                    if t <= self.clip_tol or t >= 1.0 - self.clip_tol:
                        continue
                    hits.append({
                        't': t,
                        'u': u,
                        'point': point.astype(points_np.dtype, copy=False),
                        'piece_id': piece_id,
                        'segment_id': start + local_seg_id,
                        'segment_uv': np.stack((r0, r1), axis=0).astype(points_np.dtype, copy=False),
                    })
            hits.sort(key=lambda item: item['t'])
            unique_hits: list[dict[str, Any]] = []
            for hit in hits:
                if unique_hits and abs(float(hit['t']) - float(unique_hits[-1]['t'])) <= max(self.clip_tol, self.node_merge_tol):
                    continue
                unique_hits.append(hit)
            return unique_hits

        def trim_inside_subsegments(q0: np.ndarray, q1: np.ndarray) -> list[dict[str, Any]]:
            if not has_trim_domain:
                return [{'q0': q0, 'q1': q1, 't0': 0.0, 't1': 1.0, 'hit0': None, 'hit1': None}]
            hits = trim_segment_intersections(q0, q1)
            diagnostics['num_trim_intersections'] += len(hits)
            cuts = [0.0] + [float(hit['t']) for hit in hits] + [1.0]
            delta = q1 - q0
            kept: list[dict[str, Any]] = []
            for index in range(len(cuts) - 1):
                t0 = cuts[index]
                t1 = cuts[index + 1]
                if t1 - t0 <= self.node_merge_tol:
                    continue
                midpoint = q0 + (0.5 * (t0 + t1)) * delta
                if sample_trim_sdf_np(midpoint) < -self.clip_tol:
                    continue
                kept.append({
                    'q0': (q0 + t0 * delta).astype(points_np.dtype, copy=False),
                    'q1': (q0 + t1 * delta).astype(points_np.dtype, copy=False),
                    't0': t0,
                    't1': t1,
                    'hit0': hits[index - 1] if index > 0 else None,
                    'hit1': hits[index] if index < len(hits) else None,
                })
            return kept

        def node_key_np(p: np.ndarray) -> tuple[int, int]:
            scale = 1.0 / max(float(self.node_merge_tol), 1e-12)
            return (int(round(float(p[0]) * scale)), int(round(float(p[1]) * scale)))

        def is_boundary_point_np(p: np.ndarray, tol: float=1e-8) -> bool:
            return (
                abs(float(p[0])) <= tol
                or abs(float(p[0]) - 1.0) <= tol
                or abs(float(p[1])) <= tol
                or abs(float(p[1]) - 1.0) <= tol
            )

        def add_node_from_uv(
            p: np.ndarray,
            seed_i: int,
            seed_j: int,
            scipy_vertex_id: int = -1,
            clip_source_vertices: tuple[int, int] = (-1, -1),
            source_type_if_boundary: int = 2,
            trim_hit: dict[str, Any] | None = None,
        ) -> int:
            p = np.asarray(p, dtype=points_np.dtype)
            p = np.clip(p, [0.0, 0.0], [1.0, 1.0])
            key = node_key_np(p)
            if key in node_key_to_id:
                return node_key_to_id[key]
            node_id = len(node_uv_list)
            node_key_to_id[key] = node_id
            node_uv_list.append(p)
            node_clip_source_vertices_list.append([int(clip_source_vertices[0]), int(clip_source_vertices[1])])
            if trim_hit is None:
                node_trim_curve_piece_list.append(-1)
                node_trim_curve_segment_list.append(-1)
                node_trim_curve_fraction_list.append(0.0)
                node_trim_segment_uv_list.append([[-1.0, -1.0], [-1.0, -1.0]])
            else:
                node_trim_curve_piece_list.append(int(trim_hit['piece_id']))
                node_trim_curve_segment_list.append(int(trim_hit['segment_id']))
                node_trim_curve_fraction_list.append(float(trim_hit['u']))
                node_trim_segment_uv_list.append(np.asarray(trim_hit['segment_uv'], dtype=points_np.dtype).tolist())
            if trim_hit is not None or is_boundary_point_np(p, tol=max(self.clip_tol, self.node_merge_tol)):
                node_type_list.append(1)
                node_seed_triples_list.append([seed_i, seed_j, -1])
                boundary_seed_pair_list.append([seed_i, seed_j])
                boundary_source_type_list.append(5 if trim_hit is not None else source_type_if_boundary)
            else:
                node_type_list.append(0)
                if 0 <= scipy_vertex_id < len(vertex_seed_triples_by_scipy_id):
                    triple = vertex_seed_triples_by_scipy_id[scipy_vertex_id]
                else:
                    triple = [seed_i, seed_j, -1]
                triple = [int(x) if 0 <= int(x) < num_real else -1 for x in triple]
                node_seed_triples_list.append(triple)
                boundary_seed_pair_list.append([-1, -1])
                boundary_source_type_list.append(0)
            return node_id

        for seed_pair, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
            seed_i = int(seed_pair[0]) # Index for the first seed in the ridge pair
            seed_j = int(seed_pair[1]) # Index for the second seed in the ridge pair
            if not (seed_i < num_real and seed_j < num_real):
                diagnostics['num_guard_ridges_skipped'] += 1
                continue
            if any(int(v) < 0 for v in ridge_vertices):
                diagnostics['num_real_real_infinite_ridges_skipped'] += 1
                continue
            finite_vertices = [int(v) for v in ridge_vertices if int(v) >= 0]
            if len(finite_vertices) != 2:
                diagnostics['num_non_segment_ridges_skipped'] += 1
                continue
            a, b = finite_vertices
            pa = scipy_vertices_np[a]
            pb = scipy_vertices_np[b]
            clipped = self.segment_box_clip_np(
                pa,
                pb,
                bounds=(0.0, 1.0, 0.0, 1.0),
                tol=self.clip_tol,
            )
            if clipped is None:
                diagnostics['num_ridges_outside_box_skipped'] += 1
                diagnostics['num_ridges_outside_domain_skipped'] += 1
                continue
            q0, q1, t_enter, t_exit = clipped
            if np.linalg.norm(q1 - q0) <= self.node_merge_tol:
                diagnostics['num_degenerate_edges_skipped'] += 1
                continue
            subsegments = trim_inside_subsegments(q0, q1)
            if len(subsegments) > 1:
                diagnostics['num_trim_split_edges'] += len(subsegments) - 1
            raw_span = float(t_exit - t_enter)

            def endpoint_source(local_t: float, hit: dict[str, Any] | None) -> tuple[int, tuple[int, int], dict[str, Any] | None]:
                if hit is not None:
                    return (-1, (a, b), hit)
                raw_t = float(t_enter + local_t * raw_span)
                if abs(raw_t) <= self.clip_tol:
                    return (a, (a, -1), None)
                if abs(raw_t - 1.0) <= self.clip_tol:
                    return (b, (b, -1), None)
                return (-1, (a, b), None)

            for subsegment in subsegments:
                q_sub0 = subsegment['q0']
                q_sub1 = subsegment['q1']
                if np.linalg.norm(q_sub1 - q_sub0) <= self.node_merge_tol:
                    diagnostics['num_degenerate_edges_skipped'] += 1
                    continue
                scipy0, source0, trim_hit0 = endpoint_source(float(subsegment['t0']), subsegment['hit0'])
                scipy1, source1, trim_hit1 = endpoint_source(float(subsegment['t1']), subsegment['hit1'])
                id0 = add_node_from_uv(
                    q_sub0,
                    seed_i,
                    seed_j,
                    scipy_vertex_id=scipy0,
                    clip_source_vertices=source0,
                    source_type_if_boundary=2,
                    trim_hit=trim_hit0,
                )
                id1 = add_node_from_uv(
                    q_sub1,
                    seed_i,
                    seed_j,
                    scipy_vertex_id=scipy1,
                    clip_source_vertices=source1,
                    source_type_if_boundary=2,
                    trim_hit=trim_hit1,
                )
                if id0 == id1:
                    diagnostics['num_degenerate_edges_skipped'] += 1
                    continue
                edges.append([id0, id1])
                edge_seed_pairs.append([seed_i, seed_j])
                if node_type_list[id0] == 0 and node_type_list[id1] == 0:
                    edge_types.append(0)
                elif node_type_list[id0] != node_type_list[id1]:
                    edge_types.append(1)
                else:
                    edge_types.append(3)

        vertices_uv = torch.as_tensor(
            np.asarray(node_uv_list, dtype=points_np.dtype).reshape(-1, 2),
            dtype=dtype,
            device=device,
        )
        vertex_type_t = torch.as_tensor(node_type_list, dtype=torch.long, device=device)
        vertex_seed_triples_t = torch.as_tensor(
            node_seed_triples_list,
            dtype=torch.long,
            device=device,
        ).reshape(-1, 3)
        boundary_seed_pair_t = torch.as_tensor(
            boundary_seed_pair_list,
            dtype=torch.long,
            device=device,
        ).reshape(-1, 2)
        boundary_source_type_t = torch.as_tensor(boundary_source_type_list, dtype=torch.long, device=device)
        node_clip_source_vertices_t = torch.as_tensor(
            node_clip_source_vertices_list,
            dtype=torch.long,
            device=device,
        ).reshape(-1, 2)
        node_trim_curve_piece_t = torch.as_tensor(node_trim_curve_piece_list, dtype=torch.long, device=device)
        node_trim_curve_segment_t = torch.as_tensor(node_trim_curve_segment_list, dtype=torch.long, device=device)
        node_trim_curve_fraction_t = torch.as_tensor(node_trim_curve_fraction_list, dtype=dtype, device=device)
        node_trim_segment_uv_t = torch.as_tensor(node_trim_segment_uv_list, dtype=dtype, device=device).reshape(-1, 2, 2)
        scipy_vertex_aug_seed_triples_t = torch.as_tensor(
            vertex_aug_seed_triples_by_scipy_id,
            dtype=torch.long,
            device=device,
        ).reshape(-1, 3)
        guard_seeds_uv_t = torch.as_tensor(guard_np, dtype=dtype, device=device).reshape(-1, 2)
        edges_t = torch.as_tensor(edges, dtype=torch.long, device=device).reshape(-1, 2)
        edge_seed_pairs_t = torch.as_tensor(edge_seed_pairs, dtype=torch.long, device=device).reshape(-1, 2)
        edge_type_t = torch.as_tensor(edge_types, dtype=torch.long, device=device)
        if self.strict_guard_topology and diagnostics['num_real_real_infinite_ridges_skipped'] > 0:
            raise RuntimeError("Guard seeds failed to close all real-real ridges.")
        return {'vertices_uv': vertices_uv, 'vertex_type': vertex_type_t, 'vertex_seed_triples': vertex_seed_triples_t, 'scipy_vertex_aug_seed_triples': scipy_vertex_aug_seed_triples_t, 'guard_seeds_uv': guard_seeds_uv_t, 'node_clip_source_vertices': node_clip_source_vertices_t, 'node_trim_segment_uv': node_trim_segment_uv_t, 'node_trim_curve_piece': node_trim_curve_piece_t, 'node_trim_curve_segment': node_trim_curve_segment_t, 'node_trim_curve_fraction': node_trim_curve_fraction_t, 'boundary_seed_pair': boundary_seed_pair_t, 'boundary_source_type': boundary_source_type_t, 'edges': edges_t, 'edge_seed_pairs': edge_seed_pairs_t, 'edge_type': edge_type_t, 'diagnostics': diagnostics}

    def prune_graph_vertices(self, nodes_uv: torch.Tensor, vertex_type: torch.Tensor, vertex_seed_triples: torch.Tensor, boundary_seed_pair: torch.Tensor, boundary_source_type: torch.Tensor, edges: torch.Tensor, edge_seed_pairs: torch.Tensor, edge_type: torch.Tensor, alpha: torch.Tensor | None=None, keep_isolated_vertices: bool=False) -> dict[str, torch.Tensor | int | None]:
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

        compact_type = vertex_type[active_ids]
        return {'nodes_uv': nodes_uv[active_ids], 'vertex_type': compact_type, 'vertex_seed_triples': vertex_seed_triples[active_ids], 'boundary_seed_pair': boundary_seed_pair[active_ids], 'boundary_source_type': boundary_source_type[active_ids], 'edges': compact_edges, 'edge_seed_pairs': edge_seed_pairs, 'edge_type': edge_type, 'alpha': None if alpha is None else alpha[active_ids], 'old_to_new': old_to_new, 'active_vertex_ids': active_ids}

    def differentiable_vertices_from_topology(self, seeds_uv: torch.Tensor, vertex_type: torch.Tensor, vertex_seed_triples: torch.Tensor, u_periodic: bool=False, v_periodic: bool=False, cad_domain: Any | None=None, boundary_source_type: torch.Tensor | None=None, topology_vertices_uv: torch.Tensor | None=None, node_clip_source_vertices: torch.Tensor | None=None, scipy_vertex_aug_seed_triples: torch.Tensor | None=None, guard_seeds_uv: torch.Tensor | None=None, node_trim_segment_uv: torch.Tensor | None=None) -> torch.Tensor:
        """
        Reconstruct topology nodes differentiably from finite clipped ridges.

        Requires ``node_clip_source_vertices``, ``scipy_vertex_aug_seed_triples``,
        ``guard_seeds_uv``, and stored ``topology_vertices_uv`` from the finite
        guard-seed topology. Raw SciPy vertices are recomputed from augmented
        seed triples. Guard seeds are detached constants, so gradients flow only
        through real seed coordinates. Boundary nodes are intersections on
        finite Voronoi segments; infinite ray directions are not used.
        """
        num_vertices = vertex_type.shape[0]
        if num_vertices == 0:
            return torch.empty((0, 2), dtype=seeds_uv.dtype, device=seeds_uv.device)
        required = (
            topology_vertices_uv is not None
            and node_clip_source_vertices is not None
            and scipy_vertex_aug_seed_triples is not None
            and guard_seeds_uv is not None
            and scipy_vertex_aug_seed_triples.numel() > 0
        )
        if not required:
            raise ValueError(
                "finite-segment topology reconstruction requires topology_vertices_uv, "
                "node_clip_source_vertices, scipy_vertex_aug_seed_triples, and guard_seeds_uv."
            )

        zero_node = torch.zeros((2,), dtype=seeds_uv.dtype, device=seeds_uv.device)
        node_values = [zero_node for _ in range(num_vertices)]
        if boundary_source_type is None:
            boundary_source_type = vertex_type
        topology_vertices_uv = topology_vertices_uv.to(dtype=seeds_uv.dtype, device=seeds_uv.device)
        guard_seeds_uv = guard_seeds_uv.to(dtype=seeds_uv.dtype, device=seeds_uv.device).detach()
        augmented_seeds_uv = torch.cat((seeds_uv, guard_seeds_uv), dim=0)
        raw_voronoi_vertices = self.differentiable_vertices_from_triples(
            augmented_seeds_uv,
            scipy_vertex_aug_seed_triples.to(device=seeds_uv.device),
            u_periodic=False,
            v_periodic=False,
        )
        node_clip_source_vertices = node_clip_source_vertices.to(device=seeds_uv.device)
        if node_trim_segment_uv is not None:
            node_trim_segment_uv = node_trim_segment_uv.to(dtype=seeds_uv.dtype, device=seeds_uv.device)

        def line_line_point(p0: torch.Tensor, p1: torch.Tensor, q0: torch.Tensor, q1: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
            r = p1 - p0
            s = q1 - q0
            denom = r[0] * s[1] - r[1] * s[0]
            if not bool((denom.abs() > self.eps).detach().cpu().item()):
                return fallback
            qp = q0 - p0
            t = (qp[0] * s[1] - qp[1] * s[0]) / denom
            return (p0 + t * r).clamp(0.0, 1.0)

        def clipped_source_point(source_vertices: torch.Tensor, stored_point: torch.Tensor, source_type: int, vertex_id: int) -> torch.Tensor:
            a = int(source_vertices[0].item())
            b = int(source_vertices[1].item())
            if not (0 <= a < raw_voronoi_vertices.shape[0]):
                raise RuntimeError("Finite-segment topology node is missing a valid raw SciPy source vertex.")
            p0 = raw_voronoi_vertices[a]
            if b < 0 or b >= raw_voronoi_vertices.shape[0]:
                return p0.clamp(0.0, 1.0)
            p1 = raw_voronoi_vertices[b]
            d = p1 - p0
            if source_type == 5 and node_trim_segment_uv is not None:
                trim_seg = node_trim_segment_uv[vertex_id]
                if bool((trim_seg >= 0.0).all().detach().cpu().item()):
                    return line_line_point(p0, p1, trim_seg[0], trim_seg[1], stored_point)
            tol = max(float(self.clip_tol), float(self.node_merge_tol), self.eps)
            x0 = stored_point[0]
            y0 = stored_point[1]
            on_left = abs(float(x0.detach().cpu().item())) <= tol
            on_right = abs(float(x0.detach().cpu().item()) - 1.0) <= tol
            on_bottom = abs(float(y0.detach().cpu().item())) <= tol
            on_top = abs(float(y0.detach().cpu().item()) - 1.0) <= tol
            if (on_left or on_right) and bool((d[0].abs() > self.eps).detach().cpu().item()):
                x = torch.zeros((), dtype=seeds_uv.dtype, device=seeds_uv.device) if on_left else torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device)
                t = (x - p0[0]) / d[0]
                y = p0[1] + t * d[1]
                return torch.stack((x, y.clamp(0.0, 1.0)))
            if (on_bottom or on_top) and bool((d[1].abs() > self.eps).detach().cpu().item()):
                y = torch.zeros((), dtype=seeds_uv.dtype, device=seeds_uv.device) if on_bottom else torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device)
                t = (y - p0[1]) / d[1]
                x = p0[0] + t * d[0]
                return torch.stack((x.clamp(0.0, 1.0), y))
            return stored_point

        for vertex_id in range(num_vertices):
            source_type = int(boundary_source_type[vertex_id].item())
            source_vertices = node_clip_source_vertices[vertex_id]
            if not bool((source_vertices >= 0).any().detach().cpu().item()):
                if source_type in (4, 6):
                    node_values[vertex_id] = topology_vertices_uv[vertex_id]
                    continue
                raise RuntimeError("Finite-segment topology node is missing clip-source metadata.")
            node_values[vertex_id] = clipped_source_point(source_vertices, topology_vertices_uv[vertex_id], source_type, vertex_id)
        return torch.stack(node_values, dim=0)

    def add_box_shell_corners(self, nodes_uv: torch.Tensor, node_alpha: torch.Tensor, node_type: torch.Tensor, node_seed_triples: torch.Tensor, boundary_seed_pair: torch.Tensor, boundary_source_type: torch.Tensor, tol: float=0.0001) -> dict[str, torch.Tensor]:
        """Append missing UV-box corners used by retained shell edges (source type 4)."""
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

    def add_cad_boundary_curve_endpoints(self, nodes_uv: torch.Tensor, node_alpha: torch.Tensor, node_type: torch.Tensor, node_seed_triples: torch.Tensor, boundary_seed_pair: torch.Tensor, boundary_source_type: torch.Tensor, cad_domain: Any | None=None, tol: float=1e-06) -> dict[str, torch.Tensor]:
        """Append CAD boundary C1-piece endpoints used by retained shell edges (source type 6)."""
        if cad_domain is None:
            return {'nodes_uv': nodes_uv, 'node_alpha': node_alpha, 'node_type': node_type, 'node_seed_triples': node_seed_triples, 'boundary_seed_pair': boundary_seed_pair, 'boundary_source_type': boundary_source_type}
        boundary_uv = None
        boundary_offsets = None
        if isinstance(cad_domain, dict):
            boundary_uv = cad_domain.get('boundary_curve_uv')
            boundary_offsets = cad_domain.get('boundary_curve_offsets')
        elif hasattr(cad_domain, 'boundary_curve_tensors'):
            try:
                data = cad_domain.boundary_curve_tensors(as_torch=True)
                boundary_uv = data.get('boundary_curve_uv') if isinstance(data, dict) else None
                boundary_offsets = data.get('boundary_curve_offsets') if isinstance(data, dict) else None
            except Exception:
                boundary_uv = None
                boundary_offsets = None
        if boundary_uv is None or boundary_offsets is None:
            return {'nodes_uv': nodes_uv, 'node_alpha': node_alpha, 'node_type': node_type, 'node_seed_triples': node_seed_triples, 'boundary_seed_pair': boundary_seed_pair, 'boundary_source_type': boundary_source_type}
        boundary_uv = torch.as_tensor(boundary_uv, dtype=nodes_uv.dtype, device=nodes_uv.device).reshape(-1, 2)
        boundary_offsets = torch.as_tensor(boundary_offsets, dtype=torch.long, device=nodes_uv.device).reshape(-1)
        if boundary_uv.numel() == 0 or boundary_offsets.numel() < 2:
            return {'nodes_uv': nodes_uv, 'node_alpha': node_alpha, 'node_type': node_type, 'node_seed_triples': node_seed_triples, 'boundary_seed_pair': boundary_seed_pair, 'boundary_source_type': boundary_source_type}
        endpoint_ids = []
        for piece_id in range(int(boundary_offsets.numel() - 1)):
            start = int(boundary_offsets[piece_id].item())
            end = int(boundary_offsets[piece_id + 1].item())
            if end <= start:
                continue
            endpoint_ids.append(start)
            endpoint_ids.append(end - 1)
        if not endpoint_ids:
            return {'nodes_uv': nodes_uv, 'node_alpha': node_alpha, 'node_type': node_type, 'node_seed_triples': node_seed_triples, 'boundary_seed_pair': boundary_seed_pair, 'boundary_source_type': boundary_source_type}
        boundary_endpoints = boundary_uv[torch.as_tensor(endpoint_ids, dtype=torch.long, device=nodes_uv.device)]
        keep = []
        existing = nodes_uv.detach()
        tol_t = torch.as_tensor(float(tol), dtype=nodes_uv.dtype, device=nodes_uv.device)
        for point in boundary_endpoints:
            if existing.numel() > 0 and bool((torch.linalg.vector_norm(existing - point, dim=1) <= tol_t).any().detach().cpu().item()):
                continue
            keep.append(point)
            existing = torch.cat((existing, point.reshape(1, 2).detach()), dim=0)
        if not keep:
            return {'nodes_uv': nodes_uv, 'node_alpha': node_alpha, 'node_type': node_type, 'node_seed_triples': node_seed_triples, 'boundary_seed_pair': boundary_seed_pair, 'boundary_source_type': boundary_source_type}
        added = torch.stack(keep, dim=0)
        count = int(added.shape[0])
        return {'nodes_uv': torch.cat((nodes_uv, added), dim=0), 'node_alpha': torch.cat((node_alpha, torch.ones((count,), dtype=nodes_uv.dtype, device=nodes_uv.device))), 'node_type': torch.cat((node_type, torch.ones((count,), dtype=torch.long, device=nodes_uv.device))), 'node_seed_triples': torch.cat((node_seed_triples, torch.full((count, 3), -1, dtype=torch.long, device=nodes_uv.device))), 'boundary_seed_pair': torch.cat((boundary_seed_pair, torch.full((count, 2), -1, dtype=torch.long, device=nodes_uv.device))), 'boundary_source_type': torch.cat((boundary_source_type, torch.full((count,), 6, dtype=torch.long, device=nodes_uv.device)))}

    def build_boundary_loop_edges(self, nodes_uv: torch.Tensor, vertex_type: torch.Tensor, cad_domain: Any | None=None, tol: float=0.0001) -> tuple[torch.Tensor, torch.Tensor]:
        """Connect boundary nodes cyclically as shell edges for CAD/box curve sampling."""
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
                - Guard-seed topology stores clipped UV node coordinates directly.
                - Gradients flow through downstream edge sampling and gates, not
                  through the detached SciPy/Qhull topology construction.
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
            try:
                delaunay_triples_np = Delaunay(seeds_uv.detach().cpu().numpy()).simplices
            except Exception:
                delaunay_triples_np = np.empty((0, 3), dtype=np.int64)
        vertices_uv = self.differentiable_vertices_from_topology(seeds_uv=seeds_uv, vertex_type=topo['vertex_type'], vertex_seed_triples=topo['vertex_seed_triples'], u_periodic=u_periodic, v_periodic=v_periodic, cad_domain=cad_domain, boundary_source_type=topo['boundary_source_type'], topology_vertices_uv=topo.get('vertices_uv'), node_clip_source_vertices=topo.get('node_clip_source_vertices'), scipy_vertex_aug_seed_triples=topo.get('scipy_vertex_aug_seed_triples'), guard_seeds_uv=topo.get('guard_seeds_uv'), node_trim_segment_uv=topo.get('node_trim_segment_uv'))
        # node_alpha is a soft validity/activity weight for edge/tube activity;
        # it is not used for topology trimming.
        alpha = torch.ones((vertices_uv.shape[0],), dtype=seeds_uv.dtype, device=seeds_uv.device)
        if cad_domain is not None and self.use_trim_activity:
            alpha = alpha * self.trim_gate(vertices_uv, cad_domain)
        else:
            alpha = alpha * self.box_gate(vertices_uv, u_periodic, v_periodic)
        if cad_domain is None:
            boundary_mask = topo['vertex_type'] == 1
            alpha = torch.where(boundary_mask, torch.ones_like(alpha), alpha)
        augmented = {
    "nodes_uv": vertices_uv,
    "node_alpha": alpha,
    "node_type": topo["vertex_type"],
    "node_seed_triples": topo["vertex_seed_triples"],
    "boundary_seed_pair": topo["boundary_seed_pair"],
    "boundary_source_type": topo["boundary_source_type"],
    "edges": topo["edges"],
    "edge_seed_pairs": topo["edge_seed_pairs"],
    "edge_type": topo["edge_type"],
}
        if cad_domain is not None:
            augmented.update(self.add_cad_boundary_curve_endpoints(nodes_uv=augmented['nodes_uv'], node_alpha=augmented['node_alpha'], node_type=augmented['node_type'], node_seed_triples=augmented['node_seed_triples'], boundary_seed_pair=augmented['boundary_seed_pair'], boundary_source_type=augmented['boundary_source_type'], cad_domain=cad_domain))
        else:
            augmented.update(self.add_box_shell_corners(nodes_uv=augmented['nodes_uv'], node_alpha=augmented['node_alpha'], node_type=augmented['node_type'], node_seed_triples=augmented['node_seed_triples'], boundary_seed_pair=augmented['boundary_seed_pair'], boundary_source_type=augmented['boundary_source_type']))
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
            if 'node_clip_source_vertices' in topo:
                topo['node_clip_source_vertices'] = torch.cat((topo['node_clip_source_vertices'], torch.full((total_added, 2), -1, dtype=torch.long, device=seeds_uv.device)))
            if 'node_trim_curve_piece' in topo:
                topo['node_trim_curve_piece'] = torch.cat((topo['node_trim_curve_piece'], torch.full((total_added,), -1, dtype=torch.long, device=seeds_uv.device)))
            if 'node_trim_curve_segment' in topo:
                topo['node_trim_curve_segment'] = torch.cat((topo['node_trim_curve_segment'], torch.full((total_added,), -1, dtype=torch.long, device=seeds_uv.device)))
            if 'node_trim_curve_fraction' in topo:
                topo['node_trim_curve_fraction'] = torch.cat((topo['node_trim_curve_fraction'], torch.zeros((total_added,), dtype=seeds_uv.dtype, device=seeds_uv.device)))
            if 'node_trim_segment_uv' in topo:
                topo['node_trim_segment_uv'] = torch.cat((topo['node_trim_segment_uv'], torch.full((total_added, 2, 2), -1.0, dtype=seeds_uv.dtype, device=seeds_uv.device)))
        loop_edges, loop_edge_type = self.build_boundary_loop_edges(vertices_uv, topo['vertex_type'], cad_domain=cad_domain)
        base_edges = topo['edges']
        edges = torch.cat((base_edges, loop_edges), dim=0)
        edge_type = torch.cat((topo['edge_type'], loop_edge_type), dim=0)
        loop_seed_pairs = torch.full((loop_edges.shape[0], 2), -1, dtype=torch.long, device=seeds_uv.device)
        edge_seed_pairs = torch.cat((topo['edge_seed_pairs'], loop_seed_pairs), dim=0)
        pruned = self.prune_graph_vertices(nodes_uv=vertices_uv, vertex_type=topo['vertex_type'], vertex_seed_triples=topo['vertex_seed_triples'], boundary_seed_pair=topo['boundary_seed_pair'], boundary_source_type=topo['boundary_source_type'], edges=edges, edge_seed_pairs=edge_seed_pairs, edge_type=edge_type, alpha=alpha, keep_isolated_vertices=keep_isolated_vertices)
        vertices_uv = pruned['nodes_uv']
        alpha = pruned['alpha']
        edges = pruned['edges']
        edge_seed_pairs = pruned['edge_seed_pairs']
        edge_type = pruned['edge_type']
        for key in ('vertex_type', 'vertex_seed_triples', 'boundary_seed_pair', 'boundary_source_type'):
            topo[key] = pruned[key]
        old_to_new = pruned['old_to_new']
        for key in ('node_clip_source_vertices', 'node_trim_curve_piece', 'node_trim_curve_segment', 'node_trim_curve_fraction', 'node_trim_segment_uv'):
            if key in topo and topo[key].shape[0] == old_to_new.shape[0]:
                topo[key] = topo[key][pruned['active_vertex_ids']]
        diagnostics = dict(topo['diagnostics'])
        diagnostics.update({'num_final_nodes': int(vertices_uv.shape[0]), 'num_final_edges': int(edges.shape[0])})
        topo['diagnostics'] = diagnostics
        if edges.numel() == 0:
            edge_alpha = torch.empty((0,), dtype=seeds_uv.dtype, device=seeds_uv.device)
        else:
            edge_alpha = alpha[edges[:, 0]] * alpha[edges[:, 1]]
        vertex_degree = self.exact_vertex_degree(num_vertices=vertices_uv.shape[0], edge_index=edges, dtype=vertices_uv.dtype, device=vertices_uv.device)
        active_interior = topo['vertex_type'] == 0
        num_interior = int(active_interior.sum().item())
        num_boundary = int((topo['vertex_type'] == 1).sum().item())
        graph = {'nodes_uv': vertices_uv, 'node_alpha': alpha, 'node_type': topo['vertex_type'], 'node_degree': vertex_degree, 'edge_index': edges, 'edge_seed_pair': edge_seed_pairs, 'edge_alpha': edge_alpha, 'edge_type': edge_type, 'vertex_degree': vertex_degree, 'boundary_source_type': topo['boundary_source_type'], 'boundary_source_name': [{0: 'interior', 1: 'finite_box_boundary', 2: 'finite_edge_clipping', 3: 'finite_boundary_boundary', 4: 'corner_shell', 5: 'trim_boundary_intersection', 6: 'cad_boundary_curve_sample'}.get(int(value), 'unknown') for value in topo['boundary_source_type'].detach().cpu().tolist()], 'diagnostics': topo['diagnostics'], 'num_interior_nodes': num_interior, 'num_boundary_nodes': num_boundary}
        for key in ('node_trim_curve_piece', 'node_trim_curve_segment', 'node_trim_curve_fraction', 'node_trim_segment_uv'):
            if key in topo:
                graph[key] = topo[key]
        if cad_domain is not None:
            boundary_data = cad_domain if isinstance(cad_domain, dict) else None
            if boundary_data is None and hasattr(cad_domain, 'boundary_curve_tensors'):
                try:
                    boundary_data = cad_domain.boundary_curve_tensors(as_torch=True)
                except Exception:
                    boundary_data = None
            if isinstance(boundary_data, dict):
                for key in ('boundary_curve_uv', 'boundary_curve_offsets', 'boundary_curve_loop_id'):
                    if key in boundary_data:
                        value = boundary_data[key]
                        if isinstance(value, torch.Tensor):
                            graph[key] = value.to(device=vertices_uv.device, dtype=vertices_uv.dtype if key == 'boundary_curve_uv' else torch.long)
                        else:
                            graph[key] = torch.as_tensor(value, device=vertices_uv.device, dtype=vertices_uv.dtype if key == 'boundary_curve_uv' else torch.long)
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
        out: dict[str, Any] = {'vertices_uv': vertices_uv, 'alpha': alpha, 'vertex_type': topo['vertex_type'], 'vertex_seed_triples': topo['vertex_seed_triples'], 'boundary_seed_pair': topo['boundary_seed_pair'], 'boundary_source_type': topo['boundary_source_type'], 'boundary_source_name': graph['boundary_source_name'], 'edges': {'edge_index': edges, 'edge_seed_pair': edge_seed_pairs, 'edge_alpha': edge_alpha, 'vertex_degree': vertex_degree, 'edge_type': edge_type, 'edge_trim_alpha': edge_trim_alpha}, 'delaunay_triples_np': delaunay_triples_np, 'mode': 'scipy_topology', 'vertex_degree': vertex_degree, 'graph': graph, 'diagnostics': topo['diagnostics'], 'node_clip_source_vertices': topo.get('node_clip_source_vertices'), 'scipy_vertex_aug_seed_triples': topo.get('scipy_vertex_aug_seed_triples'), 'guard_seeds_uv': topo.get('guard_seeds_uv')}
        for key in ('node_trim_curve_piece', 'node_trim_curve_segment', 'node_trim_curve_fraction', 'node_trim_segment_uv'):
            if key in topo:
                out[key] = topo[key]
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

    def forward(
        self,
        seeds_uv: torch.Tensor | None=None,
        w_raw: torch.Tensor | None=None,
        generate_density_fiber : bool=True,
    ) -> dict[str, Any]:
            if seeds_uv is None:
                raise ValueError('seeds_raw must be provided.')
            if w_raw is None:
                raise ValueError('w_raw must be provided.')
            if self.points_3d is None:
                raise ValueError('points_3d for mesh must be provided.')
            points_uv = torch.as_tensor(self.points_uv, dtype=seeds_uv.dtype, device=seeds_uv.device)
            points_3d = torch.as_tensor(self.points_3d, dtype=seeds_uv.dtype, device=seeds_uv.device)
            Xu = None if self.Xu is None else torch.as_tensor(self.Xu, dtype=seeds_uv.dtype, device=seeds_uv.device)
            Xv = None if self.Xv is None else torch.as_tensor(self.Xv, dtype=seeds_uv.dtype, device=seeds_uv.device)
            use_u_periodic = self._bool_value(self.face_u_periodic)
            use_v_periodic = self._bool_value(self.face_v_periodic)
            cad_domain =self.Cad_domain
            topo_out = self.forward_scipy_topology(
                seeds_uv=seeds_uv,
                cad_domain=cad_domain,
                u_periodic=use_u_periodic,
                v_periodic=use_v_periodic,
                return_xyz=self.return_xyz,
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
                        u_periodic=use_u_periodic,
                        v_periodic=use_v_periodic,
                    )
                if use_torch_cad:
                    coarse_curves_xyz = self.sample_smooth_edge_curves_xyz(cad_domain, curves_uv)
                else:
                    coarse_curves_xyz = self.soft_lift_uv_to_xyz(
                        curves_uv,
                        points_uv,
                        points_3d,
                        u_periodic=use_u_periodic,
                        v_periodic=use_v_periodic,
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
                        u_periodic=use_u_periodic,
                        v_periodic=use_v_periodic,
                    )
                    curves_xyz = (
                        self.sample_smooth_edge_curves_xyz(cad_domain, curves_uv)
                        if use_torch_cad else
                        self.soft_lift_uv_to_xyz(curves_uv, points_uv, points_3d, u_periodic=use_u_periodic, v_periodic=use_v_periodic)
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
                    u_periodic=use_u_periodic,
                    v_periodic=use_v_periodic,
                )
            if curves_xyz.shape[0] == 0 or not generate_density_fiber:
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

        # Original seeds are always shown.
        original_seeds_uv = seeds_uv

        # These are the seeds actually used to build SciPy topology.
        topology_seeds_uv = out.get("topology_seeds_uv", original_seeds_uv)

        # Active mask over original seeds.
        if "seed_active_mask" in out:
            active_mask = out["seed_active_mask"].to(
                device=original_seeds_uv.device,
                dtype=torch.bool,
            )
        else:
            active_mask = torch.ones(
                (original_seeds_uv.shape[0],),
                dtype=torch.bool,
                device=original_seeds_uv.device,
            )

        original_np = original_seeds_uv.detach().cpu().numpy()
        active_mask_np = active_mask.detach().cpu().numpy()

        active_np = original_np[active_mask_np]
        inactive_np = original_np[~active_mask_np]

        topology_np = topology_seeds_uv.detach().cpu().numpy()

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18, 8),
            constrained_layout=True,
        )
        left, middle = axes

        # Left: raw SciPy Voronoi from active/topology seeds only.
        try:
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
        except Exception as error:
            left.text(
                0.5,
                0.5,
                f"SciPy Voronoi unavailable\n{error}",
                ha="center",
                va="center",
            )

        # Show all original seeds, colored by activation.
        if active_np.shape[0] > 0:
            left.scatter(
                active_np[:, 0],
                active_np[:, 1],
                c="green",
                s=55,
                label="Active seeds",
                zorder=5,
            )

        if inactive_np.shape[0] > 0:
            left.scatter(
                inactive_np[:, 0],
                inactive_np[:, 1],
                c="red",
                s=55,
                label="Inactive seeds",
                zorder=5,
            )

        left.set_xlim(0, 1)
        left.set_ylim(0, 1)
        left.set_aspect("equal")
        left.set_title(
            f"VD for {original_seeds_uv.shape[0]} seeds "
            f"({topology_seeds_uv.shape[0]} active)\n"
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

        # Overlay original inactive seeds on generated graph too.
        if inactive_np.shape[0] > 0:
            middle.scatter(
                inactive_np[:, 0],
                inactive_np[:, 1],
                c="red",
                s=55,
                label="Inactive seeds",
                zorder=8,
            )

        if active_np.shape[0] > 0:
            middle.scatter(
                active_np[:, 0],
                active_np[:, 1],
                c="green",
                s=45,
                label="Active seeds",
                zorder=7,
            )

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
