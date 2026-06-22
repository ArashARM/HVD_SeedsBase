from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d, cKDTree


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
        tau_degree: float = 0.5,
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
        seed_boundary_margin: float = 0.02,
        vertex_boundary_margin: float = 0.02,
        boundary_ray_samples: int = 64,
        boundary_ray_length: float = 2.0,
        tau_boundary_project: float = 0.01,
    ):
        super().__init__()
        self.eps = float(eps)
        self.solve_reg = float(solve_reg)
        self.min_seed_dist = float(min_seed_dist)
        self.min_area = float(min_area)
        self.tau_degree = float(tau_degree)
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
        self.seed_boundary_margin = float(seed_boundary_margin)
        self.vertex_boundary_margin = float(vertex_boundary_margin)
        self.boundary_ray_samples = int(boundary_ray_samples)
        self.boundary_ray_length = float(boundary_ray_length)
        self.tau_boundary_project = float(tau_boundary_project)
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

    def unwrap_edge_uv(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an edge endpoint pair on the nearest periodic UV image."""
        p1_unwrapped = p0 + self.periodic_difference(
            p1, p0, u_periodic=u_periodic, v_periodic=v_periodic
        )
        return p0, p1_unwrapped

    def sample_boundary_box_edge_uv(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Sample a UV-box boundary path between two boundary points."""
        if p0.shape != (2,) or p1.shape != (2,):
            raise ValueError("p0 and p1 must each have shape [2].")
        if p0.device != p1.device or p0.dtype != p1.dtype:
            raise ValueError("p0 and p1 must share dtype and device.")
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2.")

        tol = 1e-5
        zero = p0.new_tensor(0.0)
        one = p0.new_tensor(1.0)
        s = torch.linspace(
            0.0, 1.0, n_samples, dtype=p0.dtype, device=p0.device
        )

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
            corners = torch.stack((
                torch.stack((zero, zero)),
                torch.stack((one, zero)),
                torch.stack((one, one)),
                torch.stack((zero, one)),
            ))
            corner_distance = (
                torch.linalg.vector_norm(corners - p0, dim=-1)
                + torch.linalg.vector_norm(corners - p1, dim=-1)
            )
            # Shell topology determines which sides are connected. The nearest
            # shared corner is a discrete routing choice; segment geometry is
            # still differentiable with respect to both endpoints.
            corner = corners[torch.argmin(corner_distance)]
            first_count = n_samples // 2 + 1
            second_count = n_samples - first_count + 1
            first_s = torch.linspace(
                0.0, 1.0, first_count, dtype=p0.dtype, device=p0.device
            )[:, None]
            second_s = torch.linspace(
                0.0, 1.0, second_count, dtype=p0.dtype, device=p0.device
            )[:, None]
            first = p0 + first_s * (corner - p0)
            second = corner + second_s * (p1 - corner)
            curve = torch.cat((first[:-1], second), dim=0)

        return curve.clamp(0.0, 1.0)

    def sample_smooth_edge_curves_uv(
        self,
        seeds_uv: torch.Tensor,
        vertices_uv: torch.Tensor,
        edges: torch.Tensor,
        edge_seed_pairs: torch.Tensor,
        n_samples: int = 64,
        tangent_scale: float = 0.5,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> torch.Tensor:
        """Sample differentiable cubic Hermite curves on fixed graph edges.

        Edge connectivity (possibly supplied by SciPy) is discrete and is not
        differentiable.  Once that connectivity is fixed, the endpoint and
        tangent calculations below remain differentiable with respect to the
        seed and vertex geometry.
        """
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError("seeds_uv must have shape [S, 2].")
        if vertices_uv.ndim != 2 or vertices_uv.shape[-1] != 2:
            raise ValueError("vertices_uv must have shape [V, 2].")
        if edges.ndim != 2 or edges.shape[-1] != 2:
            raise ValueError("edges must have shape [E, 2].")
        if edge_seed_pairs.shape != edges.shape:
            raise ValueError("edge_seed_pairs must have shape [E, 2].")
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2.")
        if edges.device != vertices_uv.device or edge_seed_pairs.device != seeds_uv.device:
            raise ValueError("seeds, vertices, edges, and seed pairs must share a device.")

        num_edges = edges.shape[0]
        if num_edges == 0:
            return vertices_uv.new_empty((0, n_samples, 2))

        p0 = vertices_uv[edges[:, 0]]
        p1 = vertices_uv[edges[:, 1]]
        p0, p1 = self.unwrap_edge_uv(
            p0, p1, u_periodic=u_periodic, v_periodic=v_periodic
        )
        chord = p1 - p0
        length = torch.linalg.vector_norm(chord, dim=-1, keepdim=True)
        straight_tangent = F.normalize(chord, dim=-1, eps=self.eps)

        seed_i = edge_seed_pairs[:, 0]
        seed_j = edge_seed_pairs[:, 1]
        valid_pair = (
            (seed_i >= 0)
            & (seed_j >= 0)
            & (seed_i < seeds_uv.shape[0])
            & (seed_j < seeds_uv.shape[0])
            & (seed_i != seed_j)
        )

        if seeds_uv.shape[0] == 0:
            tangent = straight_tangent
        else:
            # Clamp only makes gathering safe for invalid pairs. torch.where
            # replaces their values with a straight chord tangent.
            safe_i = seed_i.clamp(0, seeds_uv.shape[0] - 1)
            safe_j = seed_j.clamp(0, seeds_uv.shape[0] - 1)
            seed_delta = self.periodic_difference(
                seeds_uv[safe_j],
                seeds_uv[safe_i],
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
            pair_tangent = F.normalize(
                torch.stack((-seed_delta[:, 1], seed_delta[:, 0]), dim=-1),
                dim=-1,
                eps=self.eps,
            )
            orientation = torch.where(
                (pair_tangent * chord).sum(dim=-1, keepdim=True) < 0,
                -torch.ones_like(length),
                torch.ones_like(length),
            )
            pair_tangent = pair_tangent * orientation
            tangent = torch.where(valid_pair[:, None], pair_tangent, straight_tangent)

        endpoint_tangent = tangent * length * tangent_scale
        s = torch.linspace(
            0.0, 1.0, n_samples, dtype=vertices_uv.dtype, device=vertices_uv.device
        ).view(1, n_samples, 1)
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        curve = (
            h00 * p0[:, None, :]
            + h10 * endpoint_tangent[:, None, :]
            + h01 * p1[:, None, :]
            + h11 * endpoint_tangent[:, None, :]
        )
        return self.wrap_uv(curve, u_periodic=u_periodic, v_periodic=v_periodic)

    def sample_graph_edge_curves_uv(
        self,
        seeds_uv: torch.Tensor,
        graph: dict[str, torch.Tensor],
        n_samples: int = 64,
        tangent_scale: float = 0.5,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> torch.Tensor:
        """Sample graph edges according to their discrete edge-type semantics.

        edge_type meanings:
            0 = interior Voronoi edge
            1 = interior-to-boundary clipped Voronoi edge
            2 = reserved
            3 = boundary-to-boundary clipped Voronoi edge
            4 = boundary shell / UV box loop edge

        Only type 4 follows the UV-box boundary. Types 0, 1, and 3 remain
        differentiable Voronoi-bisector Hermite curves.
        """
        nodes_uv = graph["nodes_uv"]
        edge_index = graph["edge_index"]
        edge_seed_pair = graph["edge_seed_pair"]
        edge_type = graph.get("edge_type")
        if edge_type is None:
            edge_type = torch.zeros(
                edge_index.shape[0], dtype=torch.long, device=nodes_uv.device
            )
        if edge_type.ndim != 1 or edge_type.shape[0] != edge_index.shape[0]:
            raise ValueError("graph['edge_type'] must have shape [E].")

        curves = self.sample_smooth_edge_curves_uv(
            seeds_uv=seeds_uv,
            vertices_uv=nodes_uv,
            edges=edge_index,
            edge_seed_pairs=edge_seed_pair,
            n_samples=n_samples,
            tangent_scale=tangent_scale,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        shell_ids = torch.nonzero(edge_type == 4, as_tuple=False).flatten()
        if shell_ids.numel() == 0:
            return curves

        shell_curves = []
        for edge_id in shell_ids:
            a, b = edge_index[edge_id]
            shell_curves.append(
                self.sample_boundary_box_edge_uv(
                    nodes_uv[a], nodes_uv[b], n_samples=n_samples
                )
            )
        shell_curves_t = torch.stack(shell_curves)
        result = curves.clone()
        result[shell_ids] = shell_curves_t
        return result

    def sample_smooth_edge_curves_xyz(
        self,
        cad_domain: Any,
        curves_uv: torch.Tensor,
    ) -> torch.Tensor:
        """Lift UV curves through a differentiable Torch UV-to-XYZ evaluator."""
        evaluator = getattr(cad_domain, "eval_uv_norm_batch_torch", None)
        if evaluator is None or not callable(evaluator):
            raise TypeError(
                "cad_domain must provide differentiable "
                "eval_uv_norm_batch_torch(flat_uv) to sample edge curves in XYZ."
            )
        if curves_uv.ndim != 3 or curves_uv.shape[-1] != 2:
            raise ValueError("curves_uv must have shape [E, n_samples, 2].")

        flat_uv = curves_uv.reshape(-1, 2)
        evaluated = evaluator(flat_uv)
        xyz = evaluated["xyz"] if isinstance(evaluated, dict) else evaluated
        if not isinstance(xyz, torch.Tensor):
            raise TypeError("eval_uv_norm_batch_torch must return a torch.Tensor or {'xyz': tensor}.")
        if xyz.ndim != 2 or xyz.shape != (flat_uv.shape[0], 3):
            raise ValueError("Torch CAD evaluator must return XYZ with shape [E*n_samples, 3].")

        # A smooth cubic UV curve becomes a smooth surface curve while this
        # UV-to-XYZ mapping remains differentiable.
        return xyz.reshape(curves_uv.shape[0], curves_uv.shape[1], 3)

    def softmin_distance_to_curves(
        self,
        query_xyz: torch.Tensor,
        curves_xyz: torch.Tensor,
        tau: float = 0.01,
    ) -> torch.Tensor:
        """Return each query point's soft-min distance to all curve samples."""
        if query_xyz.ndim != 2 or query_xyz.shape[-1] != 3:
            raise ValueError("query_xyz must have shape [M, 3].")
        if curves_xyz.ndim != 3 or curves_xyz.shape[-1] != 3:
            raise ValueError("curves_xyz must have shape [E, K, 3].")
        if query_xyz.device != curves_xyz.device:
            raise ValueError("query_xyz and curves_xyz must share a device.")
        if query_xyz.dtype != curves_xyz.dtype:
            raise ValueError("query_xyz and curves_xyz must share a dtype.")
        if not query_xyz.is_floating_point() or not curves_xyz.is_floating_point():
            raise TypeError("query_xyz and curves_xyz must be floating point tensors.")
        if tau <= 0.0:
            raise ValueError("tau must be greater than zero.")

        curve_points = curves_xyz.reshape(-1, 3)
        if curve_points.shape[0] == 0:
            raise ValueError("curves_xyz must contain at least one curve sample.")
        distances = torch.cdist(query_xyz, curve_points)
        tau_t = query_xyz.new_tensor(float(tau))
        return -tau_t * torch.logsumexp(-distances / tau_t, dim=1)

    def soft_tube_occupancy(
        self,
        query_xyz: torch.Tensor,
        curves_xyz: torch.Tensor,
        radius: torch.Tensor | float,
        tau_distance: float = 0.01,
        tau_occupancy: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        """Build a differentiable swept-sphere occupancy field around curves.

        ``radius`` is the physical positive radius. For a learnable
        unconstrained parameter, apply ``F.softplus`` before calling this
        method (as done with :meth:`make_learnable_radius`).
        """
        if tau_occupancy <= 0.0:
            raise ValueError("tau_occupancy must be greater than zero.")
        radius_tensor = torch.as_tensor(
            radius, dtype=query_xyz.dtype, device=query_xyz.device
        ).clamp_min(self.eps)
        d_soft = self.softmin_distance_to_curves(
            query_xyz=query_xyz,
            curves_xyz=curves_xyz,
            tau=tau_distance,
        )
        tau_occupancy_t = query_xyz.new_tensor(float(tau_occupancy))
        occupancy = torch.sigmoid(
            (radius_tensor - d_soft) / tau_occupancy_t
        )
        return {
            "distance": d_soft,
            "occupancy": occupancy,
            "radius": radius_tensor,
        }

    def make_learnable_radius(
        self,
        initial_radius: float = 0.02,
    ) -> nn.Parameter:
        """Return a parameter whose softplus is ``initial_radius``."""
        if initial_radius <= 0.0:
            raise ValueError("initial_radius must be greater than zero.")
        initial = torch.tensor(float(initial_radius), dtype=torch.get_default_dtype())
        unconstrained = torch.log(torch.expm1(initial))
        return nn.Parameter(unconstrained)

    def graph_tube_field_xyz(
        self,
        seeds_uv: torch.Tensor,
        cad_domain: Any,
        query_xyz: torch.Tensor,
        radius: torch.Tensor | float,
        topology_mode: str = "scipy",
        n_samples: int = 128,
        u_periodic: bool = False,
        v_periodic: bool = False,
        tau_distance: float = 0.01,
        tau_occupancy: float = 0.01,
    ) -> dict[str, Any]:
        """Map seeds to a differentiable XYZ swept-tube field.

        Curve geometry and radius are differentiable while topology is fixed.
        SciPy connectivity, pruning, edge types, and box/corner routing remain
        discrete, matching MeshSDF-style local differentiability.
        """
        out = self(
            seeds_uv,
            topology_mode=topology_mode,
            cad_domain=cad_domain,
            return_xyz=True,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        graph = out["graph"]
        curves_uv = self.sample_graph_edge_curves_uv(
            seeds_uv=seeds_uv,
            graph=graph,
            n_samples=n_samples,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        curves_xyz = self.sample_smooth_edge_curves_xyz(cad_domain, curves_uv)
        tube = self.soft_tube_occupancy(
            query_xyz=query_xyz,
            curves_xyz=curves_xyz,
            radius=radius,
            tau_distance=tau_distance,
            tau_occupancy=tau_occupancy,
        )
        return {
            "occupancy": tube["occupancy"],
            "distance": tube["distance"],
            "radius": tube["radius"],
            "curves_uv": curves_uv,
            "curves_xyz": curves_xyz,
            "graph": graph,
            "decoder_out": out,
        }

    def curve_points_and_tangents_xyz(
        self,
        curves_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten curve samples and their normalized finite-difference tangents."""
        if curves_xyz.ndim != 3 or curves_xyz.shape[-1] != 3:
            raise ValueError("curves_xyz must have shape [E, K, 3].")
        if not curves_xyz.is_floating_point():
            raise TypeError("curves_xyz must be a floating point tensor.")
        if curves_xyz.shape[0] == 0:
            raise ValueError("curves_xyz must contain at least one edge.")
        if curves_xyz.shape[1] < 2:
            raise ValueError("Each curve must contain at least two samples.")

        first = curves_xyz[:, 1] - curves_xyz[:, 0]
        middle = curves_xyz[:, 2:] - curves_xyz[:, :-2]
        last = curves_xyz[:, -1] - curves_xyz[:, -2]
        tangents = torch.cat(
            (first[:, None, :], middle, last[:, None, :]), dim=1
        )
        tangents = tangents / torch.linalg.vector_norm(
            tangents, dim=-1, keepdim=True
        ).clamp_min(self.eps)
        return curves_xyz.reshape(-1, 3), tangents.reshape(-1, 3)

    def soft_tube_density_and_fiber_to_elements(
        self,
        elem_centers_xyz: torch.Tensor,
        curves_xyz: torch.Tensor,
        radius: torch.Tensor | float,
        tau_distance: float = 0.01,
        tau_density: float = 0.01,
        tau_fiber: float = 0.01,
        rho_min: float = 1e-3,
    ) -> dict[str, torch.Tensor]:
        """Map swept graph tubes to structured-grid density and fiber fields."""
        if elem_centers_xyz.ndim != 2 or elem_centers_xyz.shape[-1] != 3:
            raise ValueError("elem_centers_xyz must have shape [numElems, 3].")
        if not elem_centers_xyz.is_floating_point():
            raise TypeError("elem_centers_xyz must be a floating point tensor.")
        if elem_centers_xyz.device != curves_xyz.device:
            raise ValueError("elem_centers_xyz and curves_xyz must share a device.")
        if elem_centers_xyz.dtype != curves_xyz.dtype:
            raise ValueError("elem_centers_xyz and curves_xyz must share a dtype.")
        if tau_distance <= 0.0 or tau_density <= 0.0 or tau_fiber <= 0.0:
            raise ValueError("All distance, density, and fiber temperatures must be positive.")
        if not 0.0 <= rho_min < 1.0:
            raise ValueError("rho_min must satisfy 0 <= rho_min < 1.")

        curve_points, curve_tangents = self.curve_points_and_tangents_xyz(
            curves_xyz
        )
        distances = torch.cdist(elem_centers_xyz, curve_points)
        tau_distance_t = elem_centers_xyz.new_tensor(float(tau_distance))
        d_soft = -tau_distance_t * torch.logsumexp(
            -distances / tau_distance_t, dim=1
        )

        radius_tensor = torch.as_tensor(
            radius,
            dtype=elem_centers_xyz.dtype,
            device=elem_centers_xyz.device,
        ).clamp_min(self.eps)
        tau_density_t = elem_centers_xyz.new_tensor(float(tau_density))
        occupancy = torch.sigmoid(
            (radius_tensor - d_soft) / tau_density_t
        )
        density = float(rho_min) + (1.0 - float(rho_min)) * occupancy

        tau_fiber_t = elem_centers_xyz.new_tensor(float(tau_fiber))
        fiber_weights = torch.softmax(-distances / tau_fiber_t, dim=1)
        fiber = fiber_weights @ curve_tangents
        # Fiber direction is axis-like: f and -f are physically equivalent.
        # Direct vector averaging is used for now; dyadic averaging can replace
        # it later if tangent sign cancellation becomes problematic.
        fiber = fiber / torch.linalg.vector_norm(
            fiber, dim=1, keepdim=True
        ).clamp_min(self.eps)

        ax, ay, az = fiber.unbind(dim=1)
        phi = torch.atan2(ay, ax)
        theta = torch.acos(az.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        return {
            "density": density,
            "fiber": fiber,
            "phi": phi,
            "theta": theta,
            "distance": d_soft,
        }

    def graph_tube_fem_fields(
        self,
        seeds_uv: torch.Tensor,
        cad_domain: Any,
        elem_centers_xyz: torch.Tensor,
        radius: torch.Tensor | float,
        topology_mode: str = "scipy",
        n_samples: int = 128,
        u_periodic: bool = False,
        v_periodic: bool = False,
        tau_distance: float = 0.01,
        tau_density: float = 0.01,
        tau_fiber: float = 0.01,
        rho_min: float = 1e-3,
    ) -> dict[str, Any]:
        """Build differentiable FEM density and orientation fields from seeds.

        Geometry, density, fiber direction, and radius are differentiable for
        fixed topology. SciPy topology, graph pruning, edge typing, and shell
        corner routing remain discrete/local choices.
        """
        out = self(
            seeds_uv,
            topology_mode=topology_mode,
            cad_domain=cad_domain,
            return_xyz=True,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        graph = out["graph"]
        curves_uv = self.sample_graph_edge_curves_uv(
            seeds_uv=seeds_uv,
            graph=graph,
            n_samples=n_samples,
            u_periodic=u_periodic,
            v_periodic=v_periodic,
        )
        curves_xyz = self.sample_smooth_edge_curves_xyz(cad_domain, curves_uv)
        fields = self.soft_tube_density_and_fiber_to_elements(
            elem_centers_xyz=elem_centers_xyz,
            curves_xyz=curves_xyz,
            radius=radius,
            tau_distance=tau_distance,
            tau_density=tau_density,
            tau_fiber=tau_fiber,
            rho_min=rho_min,
        )
        fields.update({
            "curves_uv": curves_uv,
            "curves_xyz": curves_xyz,
            "graph": graph,
            "decoder_out": out,
        })
        return fields

    def plot_tube_curves_pyvista(
        self,
        curves_xyz: torch.Tensor,
        radius: float = 0.01,
    ):
        """Visualize curve tubes with PyVista; never use this in training."""
        if curves_xyz.ndim != 3 or curves_xyz.shape[-1] != 3:
            raise ValueError("curves_xyz must have shape [E, K, 3].")
        if radius <= 0.0:
            raise ValueError("radius must be greater than zero.")
        try:
            import pyvista as pv
        except ImportError as error:
            raise ImportError(
                "PyVista is required for plot_tube_curves_pyvista()."
            ) from error

        curves_np = curves_xyz.detach().cpu().numpy()
        plotter = pv.Plotter()
        for curve in curves_np:
            line = pv.lines_from_points(curve)
            plotter.add_mesh(line.tube(radius=float(radius)), smooth_shading=True)
        plotter.add_axes()
        plotter.show()
        return plotter

    def plot_fem_density_and_fiber_pyvista(
        self,
        elem_centers_xyz: torch.Tensor,
        density: torch.Tensor,
        fiber: torch.Tensor,
        density_threshold: float = 0.2,
    ):
        """Visualize dense FEM centers and fiber glyphs; debugging only."""
        if elem_centers_xyz.ndim != 2 or elem_centers_xyz.shape[-1] != 3:
            raise ValueError("elem_centers_xyz must have shape [numElems, 3].")
        if density.ndim != 1 or density.shape[0] != elem_centers_xyz.shape[0]:
            raise ValueError("density must have shape [numElems].")
        if fiber.shape != elem_centers_xyz.shape:
            raise ValueError("fiber must have shape [numElems, 3].")
        try:
            import pyvista as pv
        except ImportError as error:
            raise ImportError(
                "PyVista is required for plot_fem_density_and_fiber_pyvista()."
            ) from error

        centers_np = elem_centers_xyz.detach().cpu().numpy()
        density_np = density.detach().cpu().numpy()
        fiber_np = fiber.detach().cpu().numpy()
        keep = density_np > float(density_threshold)
        cloud = pv.PolyData(centers_np[keep])
        cloud["density"] = density_np[keep]
        cloud["fiber"] = fiber_np[keep]

        plotter = pv.Plotter()
        if cloud.n_points > 0:
            plotter.add_mesh(
                cloud,
                scalars="density",
                cmap="viridis",
                point_size=8,
                render_points_as_spheres=True,
            )
            span = centers_np.max(axis=0) - centers_np.min(axis=0)
            glyph_scale = max(float((span * span).sum() ** 0.5) * 0.03, self.eps)
            arrows = cloud.glyph(orient="fiber", scale=False, factor=glyph_scale)
            plotter.add_mesh(arrows, color="orange")
        plotter.add_axes()
        plotter.show()
        return plotter

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
            domain_weight = torch.sigmoid(
                (sdf + self.seed_boundary_margin)
                / self._tau_tensor(self.tau_trim, seeds_uv)
            )
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

        if hasattr(cad_domain, "sample_trim_sdf"):
            sdf = cad_domain.sample_trim_sdf(P_uv)
            sdf = torch.as_tensor(sdf, dtype=P_uv.dtype, device=P_uv.device)
            return torch.sigmoid(
                (sdf + self.vertex_boundary_margin)
                / self._tau_tensor(self.tau_trim, P_uv)
            )

        g_trim = cad_domain.smooth_inside_activity(P_uv, tau=self.tau_trim)
        return torch.as_tensor(g_trim, dtype=P_uv.dtype, device=P_uv.device)

    def boundary_distance(
        self,
        P_uv: torch.Tensor,
        cad_domain: Any | None = None,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> torch.Tensor:
        """
        Returns signed/unsigned distance-like value to boundary.
        Smaller absolute value means closer to boundary.
        """
        if cad_domain is not None and hasattr(cad_domain, "sample_trim_sdf"):
            sdf = cad_domain.sample_trim_sdf(P_uv)
            return torch.as_tensor(sdf, dtype=P_uv.dtype, device=P_uv.device)

        distances = []
        if not u_periodic:
            distances += [P_uv[:, 0], 1.0 - P_uv[:, 0]]
        if not v_periodic:
            distances += [P_uv[:, 1], 1.0 - P_uv[:, 1]]

        if len(distances) == 0:
            return torch.zeros(P_uv.shape[0], dtype=P_uv.dtype, device=P_uv.device)

        return torch.stack(distances, dim=-1).amin(dim=-1)

    def _safe_div(self, numerator, denominator, eps):
        sign = torch.where(
            denominator < 0,
            -torch.ones_like(denominator),
            torch.ones_like(denominator),
        )
        denom = torch.where(
            denominator.abs() < eps,
            sign * eps,
            denominator,
        )
        return numerator / denom

    def ray_box_intersection_uv(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> torch.Tensor:
        """
        Intersect ray origin + t direction, t > 0, with normalized UV box [0,1]^2.
        Returns boundary point [2].
        """
        hit, _, valid = self.ray_box_hit_torch(
            origin, direction, u_periodic=u_periodic, v_periodic=v_periodic
        )
        return torch.where(valid, hit, origin)

    def choose_outward_ray_direction(
        self,
        origin,
        seed_i,
        seed_j,
        vertices_uv,
        finite_edges,
        vertex_id,
        u_periodic=False,
        v_periodic=False,
    ):
        """
        Choose the sign of the Voronoi ray direction that points away from existing finite neighbors.
        """
        diff = self.periodic_difference(seed_j, seed_i, u_periodic, v_periodic)
        t = torch.stack([-diff[1], diff[0]])
        t = t / torch.sqrt((t * t).sum() + self.eps)

        neighbor_ids = []
        if finite_edges.numel() > 0:
            mask_a = finite_edges[:, 0] == vertex_id
            mask_b = finite_edges[:, 1] == vertex_id
            neighbor_ids += finite_edges[mask_a, 1].detach().cpu().tolist()
            neighbor_ids += finite_edges[mask_b, 0].detach().cpu().tolist()

        if len(neighbor_ids) == 0:
            return t

        neighbor_vec = vertices_uv[neighbor_ids] - origin.unsqueeze(0)
        mean_neighbor_vec = neighbor_vec.mean(dim=0)

        if torch.dot(t, mean_neighbor_vec) > 0:
            t = -t

        return t

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

    @staticmethod
    def point_inside_box_np(p: np.ndarray, tol: float = 1e-9) -> bool:
        """Hard topology test for the normalized UV box."""
        return bool(
            -tol <= float(p[0]) <= 1.0 + tol
            and -tol <= float(p[1]) <= 1.0 + tol
        )

    @staticmethod
    def segment_box_clip_np(
        p0: np.ndarray,
        p1: np.ndarray,
        bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        tol: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """Liang--Barsky clip of a 2-D segment, including entry/exit parameters."""
        xmin, xmax, ymin, ymax = bounds
        delta = p1 - p0
        t_enter, t_exit = 0.0, 1.0
        for p, q in (
            (-delta[0], p0[0] - xmin),
            (delta[0], xmax - p0[0]),
            (-delta[1], p0[1] - ymin),
            (delta[1], ymax - p0[1]),
        ):
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
        return q0, q1, t_enter, t_exit

    def ray_box_hit_torch(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            return origin, torch.zeros_like(ox), torch.zeros((), dtype=torch.bool, device=origin.device)
        big = torch.as_tensor(float("inf"), dtype=origin.dtype, device=origin.device)
        ts = torch.stack([torch.where(valid, t, big) for t, _, valid in candidates])
        points = torch.stack([point for _, point, _ in candidates])
        index = torch.argmin(ts)
        valid = torch.isfinite(ts[index])
        hit = torch.where(valid, points[index].clamp(0.0, 1.0), origin)
        return hit, ts[index], valid

    def choose_valid_boundary_ray_direction(
        self,
        origin: torch.Tensor,
        seed_i: torch.Tensor,
        seed_j: torch.Tensor,
        cad_domain: Any | None = None,
        u_periodic: bool = False,
        v_periodic: bool = False,
        all_seeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Validate both perpendicular directions and return the shortest valid hit."""
        tangent = self.periodic_difference(seed_j, seed_i, u_periodic, v_periodic)
        normal = torch.stack((-tangent[1], tangent[0]))
        normal = normal / torch.sqrt((normal * normal).sum() + self.eps)
        valid_candidates: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
        ] = []
        for direction in (normal, -normal):
            hit, t, valid = self.ray_box_hit_torch(
                origin, direction, u_periodic=u_periodic, v_periodic=v_periodic
            )
            midpoint = 0.5 * (origin + hit)
            boundary_distance = torch.minimum(
                torch.minimum(hit[0], 1.0 - hit[0]),
                torch.minimum(hit[1], 1.0 - hit[1]),
            ).abs()
            boundary_ok = boundary_distance <= 1e-4
            if cad_domain is not None and self.use_trim_activity:
                midpoint_ok = self.trim_gate(midpoint.unsqueeze(0), cad_domain)[0] > 0.5
            else:
                midpoint_ok = self.point_inside_box_np(
                    midpoint.detach().cpu().numpy(), tol=1e-6
                )
            if bool((valid & boundary_ok & midpoint_ok).detach().cpu().item()):
                voronoi_margin = 0.0
                if all_seeds is not None and all_seeds.shape[0] > 2:
                    probe = origin + 1e-4 * direction
                    distances = self.periodic_distance(
                        probe.unsqueeze(0),
                        all_seeds,
                        u_periodic=u_periodic,
                        v_periodic=v_periodic,
                    )
                    pair_distance = 0.5 * (
                        self.periodic_distance(probe, seed_i, u_periodic, v_periodic)
                        + self.periodic_distance(probe, seed_j, u_periodic, v_periodic)
                    )
                    pair_mask = torch.isclose(distances, pair_distance, atol=1e-5, rtol=1e-5)
                    other_distances = distances.masked_fill(pair_mask, float("inf"))
                    voronoi_margin = float(
                        (other_distances.min() - pair_distance).detach().cpu().item()
                    )
                valid_candidates.append((t, direction, hit, voronoi_margin))
        if not valid_candidates:
            return None
        best_margin = max(item[3] for item in valid_candidates)
        best = [item for item in valid_candidates if item[3] >= best_margin - 1e-8]
        t, direction, hit, _ = min(
            best, key=lambda item: float(item[0].detach().cpu().item())
        )
        return direction, hit, t

    def build_scipy_voronoi_topology(
        self,
        seeds_uv: torch.Tensor,
        cad_domain: Any | None = None,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> dict[str, Any]:
        """
        seeds_uv: torch.Tensor[S,2]
        Returns topology dict built with scipy.spatial.Voronoi using detached NumPy seeds.
        """
        if not isinstance(seeds_uv, torch.Tensor):
            raise TypeError("seeds_uv must be a torch.Tensor.")
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError(f"seeds_uv must have shape [S, 2], got {tuple(seeds_uv.shape)}.")

        device = seeds_uv.device
        dtype = seeds_uv.dtype
        points_np = seeds_uv.detach().cpu().numpy()
        empty_long_2 = lambda: torch.empty((0, 2), dtype=torch.long, device=device)
        empty_long_3 = lambda: torch.empty((0, 3), dtype=torch.long, device=device)
        empty_float_2 = lambda: torch.empty((0, 2), dtype=dtype, device=device)

        def empty_topology() -> dict[str, Any]:
            return {
                "triples": empty_long_3(),
                "vertex_type": torch.empty((0,), dtype=torch.long, device=device),
                "vertex_seed_triples": empty_long_3(),
                "boundary_origin_vertex": torch.empty((0,), dtype=torch.long, device=device),
                "boundary_target_vertex": torch.empty((0,), dtype=torch.long, device=device),
                "boundary_seed_pair": empty_long_2(),
                "boundary_ray_dir": empty_float_2(),
                "boundary_source_type": torch.empty((0,), dtype=torch.long, device=device),
                "edges": empty_long_2(),
                "edge_seed_pairs": empty_long_2(),
                "edge_type": torch.empty((0,), dtype=torch.long, device=device),
                "boundary_rays": empty_long_3(),
                "boundary_ray_dirs": empty_float_2(),
                "scipy_vertices_np": np.empty((0, 2), dtype=points_np.dtype),
                "isolated_vertices": torch.empty((0,), dtype=torch.long, device=device),
                "delaunay_triples_np": np.empty((0, 3), dtype=np.int64),
                "diagnostics": {
                    "num_finite_edges_inside": 0,
                    "num_finite_edges_clipped_once": 0,
                    "num_finite_edges_clipped_twice": 0,
                    "num_infinite_rays_clipped": 0,
                    "num_discarded_rays": 0,
                    "num_raw_scipy_vertices": 0,
                    "num_raw_boundary_vertices": 0,
                    "num_pruned_vertices": 0,
                    "num_final_vertices": 0,
                    "num_final_interior_vertices": 0,
                    "num_final_boundary_vertices": 0,
                },
            }

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
        diagnostics = {
            "num_finite_edges_inside": 0,
            "num_finite_edges_clipped_once": 0,
            "num_finite_edges_clipped_twice": 0,
            "num_infinite_rays_clipped": 0,
            "num_discarded_rays": 0,
            "num_raw_scipy_vertices": num_raw_scipy_vertices,
        }

        def add_boundary_vertex(
            origin_vertex: int,
            target_vertex: int,
            seed_i: int,
            seed_j: int,
            direction: np.ndarray,
            source_type: int,
        ) -> int:
            boundary_id = len(vertex_type)
            vertex_type.append(1)
            vertex_seed_triples.append([seed_i, seed_j, -1])
            boundary_origin_vertex.append(origin_vertex)
            boundary_target_vertex.append(target_vertex)
            boundary_seed_pair.append([seed_i, seed_j])
            boundary_ray_dir.append([float(direction[0]), float(direction[1])])
            boundary_source_type.append(source_type)
            return boundary_id

        for seed_pair, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
            finite_vertices = [int(v) for v in ridge_vertices if int(v) >= 0]
            seed_i = int(seed_pair[0])
            seed_j = int(seed_pair[1])

            if len(finite_vertices) == 2:
                a, b = finite_vertices
                pa, pb = scipy_vertices_np[a], scipy_vertices_np[b]
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
                    diagnostics["num_finite_edges_inside"] += 1
                elif a_inside != b_inside:
                    inside_id, outside_id = (a, b) if a_inside else (b, a)
                    direction = scipy_vertices_np[outside_id] - scipy_vertices_np[inside_id]
                    direction /= np.linalg.norm(direction) + 1e-12
                    boundary_id = add_boundary_vertex(
                        inside_id, outside_id, seed_i, seed_j, direction, 2
                    )
                    edges.append([inside_id, boundary_id])
                    edge_seed_pairs.append([seed_i, seed_j])
                    edge_types.append(1)
                    diagnostics["num_finite_edges_clipped_once"] += 1
                elif t_exit - t_enter > 1e-12:
                    direction_ab = pb - pa
                    direction_ab /= np.linalg.norm(direction_ab) + 1e-12
                    entry_id = add_boundary_vertex(
                        a, b, seed_i, seed_j, direction_ab, 2
                    )
                    exit_id = add_boundary_vertex(
                        b, a, seed_i, seed_j, -direction_ab, 2
                    )
                    edges.append([entry_id, exit_id])
                    edge_seed_pairs.append([seed_i, seed_j])
                    edge_types.append(3)
                    diagnostics["num_finite_edges_clipped_twice"] += 1
            elif len(finite_vertices) == 1 and any(int(v) == -1 for v in ridge_vertices):
                finite_v = finite_vertices[0]
                origin = torch.as_tensor(
                    scipy_vertices_np[finite_v], dtype=dtype, device=device
                )
                selected = self.choose_valid_boundary_ray_direction(
                    origin,
                    seeds_uv[seed_i].detach(),
                    seeds_uv[seed_j].detach(),
                    cad_domain=cad_domain,
                    u_periodic=u_periodic,
                    v_periodic=v_periodic,
                    all_seeds=seeds_uv.detach(),
                )
                if selected is None:
                    diagnostics["num_discarded_rays"] += 1
                    continue
                direction_t, _, _ = selected
                direction = direction_t.detach().cpu().numpy()
                boundary_id = add_boundary_vertex(
                    finite_v, -1, seed_i, seed_j, direction, 1
                )
                edges.append([finite_v, boundary_id])
                edge_seed_pairs.append([seed_i, seed_j])
                edge_types.append(1)
                boundary_rays.append([finite_v, seed_i, seed_j])
                boundary_ray_dirs.append([float(direction[0]), float(direction[1])])
                diagnostics["num_infinite_rays_clipped"] += 1

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

        boundary_ray_dirs_t = torch.as_tensor(
            boundary_ray_dirs,
            dtype=seeds_uv.dtype,
            device=device,
        )
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

        diagnostics["num_raw_boundary_vertices"] = (
            len(vertex_type) - num_raw_scipy_vertices
        )

        return {
            "triples": vertex_seed_triples_t,
            "vertex_type": vertex_type_t,
            "vertex_seed_triples": vertex_seed_triples_t,
            "boundary_origin_vertex": boundary_origin_vertex_t,
            "boundary_target_vertex": boundary_target_vertex_t,
            "boundary_seed_pair": boundary_seed_pair_t,
            "boundary_ray_dir": boundary_ray_dir_t,
            "boundary_source_type": boundary_source_type_t,
            "edges": edges_t,
            "edge_seed_pairs": edge_seed_pairs_t,
            "edge_type": torch.as_tensor(edge_types, dtype=torch.long, device=device),
            "boundary_rays": boundary_rays_t,
            "boundary_ray_dirs": boundary_ray_dirs_t,
            "scipy_vertices_np": scipy_vertices_np,
            "isolated_vertices": isolated_t,
            "delaunay_triples_np": delaunay_triples_np,
            "diagnostics": diagnostics,
        }

    def prune_graph_vertices(
        self,
        nodes_uv: torch.Tensor,
        vertex_type: torch.Tensor,
        vertex_seed_triples: torch.Tensor,
        boundary_origin_vertex: torch.Tensor,
        boundary_target_vertex: torch.Tensor,
        boundary_seed_pair: torch.Tensor,
        boundary_ray_dir: torch.Tensor,
        boundary_source_type: torch.Tensor,
        edges: torch.Tensor,
        edge_seed_pairs: torch.Tensor,
        edge_type: torch.Tensor,
        alpha: torch.Tensor | None = None,
        keep_isolated_vertices: bool = False,
    ) -> dict[str, torch.Tensor | int | None]:
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
        return {
            "nodes_uv": nodes_uv[active_ids],
            "vertex_type": compact_type,
            "vertex_seed_triples": vertex_seed_triples[active_ids],
            "boundary_origin_vertex": remap_reference(boundary_origin_vertex),
            "boundary_target_vertex": remap_reference(boundary_target_vertex),
            "boundary_seed_pair": boundary_seed_pair[active_ids],
            "boundary_ray_dir": boundary_ray_dir[active_ids],
            "boundary_source_type": boundary_source_type[active_ids],
            "edges": compact_edges,
            "edge_seed_pairs": edge_seed_pairs,
            "edge_type": edge_type,
            "alpha": None if alpha is None else alpha[active_ids],
            "old_to_new": old_to_new,
            "active_vertex_ids": active_ids,
            "num_pruned_vertices": num_vertices - int(active_ids.numel()),
        }

    def differentiable_vertices_from_topology(
        self,
        seeds_uv: torch.Tensor,
        vertex_type: torch.Tensor,
        vertex_seed_triples: torch.Tensor,
        boundary_origin_vertex: torch.Tensor,
        boundary_seed_pair: torch.Tensor,
        boundary_ray_dir: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
        cad_domain: Any | None = None,
        boundary_target_vertex: torch.Tensor | None = None,
        boundary_source_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct unified SciPy topology with differentiable coordinates."""
        num_vertices = vertex_type.shape[0]
        if num_vertices == 0:
            return torch.empty((0, 2), dtype=seeds_uv.dtype, device=seeds_uv.device)

        nodes = torch.zeros((num_vertices, 2), dtype=seeds_uv.dtype, device=seeds_uv.device)
        if boundary_target_vertex is None:
            boundary_target_vertex = torch.full_like(boundary_origin_vertex, -1)
        if boundary_source_type is None:
            boundary_source_type = vertex_type
        interior_ids = torch.nonzero(vertex_type == 0, as_tuple=False).flatten()
        if interior_ids.numel() > 0:
            triples = vertex_seed_triples[interior_ids]
            nodes[interior_ids] = self.differentiable_vertices_from_triples(
                seeds_uv, triples, u_periodic, v_periodic
            )

        for boundary_id in torch.nonzero(vertex_type == 1, as_tuple=False).flatten().tolist():
            origin_id = int(boundary_origin_vertex[boundary_id].item())
            if origin_id < 0 or origin_id >= num_vertices:
                continue
            pair = boundary_seed_pair[boundary_id]
            i, j = int(pair[0].item()), int(pair[1].item())
            stored_direction = boundary_ray_dir[boundary_id].to(dtype=seeds_uv.dtype)
            source_type = int(boundary_source_type[boundary_id].item())

            if source_type == 2:
                target_id = int(boundary_target_vertex[boundary_id].item())
                if target_id < 0 or target_id >= num_vertices:
                    continue
                direction = nodes[target_id] - nodes[origin_id]
                direction = direction / torch.sqrt((direction * direction).sum() + self.eps)
            elif 0 <= i < seeds_uv.shape[0] and 0 <= j < seeds_uv.shape[0]:
                tangent = self.periodic_difference(seeds_uv[j], seeds_uv[i], u_periodic, v_periodic)
                direction = torch.stack((-tangent[1], tangent[0]))
                direction = direction / torch.sqrt((direction * direction).sum() + self.eps)
                sign = torch.where(
                    torch.dot(direction, stored_direction) < 0,
                    -torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device),
                    torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device),
                )
                direction = direction * sign
            else:
                direction = stored_direction
                direction = direction / torch.sqrt((direction * direction).sum() + self.eps)

            # TODO: replace this with differentiable trim-curve intersection for CAD domains.
            nodes[boundary_id] = self.ray_box_intersection_uv(
                nodes[origin_id], direction, u_periodic=u_periodic, v_periodic=v_periodic
            )
        return nodes

    def _box_bisector_intersections(
        self,
        seed_i: torch.Tensor,
        seed_j: torch.Tensor,
        tol: float = 1e-7,
    ) -> list[torch.Tensor]:
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
            if not any(
                torch.linalg.vector_norm(candidate.detach() - other.detach()) <= tol
                for other in unique
            ):
                unique.append(candidate)
        return unique

    def _pair_boundary_candidate_alpha(
        self,
        seeds_uv: torch.Tensor,
        pair: torch.Tensor,
        candidate: torch.Tensor,
        seed_activity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Soft nearest-pair validity at a bisector-boundary intersection."""
        i, j = int(pair[0].item()), int(pair[1].item())
        distances = self.periodic_distance(candidate.unsqueeze(0), seeds_uv)
        pair_distance = 0.5 * (distances[i] + distances[j])
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask[i] = False
        mask[j] = False
        if bool(mask.any().detach().cpu().item()):
            margin = distances[mask].min() - pair_distance
            nearest_gate = torch.sigmoid(
                margin / self._tau_tensor(self.tau_voronoi, seeds_uv)
            )
        else:
            nearest_gate = torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device)
        equality_gate = torch.exp(
            -((distances[i] - distances[j]) / self._tau_tensor(self.tau_voronoi, seeds_uv)) ** 2
        )
        activity = torch.ones((), dtype=seeds_uv.dtype, device=seeds_uv.device)
        if seed_activity is not None:
            activity = seed_activity[i] * seed_activity[j]
        return (activity * nearest_gate * equality_gate).clamp(0.0, 1.0)

    def _cad_bisector_intersections(
        self,
        seed_i: torch.Tensor,
        seed_j: torch.Tensor,
        cad_domain: Any,
        samples: int = 129,
    ) -> list[torch.Tensor]:
        """Find trim-boundary crossings along the box-clipped pair bisector."""
        segment = self._box_bisector_intersections(seed_i, seed_j)
        if len(segment) != 2:
            return []
        t = torch.linspace(0.0, 1.0, samples, dtype=seed_i.dtype, device=seed_i.device)
        points = segment[0].unsqueeze(0) + t.unsqueeze(1) * (segment[1] - segment[0]).unsqueeze(0)
        if hasattr(cad_domain, "sample_trim_sdf"):
            values = cad_domain.sample_trim_sdf(points)
            values = torch.as_tensor(values, dtype=seed_i.dtype, device=seed_i.device).reshape(-1)
        elif hasattr(cad_domain, "smooth_inside_activity"):
            values = cad_domain.smooth_inside_activity(points, tau=self.tau_trim)
            values = torch.as_tensor(values, dtype=seed_i.dtype, device=seed_i.device).reshape(-1) - 0.5
        else:
            return []

        intersections: list[torch.Tensor] = []
        for index in range(samples - 1):
            a, b = values[index], values[index + 1]
            crosses = (a == 0) | (b == 0) | ((a < 0) != (b < 0))
            if not bool(crosses.detach().cpu().item()):
                continue
            weight = (a / (a - b + self.eps)).clamp(0.0, 1.0)
            point = points[index] + weight * (points[index + 1] - points[index])
            if not intersections or torch.linalg.vector_norm(
                point.detach() - intersections[-1].detach()
            ) > 1e-4:
                intersections.append(point)
        return intersections

    def add_pair_boundary_candidates(
        self,
        seeds_uv: torch.Tensor,
        nodes_uv: torch.Tensor,
        node_alpha: torch.Tensor,
        node_type: torch.Tensor,
        node_seed_triples: torch.Tensor,
        boundary_seed_pair: torch.Tensor,
        boundary_source_type: torch.Tensor,
        edges: torch.Tensor,
        edge_seed_pairs: torch.Tensor,
        edge_type: torch.Tensor,
        cad_domain: Any | None = None,
        seed_activity: torch.Tensor | None = None,
        hard_validity: bool = True,
        tol: float = 1e-4,
    ) -> dict[str, torch.Tensor]:
        """Add missing valid pair-bisector intersections and their Voronoi edges."""
        device, dtype = seeds_uv.device, seeds_uv.dtype
        pairs = torch.combinations(torch.arange(seeds_uv.shape[0], device=device), r=2)
        added_nodes: list[torch.Tensor] = []
        added_alpha: list[torch.Tensor] = []
        added_pairs: list[list[int]] = []
        added_edges: list[list[int]] = []
        added_edge_pairs: list[list[int]] = []
        added_edge_types: list[int] = []

        for pair in pairs:
            i, j = int(pair[0].item()), int(pair[1].item())
            if cad_domain is None:
                candidates = self._box_bisector_intersections(seeds_uv[i], seeds_uv[j])
            elif hasattr(cad_domain, "intersect_bisector_boundary"):
                raw = cad_domain.intersect_bisector_boundary(
                    seeds_uv[i], seeds_uv[j]
                )
                raw = torch.as_tensor(raw, dtype=dtype, device=device).reshape(-1, 2)
                candidates = list(raw.unbind(0))
            else:
                candidates = self._cad_bisector_intersections(
                    seeds_uv[i], seeds_uv[j], cad_domain
                )

            pair_in_triple = (
                (node_seed_triples == i).any(dim=1)
                & (node_seed_triples == j).any(dim=1)
                & (node_type == 0)
            ) if node_seed_triples.numel() > 0 else torch.zeros(
                (nodes_uv.shape[0],), dtype=torch.bool, device=device
            )
            interior_ids = torch.nonzero(pair_in_triple, as_tuple=False).flatten()
            if interior_ids.numel() > 0:
                inside = (
                    (nodes_uv[interior_ids, 0] >= -tol)
                    & (nodes_uv[interior_ids, 0] <= 1.0 + tol)
                    & (nodes_uv[interior_ids, 1] >= -tol)
                    & (nodes_uv[interior_ids, 1] <= 1.0 + tol)
                    & (node_alpha[interior_ids] >= 0.5)
                )
                interior_ids = interior_ids[inside]

            valid_for_pair: list[tuple[torch.Tensor, torch.Tensor]] = []
            for candidate in candidates:
                alpha = self._pair_boundary_candidate_alpha(
                    seeds_uv, pair, candidate, seed_activity
                )
                if hard_validity and float(alpha.detach().cpu().item()) < 0.5:
                    continue
                existing_pair = (
                    (boundary_seed_pair[:, 0] == i)
                    & (boundary_seed_pair[:, 1] == j)
                    | (boundary_seed_pair[:, 0] == j)
                    & (boundary_seed_pair[:, 1] == i)
                ) if boundary_seed_pair.numel() > 0 else torch.zeros(
                    (nodes_uv.shape[0],), dtype=torch.bool, device=device
                )
                existing_ids = torch.nonzero(existing_pair, as_tuple=False).flatten()
                if existing_ids.numel() > 0 and bool(
                    (torch.linalg.vector_norm(
                        nodes_uv[existing_ids].detach() - candidate.detach(), dim=1
                    ).min() <= tol).detach().cpu().item()
                ):
                    continue
                valid_for_pair.append((candidate, alpha))

            if interior_ids.numel() == 0 and len(valid_for_pair) == 2:
                midpoint = 0.5 * (valid_for_pair[0][0] + valid_for_pair[1][0])
                midpoint_alpha = self._pair_boundary_candidate_alpha(
                    seeds_uv, pair, midpoint, seed_activity
                )
                domain_ok = (
                    self.trim_gate(midpoint.unsqueeze(0), cad_domain)[0] > 0.5
                    if cad_domain is not None else torch.ones((), dtype=torch.bool, device=device)
                )
                if (
                    float(midpoint_alpha.detach().cpu().item()) < 0.5
                    or not bool(domain_ok.detach().cpu().item())
                ):
                    valid_for_pair = []

            new_ids: list[int] = []
            for candidate, alpha in valid_for_pair:
                target = None
                if interior_ids.numel() > 0:
                    distances = torch.linalg.vector_norm(
                        nodes_uv[interior_ids].detach() - candidate.detach(), dim=1
                    )
                    target = int(interior_ids[torch.argmin(distances)].item())
                    segment_midpoint = 0.5 * (candidate + nodes_uv[target])
                    midpoint_alpha = self._pair_boundary_candidate_alpha(
                        seeds_uv, pair, segment_midpoint, seed_activity
                    )
                    domain_ok = (
                        self.trim_gate(segment_midpoint.unsqueeze(0), cad_domain)[0] > 0.5
                        if cad_domain is not None else torch.ones((), dtype=torch.bool, device=device)
                    )
                    if (
                        float(midpoint_alpha.detach().cpu().item()) < 0.5
                        or not bool(domain_ok.detach().cpu().item())
                    ):
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
                midpoint_alpha = self._pair_boundary_candidate_alpha(
                    seeds_uv, pair, midpoint, seed_activity
                )
                if float(midpoint_alpha.detach().cpu().item()) >= 0.5:
                    added_edges.append(new_ids)
                    added_edge_pairs.append([i, j])
                    added_edge_types.append(3)

        if not added_nodes:
            return {
                "nodes_uv": nodes_uv, "node_alpha": node_alpha, "node_type": node_type,
                "node_seed_triples": node_seed_triples,
                "boundary_seed_pair": boundary_seed_pair,
                "boundary_source_type": boundary_source_type,
                "edges": edges, "edge_seed_pairs": edge_seed_pairs, "edge_type": edge_type,
            }

        added_nodes_t = torch.stack(added_nodes)
        added_pairs_t = torch.as_tensor(added_pairs, dtype=torch.long, device=device)
        added_count = added_nodes_t.shape[0]
        nodes_uv = torch.cat((nodes_uv, added_nodes_t), dim=0)
        node_alpha = torch.cat((node_alpha, torch.stack(added_alpha)), dim=0)
        node_type = torch.cat((node_type, torch.ones(added_count, dtype=torch.long, device=device)))
        node_seed_triples = torch.cat((
            node_seed_triples,
            torch.cat((added_pairs_t, -torch.ones((added_count, 1), dtype=torch.long, device=device)), dim=1),
        ), dim=0)
        boundary_seed_pair = torch.cat((boundary_seed_pair, added_pairs_t), dim=0)
        boundary_source_type = torch.cat((
            boundary_source_type,
            torch.full((added_count,), 3, dtype=torch.long, device=device),
        ))
        if added_edges:
            edges = torch.cat((edges, torch.as_tensor(added_edges, dtype=torch.long, device=device)), dim=0)
            edge_seed_pairs = torch.cat((
                edge_seed_pairs,
                torch.as_tensor(added_edge_pairs, dtype=torch.long, device=device),
            ), dim=0)
            edge_type = torch.cat((
                edge_type,
                torch.as_tensor(added_edge_types, dtype=torch.long, device=device),
            ))
        return {
            "nodes_uv": nodes_uv, "node_alpha": node_alpha, "node_type": node_type,
            "node_seed_triples": node_seed_triples,
            "boundary_seed_pair": boundary_seed_pair,
            "boundary_source_type": boundary_source_type,
            "edges": edges, "edge_seed_pairs": edge_seed_pairs, "edge_type": edge_type,
        }

    def add_box_shell_corners(
        self,
        nodes_uv: torch.Tensor,
        node_alpha: torch.Tensor,
        node_type: torch.Tensor,
        node_seed_triples: torch.Tensor,
        boundary_seed_pair: torch.Tensor,
        boundary_source_type: torch.Tensor,
        tol: float = 1e-4,
    ) -> dict[str, torch.Tensor]:
        """Append missing UV-box corners as boundary shell nodes (source type 4)."""
        device, dtype = nodes_uv.device, nodes_uv.dtype
        corners = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=dtype, device=device,
        )
        missing = []
        for corner in corners:
            if nodes_uv.numel() == 0 or not bool(
                (torch.linalg.vector_norm(nodes_uv.detach() - corner, dim=1) <= tol).any().item()
            ):
                missing.append(corner)
        if not missing:
            return {
                "nodes_uv": nodes_uv, "node_alpha": node_alpha, "node_type": node_type,
                "node_seed_triples": node_seed_triples,
                "boundary_seed_pair": boundary_seed_pair,
                "boundary_source_type": boundary_source_type,
            }
        added = torch.stack(missing)
        count = added.shape[0]
        return {
            "nodes_uv": torch.cat((nodes_uv, added), dim=0),
            "node_alpha": torch.cat((node_alpha, torch.ones(count, dtype=dtype, device=device))),
            "node_type": torch.cat((node_type, torch.ones(count, dtype=torch.long, device=device))),
            "node_seed_triples": torch.cat((node_seed_triples, torch.full((count, 3), -1, dtype=torch.long, device=device))),
            "boundary_seed_pair": torch.cat((boundary_seed_pair, torch.full((count, 2), -1, dtype=torch.long, device=device))),
            "boundary_source_type": torch.cat((boundary_source_type, torch.full((count,), 4, dtype=torch.long, device=device))),
        }

    def build_soft_completed_graph(
        self,
        seeds_uv: torch.Tensor,
        vertices_uv: torch.Tensor,
        alpha: torch.Tensor,
        triples: torch.Tensor,
        edge_data: dict[str, torch.Tensor],
        seed_activity: torch.Tensor,
        cad_domain: Any | None = None,
    ) -> dict[str, Any]:
        """Complete soft triple topology with pair-boundary and shell nodes."""
        device, dtype = seeds_uv.device, seeds_uv.dtype
        num_vertices = vertices_uv.shape[0]
        augmented = self.add_pair_boundary_candidates(
            seeds_uv=seeds_uv,
            nodes_uv=vertices_uv,
            node_alpha=alpha,
            node_type=torch.zeros(num_vertices, dtype=torch.long, device=device),
            node_seed_triples=triples,
            boundary_seed_pair=torch.full((num_vertices, 2), -1, dtype=torch.long, device=device),
            boundary_source_type=torch.zeros(num_vertices, dtype=torch.long, device=device),
            edges=edge_data["edge_index"],
            edge_seed_pairs=edge_data["edge_seed_pair"],
            edge_type=torch.zeros(edge_data["edge_index"].shape[0], dtype=torch.long, device=device),
            cad_domain=cad_domain,
            seed_activity=seed_activity,
            hard_validity=True,
        )
        if cad_domain is None:
            augmented.update(self.add_box_shell_corners(
                nodes_uv=augmented["nodes_uv"],
                node_alpha=augmented["node_alpha"],
                node_type=augmented["node_type"],
                node_seed_triples=augmented["node_seed_triples"],
                boundary_seed_pair=augmented["boundary_seed_pair"],
                boundary_source_type=augmented["boundary_source_type"],
            ))
        shell_edges, shell_types = self.build_boundary_loop_edges(
            augmented["nodes_uv"], augmented["node_type"], cad_domain=cad_domain
        )
        shell_pairs = torch.full((shell_edges.shape[0], 2), -1, dtype=torch.long, device=device)
        edges = torch.cat((augmented["edges"], shell_edges), dim=0)
        edge_seed_pairs = torch.cat((augmented["edge_seed_pairs"], shell_pairs), dim=0)
        edge_type = torch.cat((augmented["edge_type"], shell_types), dim=0)
        if edges.numel() > 0:
            edge_alpha = augmented["node_alpha"][edges[:, 0]] * augmented["node_alpha"][edges[:, 1]]
        else:
            edge_alpha = torch.empty((0,), dtype=dtype, device=device)
        degree = self.exact_vertex_degree(
            augmented["nodes_uv"].shape[0], edges, dtype, device
        )
        source_names = [
            {0: "interior", 3: "pair_bisector_boundary", 4: "corner_shell"}.get(
                int(value), "boundary"
            )
            for value in augmented["boundary_source_type"].detach().cpu().tolist()
        ]
        return {
            "nodes_uv": augmented["nodes_uv"],
            "node_alpha": augmented["node_alpha"],
            "node_type": augmented["node_type"],
            "node_degree": degree,
            "edge_index": edges,
            "edge_seed_pair": edge_seed_pairs,
            "edge_alpha": edge_alpha,
            "edge_type": edge_type,
            "vertex_degree": degree,
            "vertex_seed_triples": augmented["node_seed_triples"],
            "boundary_seed_pair": augmented["boundary_seed_pair"],
            "boundary_source_type": augmented["boundary_source_type"],
            "boundary_source_name": source_names,
            "num_interior_nodes": int((augmented["node_type"] == 0).sum().item()),
            "num_boundary_nodes": int((augmented["node_type"] == 1).sum().item()),
        }

    def build_boundary_loop_edges(
        self,
        nodes_uv: torch.Tensor,
        vertex_type: torch.Tensor,
        cad_domain: Any | None = None,
        tol: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Connect boundary nodes cyclically in shell-parameter order."""
        device = nodes_uv.device
        boundary_ids = torch.nonzero(vertex_type == 1, as_tuple=False).flatten()
        if boundary_ids.numel() < 2:
            return (
                torch.empty((0, 2), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )

        p = nodes_uv.detach()[boundary_ids]
        loops: list[torch.Tensor] = []
        if cad_domain is None:
            u, v = p[:, 0], p[:, 1]
            parameter = torch.empty_like(u)
            bottom = torch.abs(v) <= tol
            right = (~bottom) & (torch.abs(u - 1.0) <= tol)
            top = (~bottom) & (~right) & (torch.abs(v - 1.0) <= tol)
            left = ~(bottom | right | top)
            parameter[bottom] = u[bottom]
            parameter[right] = 1.0 + v[right]
            parameter[top] = 3.0 - u[top]
            parameter[left] = 4.0 - v[left]
            loops = [boundary_ids[torch.argsort(parameter)]]
        else:
            used_projection_hook = False
            if hasattr(cad_domain, "boundary_parameter"):
                result = cad_domain.boundary_parameter(nodes_uv[boundary_ids])
            elif hasattr(cad_domain, "project_to_boundary_with_parameter"):
                used_projection_hook = True
                result = cad_domain.project_to_boundary_with_parameter(nodes_uv[boundary_ids])
            else:
                raise AttributeError(
                    "CAD shell completion requires cad_domain.boundary_parameter(P_uv) "
                    "or cad_domain.project_to_boundary_with_parameter(P_uv)."
                )
            if isinstance(result, dict):
                loop_id = torch.as_tensor(result.get("loop_id", 0), device=device).reshape(-1)
                parameter_value = result.get("parameter", result.get("s"))
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
        # edge_type contract:
        #   0 interior Voronoi, 1 clipped interior-boundary, 2 reserved,
        #   3 clipped boundary-boundary, 4 boundary shell / UV-box loop.
        return edge_index, torch.full((edge_index.shape[0],), 4, dtype=torch.long, device=device)

    def differentiable_vertices_from_triples(
        self,
        seeds_uv: torch.Tensor,
        triples: torch.Tensor,
        u_periodic: bool = False,
        v_periodic: bool = False,
    ) -> torch.Tensor:
        """
        Recompute SciPy-selected vertices using differentiable PyTorch circumcenter.
        """
        if triples.numel() == 0:
            return torch.empty((0, 2), dtype=seeds_uv.dtype, device=seeds_uv.device)

        _, P_uv, _, _ = self.circumcenters_from_triples(
            seeds_uv,
            triples,
            u_periodic,
            v_periodic,
        )
        return P_uv

    def complete_boundary_graph(
        self,
        seeds_uv,
        vertices_uv,
        alpha,
        edges,
        edge_seed_pairs,
        boundary_rays,
        boundary_ray_dirs,
        cad_domain=None,
        u_periodic=False,
        v_periodic=False,
    ):
        """
        Convert open Voronoi rays into boundary nodes and clipped boundary edges.

        Final graph:
            interior nodes = vertices_uv
            boundary nodes = clipped boundary intersections
            edges = finite Voronoi edges + clipped boundary edges
        """
        V = vertices_uv.shape[0]
        device = vertices_uv.device
        dtype = vertices_uv.dtype

        if boundary_rays.numel() == 0:
            node_type = torch.zeros(V, dtype=torch.long, device=device)

            if edges.numel() == 0:
                full_edge_alpha = torch.empty((0,), dtype=dtype, device=device)
            else:
                full_edge_alpha = alpha[edges[:, 0]] * alpha[edges[:, 1]]

            node_degree = self.exact_vertex_degree(
                num_vertices=V,
                edge_index=edges,
                dtype=dtype,
                device=device,
            )

            return {
                "nodes_uv": vertices_uv,
                "node_alpha": alpha,
                "node_type": node_type,
                "node_degree": node_degree,
                "edge_index": edges,
                "edge_seed_pair": edge_seed_pairs,
                "edge_alpha": full_edge_alpha,
                "edge_type": torch.zeros(
                    edges.shape[0], dtype=torch.long, device=device
                ),
                "num_interior_nodes": V,
                "num_boundary_nodes": 0,
            }

        boundary_vertices = []
        boundary_alpha = []
        boundary_edges = []
        boundary_seed_pairs = []

        for ray_idx, ray in enumerate(boundary_rays):
            v_id = int(ray[0].item())
            i = int(ray[1].item())
            j = int(ray[2].item())

            origin = vertices_uv[v_id]

            if (
                boundary_ray_dirs is not None
                and boundary_ray_dirs.ndim == 2
                and boundary_ray_dirs.shape[0] == boundary_rays.shape[0]
                and boundary_ray_dirs.shape[1] == 2
            ):
                direction = boundary_ray_dirs[ray_idx].to(device=device, dtype=dtype)
                direction = direction / torch.sqrt((direction * direction).sum() + self.eps)
            else:
                direction = self.choose_outward_ray_direction(
                    origin=origin,
                    seed_i=seeds_uv[i],
                    seed_j=seeds_uv[j],
                    vertices_uv=vertices_uv,
                    finite_edges=edges,
                    vertex_id=v_id,
                    u_periodic=u_periodic,
                    v_periodic=v_periodic,
                )

            B = self.ray_box_intersection_uv(
                origin,
                direction,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )

            # Boundary point must really lie on the UV box boundary.
            boundary_dist = torch.minimum(
                torch.minimum(B[0], 1.0 - B[0]),
                torch.minimum(B[1], 1.0 - B[1]),
            )
            boundary_ok = boundary_dist < 1.0e-4

            # Segment midpoint should remain inside the active domain.
            # This removes wrong rays that cut across the bounded domain.
            seg_mid = 0.5 * (origin + B)

            if cad_domain is not None and self.use_trim_activity:
                mid_activity = self.trim_gate(seg_mid.unsqueeze(0), cad_domain)[0]
            else:
                mid_activity = self.box_gate(
                    seg_mid.unsqueeze(0),
                    u_periodic=u_periodic,
                    v_periodic=v_periodic,
                )[0]

            mid_ok = mid_activity > 0.5

            # This is a topology/filtering decision, so hard bool is acceptable
            # in the MeshSDF-style local-differentiability setting.
            if not bool((boundary_ok & mid_ok).detach().cpu().item()):
                continue

            new_boundary_id = V + len(boundary_vertices)

            B_alpha = alpha[v_id]

            boundary_vertices.append(B)
            boundary_alpha.append(B_alpha)
            boundary_edges.append([v_id, new_boundary_id])
            boundary_seed_pairs.append([i, j])

        if len(boundary_vertices) == 0:
            node_type = torch.zeros(V, dtype=torch.long, device=device)

            if edges.numel() == 0:
                full_edge_alpha = torch.empty((0,), dtype=dtype, device=device)
            else:
                full_edge_alpha = alpha[edges[:, 0]] * alpha[edges[:, 1]]

            node_degree = self.exact_vertex_degree(
                num_vertices=V,
                edge_index=edges,
                dtype=dtype,
                device=device,
            )

            return {
                "nodes_uv": vertices_uv,
                "node_alpha": alpha,
                "node_type": node_type,
                "node_degree": node_degree,
                "edge_index": edges,
                "edge_seed_pair": edge_seed_pairs,
                "edge_alpha": full_edge_alpha,
                "edge_type": torch.zeros(
                    edges.shape[0], dtype=torch.long, device=device
                ),
                "num_interior_nodes": V,
                "num_boundary_nodes": 0,
            }

        boundary_vertices_t = torch.stack(boundary_vertices, dim=0)
        boundary_alpha_t = torch.stack(boundary_alpha, dim=0)

        boundary_edges_t = torch.as_tensor(
            boundary_edges,
            dtype=torch.long,
            device=device,
        )

        boundary_seed_pairs_t = torch.as_tensor(
            boundary_seed_pairs,
            dtype=torch.long,
            device=device,
        )

        nodes_uv = torch.cat([vertices_uv, boundary_vertices_t], dim=0)
        node_alpha = torch.cat([alpha, boundary_alpha_t], dim=0)

        node_type = torch.cat(
            [
                torch.zeros(V, dtype=torch.long, device=device),
                torch.ones(boundary_vertices_t.shape[0], dtype=torch.long, device=device),
            ],
            dim=0,
        )

        full_edge_index = torch.cat([edges, boundary_edges_t], dim=0)
        full_edge_seed_pair = torch.cat([edge_seed_pairs, boundary_seed_pairs_t], dim=0)
        full_edge_type = torch.cat((
            torch.zeros(edges.shape[0], dtype=torch.long, device=device),
            torch.ones(boundary_edges_t.shape[0], dtype=torch.long, device=device),
        ))

        full_edge_alpha = (
            node_alpha[full_edge_index[:, 0]]
            * node_alpha[full_edge_index[:, 1]]
        )

        node_degree = self.exact_vertex_degree(
            num_vertices=nodes_uv.shape[0],
            edge_index=full_edge_index,
            dtype=dtype,
            device=device,
        )

        return {
            "nodes_uv": nodes_uv,
            "node_alpha": node_alpha,
            "node_type": node_type,
            "node_degree": node_degree,
            "edge_index": full_edge_index,
            "edge_seed_pair": full_edge_seed_pair,
            "edge_alpha": full_edge_alpha,
            "edge_type": full_edge_type,
            "num_interior_nodes": V,
            "num_boundary_nodes": boundary_vertices_t.shape[0],
        }

    def forward_scipy_topology(
        self,
        seeds_uv: torch.Tensor,
        cad_domain: Any | None = None,
        u_periodic: bool = False,
        v_periodic: bool = False,
        return_xyz: bool | None = None,
        keep_isolated_vertices: bool = False,
    ) -> dict[str, Any]:
        """
        - SciPy builds graph topology without gradients.
        - PyTorch recomputes vertex positions with gradients.
        - Therefore gradients flow through geometry, not topology.
        """
        if not isinstance(seeds_uv, torch.Tensor):
            raise TypeError("seeds_uv must be a torch.Tensor.")
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError(f"seeds_uv must have shape [S, 2], got {tuple(seeds_uv.shape)}.")
        if not seeds_uv.is_floating_point():
            raise TypeError("seeds_uv must be a floating point tensor.")

        want_xyz = self.return_xyz if return_xyz is None else bool(return_xyz)
        # SciPy connectivity is a discrete choice. Rebuild it as needed, but
        # never place topology construction on the autograd graph.
        with torch.no_grad():
            topo = self.build_scipy_voronoi_topology(
                seeds_uv,
                cad_domain=cad_domain,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
        vertices_uv = self.differentiable_vertices_from_topology(
            seeds_uv=seeds_uv,
            vertex_type=topo["vertex_type"],
            vertex_seed_triples=topo["vertex_seed_triples"],
            boundary_origin_vertex=topo["boundary_origin_vertex"],
            boundary_seed_pair=topo["boundary_seed_pair"],
            boundary_ray_dir=topo["boundary_ray_dir"],
            u_periodic=u_periodic,
            v_periodic=v_periodic,
            cad_domain=cad_domain,
            boundary_target_vertex=topo["boundary_target_vertex"],
            boundary_source_type=topo["boundary_source_type"],
        )

        alpha = torch.ones(
            (vertices_uv.shape[0],),
            dtype=seeds_uv.dtype,
            device=seeds_uv.device,
        )
        if cad_domain is not None and self.use_trim_activity:
            alpha = alpha * self.trim_gate(vertices_uv, cad_domain)
        else:
            alpha = alpha * self.box_gate(vertices_uv, u_periodic, v_periodic)

        # Clipped vertices are valid boundary nodes by construction.  A smooth
        # box gate is 0.5 exactly on an edge, which should not deactivate them.
        if cad_domain is None:
            boundary_mask = topo["vertex_type"] == 1
            alpha = torch.where(boundary_mask, torch.ones_like(alpha), alpha)

        num_before_pair_completion = int(vertices_uv.shape[0])
        augmented = self.add_pair_boundary_candidates(
            seeds_uv=seeds_uv,
            nodes_uv=vertices_uv,
            node_alpha=alpha,
            node_type=topo["vertex_type"],
            node_seed_triples=topo["vertex_seed_triples"],
            boundary_seed_pair=topo["boundary_seed_pair"],
            boundary_source_type=topo["boundary_source_type"],
            edges=topo["edges"],
            edge_seed_pairs=topo["edge_seed_pairs"],
            edge_type=topo["edge_type"],
            cad_domain=cad_domain,
            hard_validity=True,
        )
        num_pair_boundary_vertices = int(augmented["nodes_uv"].shape[0]) - num_before_pair_completion
        if cad_domain is None:
            augmented.update(self.add_box_shell_corners(
                nodes_uv=augmented["nodes_uv"],
                node_alpha=augmented["node_alpha"],
                node_type=augmented["node_type"],
                node_seed_triples=augmented["node_seed_triples"],
                boundary_seed_pair=augmented["boundary_seed_pair"],
                boundary_source_type=augmented["boundary_source_type"],
            ))
        num_corner_vertices = (
            int((augmented["boundary_source_type"] == 4).sum().item())
            - int((topo["boundary_source_type"] == 4).sum().item())
        )
        total_added = int(augmented["nodes_uv"].shape[0]) - int(vertices_uv.shape[0])
        vertices_uv = augmented["nodes_uv"]
        alpha = augmented["node_alpha"]
        topo["vertex_type"] = augmented["node_type"]
        topo["vertex_seed_triples"] = augmented["node_seed_triples"]
        topo["boundary_seed_pair"] = augmented["boundary_seed_pair"]
        topo["boundary_source_type"] = augmented["boundary_source_type"]
        topo["edges"] = augmented["edges"]
        topo["edge_seed_pairs"] = augmented["edge_seed_pairs"]
        topo["edge_type"] = augmented["edge_type"]
        if total_added > 0:
            topo["boundary_origin_vertex"] = torch.cat((
                topo["boundary_origin_vertex"],
                torch.full((total_added,), -1, dtype=torch.long, device=seeds_uv.device),
            ))
            topo["boundary_target_vertex"] = torch.cat((
                topo["boundary_target_vertex"],
                torch.full((total_added,), -1, dtype=torch.long, device=seeds_uv.device),
            ))
            topo["boundary_ray_dir"] = torch.cat((
                topo["boundary_ray_dir"],
                torch.zeros((total_added, 2), dtype=seeds_uv.dtype, device=seeds_uv.device),
            ))

        loop_edges, loop_edge_type = self.build_boundary_loop_edges(
            vertices_uv, topo["vertex_type"], cad_domain=cad_domain
        )
        base_edges = topo["edges"]
        edges = torch.cat((base_edges, loop_edges), dim=0)
        edge_type = torch.cat((topo["edge_type"], loop_edge_type), dim=0)
        loop_seed_pairs = torch.full(
            (loop_edges.shape[0], 2), -1, dtype=torch.long, device=seeds_uv.device
        )
        edge_seed_pairs = torch.cat((topo["edge_seed_pairs"], loop_seed_pairs), dim=0)

        pruned = self.prune_graph_vertices(
            nodes_uv=vertices_uv,
            vertex_type=topo["vertex_type"],
            vertex_seed_triples=topo["vertex_seed_triples"],
            boundary_origin_vertex=topo["boundary_origin_vertex"],
            boundary_target_vertex=topo["boundary_target_vertex"],
            boundary_seed_pair=topo["boundary_seed_pair"],
            boundary_ray_dir=topo["boundary_ray_dir"],
            boundary_source_type=topo["boundary_source_type"],
            edges=edges,
            edge_seed_pairs=edge_seed_pairs,
            edge_type=edge_type,
            alpha=alpha,
            keep_isolated_vertices=keep_isolated_vertices,
        )
        inactive_vertex_ids = torch.nonzero(
            pruned["old_to_new"] < 0, as_tuple=False
        ).flatten()
        pruned_vertices_uv = vertices_uv[inactive_vertex_ids]
        pruned_vertex_type = topo["vertex_type"][inactive_vertex_ids]
        vertices_uv = pruned["nodes_uv"]
        alpha = pruned["alpha"]
        edges = pruned["edges"]
        edge_seed_pairs = pruned["edge_seed_pairs"]
        edge_type = pruned["edge_type"]
        for key in (
            "vertex_type",
            "vertex_seed_triples",
            "boundary_origin_vertex",
            "boundary_target_vertex",
            "boundary_seed_pair",
            "boundary_ray_dir",
            "boundary_source_type",
        ):
            topo[key] = pruned[key]

        old_to_new = pruned["old_to_new"]
        boundary_rays = topo["boundary_rays"].clone()
        if boundary_rays.numel() > 0:
            mapped_ray_origins = old_to_new[boundary_rays[:, 0]]
            keep_rays = mapped_ray_origins >= 0
            boundary_rays = boundary_rays[keep_rays]
            boundary_rays[:, 0] = mapped_ray_origins[keep_rays]
            boundary_ray_dirs = topo["boundary_ray_dirs"][keep_rays]
        else:
            boundary_ray_dirs = topo["boundary_ray_dirs"]

        diagnostics = dict(topo["diagnostics"])
        diagnostics["num_raw_boundary_vertices"] = (
            diagnostics.get("num_raw_boundary_vertices", 0) + total_added
        )
        diagnostics.update({
            "num_pair_boundary_vertices": num_pair_boundary_vertices,
            "num_corner_shell_vertices": num_corner_vertices,
            "num_pruned_vertices": pruned["num_pruned_vertices"],
            "num_final_vertices": int(vertices_uv.shape[0]),
            "num_final_interior_vertices": int((topo["vertex_type"] == 0).sum().item()),
            "num_final_boundary_vertices": int((topo["vertex_type"] == 1).sum().item()),
        })
        topo["diagnostics"] = diagnostics

        if edges.numel() == 0:
            edge_alpha = torch.empty((0,), dtype=seeds_uv.dtype, device=seeds_uv.device)
        else:
            edge_alpha = alpha[edges[:, 0]] * alpha[edges[:, 1]]

        vertex_degree = self.exact_vertex_degree(
            num_vertices=vertices_uv.shape[0],
            edge_index=edges,
            dtype=vertices_uv.dtype,
            device=vertices_uv.device,
        )

        active_interior = topo["vertex_type"] == 0
        num_interior = int(active_interior.sum().item())
        num_boundary = int((topo["vertex_type"] == 1).sum().item())
        graph = {
            "nodes_uv": vertices_uv,
            "node_alpha": alpha,
            "node_type": topo["vertex_type"],
            "node_degree": vertex_degree,
            "edge_index": edges,
            "edge_seed_pair": edge_seed_pairs,
            "edge_alpha": edge_alpha,
            "edge_type": edge_type,
            "vertex_degree": vertex_degree,
            "boundary_source_type": topo["boundary_source_type"],
            "boundary_source_name": [
                {0: "interior", 1: "infinite_ray_clipping", 2: "finite_edge_clipping",
                 3: "pair_bisector_boundary", 4: "corner_shell"}.get(int(value), "unknown")
                for value in topo["boundary_source_type"].detach().cpu().tolist()
            ],
            "diagnostics": topo["diagnostics"],
            "num_interior_nodes": num_interior,
            "num_boundary_nodes": num_boundary,
        }

        out: dict[str, Any] = {
            "vertices_uv": vertices_uv,
            "alpha": alpha,
            "triple_idx": topo["vertex_seed_triples"],
            "vertex_type": topo["vertex_type"],
            "vertex_seed_triples": topo["vertex_seed_triples"],
            "boundary_origin_vertex": topo["boundary_origin_vertex"],
            "boundary_target_vertex": topo["boundary_target_vertex"],
            "boundary_seed_pair": topo["boundary_seed_pair"],
            "boundary_ray_dir": topo["boundary_ray_dir"],
            "boundary_source_type": topo["boundary_source_type"],
            "boundary_source_name": graph["boundary_source_name"],
            "edges": {
                "edge_index": edges,
                "edge_seed_pair": edge_seed_pairs,
                "edge_alpha": edge_alpha,
                "vertex_degree": vertex_degree,
                "edge_type": edge_type,
            },
            "boundary_rays": boundary_rays,
            "boundary_ray_dirs": boundary_ray_dirs,
            "scipy_vertices_np": topo["scipy_vertices_np"],
            "pruned_vertices_uv": pruned_vertices_uv,
            "pruned_vertex_type": pruned_vertex_type,
            "isolated_vertices": (
                torch.nonzero(vertex_degree == 0, as_tuple=False).flatten()
                if keep_isolated_vertices
                else torch.empty((0,), dtype=torch.long, device=seeds_uv.device)
            ),
            "delaunay_triples_np": topo["delaunay_triples_np"],
            "mode": "scipy_topology",
            "vertex_degree": vertex_degree,
            "graph": graph,
            "diagnostics": topo["diagnostics"],
        }
        out.update(topo["diagnostics"])

        if edges.numel() > 0:
            # SciPy connectivity is discrete, but these sampled curve
            # coordinates remain differentiable with respect to the seeds.
            out["edge_curves_uv"] = self.sample_graph_edge_curves_uv(
                seeds_uv=seeds_uv,
                graph=graph,
                n_samples=64,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
            if (
                cad_domain is not None
                and want_xyz
                and callable(getattr(cad_domain, "eval_uv_norm_batch_torch", None))
            ):
                out["edge_curves_xyz"] = self.sample_smooth_edge_curves_xyz(
                    cad_domain, out["edge_curves_uv"]
                )

        if cad_domain is not None and want_xyz:
            xyz = cad_domain.eval_uv_norm_batch(vertices_uv, return_inside_mask=False)["xyz"]
            out["vertices_xyz"] = torch.as_tensor(xyz, dtype=seeds_uv.dtype, device=seeds_uv.device)

        return out

    def forward(
        self,
        seeds_uv: torch.Tensor, #Tensor of seed points in normalized UV space, shape [S, 2].
        seed_activity: torch.Tensor | None = None, #Optional tensor of seed activity values, shape [S]
        cad_domain: Any | None = None, #Optional CAD/domain object used for trimming and UV-to-XYZ conversion.
        u_periodic: bool = False,
        v_periodic: bool = False,
        return_xyz: bool | None = None,
        debug_compare_scipy: bool = False,
        topology_mode: str = "soft",
        keep_isolated_vertices: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(seeds_uv, torch.Tensor):
            raise TypeError("seeds_uv must be a torch.Tensor.")
        if seeds_uv.ndim != 2 or seeds_uv.shape[-1] != 2:
            raise ValueError(f"seeds_uv must have shape [S, 2], got {tuple(seeds_uv.shape)}.")
        if not seeds_uv.is_floating_point():
            raise TypeError("seeds_uv must be a floating point tensor.")

        if topology_mode == "scipy":
            return self.forward_scipy_topology(
                seeds_uv=seeds_uv,
                cad_domain=cad_domain,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
                return_xyz=return_xyz,
                keep_isolated_vertices=keep_isolated_vertices,
            )
        if topology_mode != "soft":
            raise ValueError(
                f"topology_mode must be 'soft' or 'scipy', got {topology_mode!r}."
            )

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
        g_domain = g_trim if cad_domain is not None and self.use_trim_activity else g_box
        g_vor = self.empty_circle_gate(seeds_uv, P_uv, triples, u_periodic, v_periodic)

        if debug_compare_scipy:
            g_seed = torch.ones_like(g_vor)
            g_domain = g_box

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
            alpha_base = alpha_base * g_close
        g_compete = self.vertex_soft_competition_gate(
            P_uv,
            alpha_base,
            sigma=0.005,
            temperature=0.08,
            floor=0.2,
        )

        alpha = alpha_base * g_compete

        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        P_uv = torch.nan_to_num(P_uv, nan=0.0, posinf=0.0, neginf=0.0)

        edge_data = self.build_soft_edges(
    vertices_uv=P_uv,
    alpha=alpha,
    triples=triples,
)

        soft_degree = self.soft_vertex_degree(
            num_vertices=P_uv.shape[0],
            edge_index=edge_data["edge_index"],
            edge_alpha=edge_data["edge_alpha"],
            dtype=P_uv.dtype,
            device=P_uv.device,
        )

        edge_data["soft_vertex_degree"] = soft_degree

        topk = min(5, seeds_uv.shape[0])
        topk_weights, topk_seed_idx = torch.topk(vertex_seed_weights, k=topk, dim=1)

        validity = {
            "seed": g_seed,
            "close": g_close,
            "area": g_area,
            "box": g_box,
            "trim": g_trim,
            "domain": g_domain,
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
            "alpha_base": alpha_base,
            "vertex_competition_gate": g_compete,
        }
        diagnostics.update({
            "seed_activity_weights": seed_state["weights"],
            "seed_domain_activity": seed_state["domain_activity"],
            "seed_duplicate_weight": seed_state["duplicate_weight"],
            "seed_sdf": seed_state["seed_sdf"],
            "triple_seed_activity": g_seed,

        })

        graph = self.build_soft_completed_graph(
            seeds_uv=seeds_uv,
            vertices_uv=P_uv,
            alpha=alpha,
            triples=triples,
            edge_data=edge_data,
            seed_activity=act,
            cad_domain=cad_domain,
        )
        diagnostics.update({
            "num_pair_boundary_vertices": int(
                (graph["boundary_source_type"] == 3).sum().item()
            ),
            "num_corner_shell_vertices": int(
                (graph["boundary_source_type"] == 4).sum().item()
            ),
        })

        out: dict[str, Any] = {
            "vertices_uv": P_uv,
            "alpha": alpha,
            "triple_idx": triples,
            "validity": validity,
            "diagnostics": diagnostics,
            "edges" : edge_data,
            "vertex_degree": soft_degree,
            "graph": graph,
            "vertex_type": graph["node_type"],
            "boundary_source_type": graph["boundary_source_type"],
            "boundary_source_name": graph["boundary_source_name"],
            "mode": "soft",
        }

        edge_index = graph["edge_index"]
        edge_seed_pairs = graph["edge_seed_pair"]
        if edge_index.numel() > 0:
            # Graph topology is discrete; Hermite endpoint/tangent geometry is
            # pure Torch and therefore differentiable with respect to seeds.
            out["edge_curves_uv"] = self.sample_graph_edge_curves_uv(
                seeds_uv=seeds_uv,
                graph=graph,
                n_samples=64,
                u_periodic=u_periodic,
                v_periodic=v_periodic,
            )
            if (
                cad_domain is not None
                and want_xyz
                and callable(getattr(cad_domain, "eval_uv_norm_batch_torch", None))
            ):
                out["edge_curves_xyz"] = self.sample_smooth_edge_curves_xyz(
                    cad_domain, out["edge_curves_uv"]
                )

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
            "domain": empty,
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

        edge_data= {
    "edge_index": torch.empty((0, 2), dtype=torch.long, device=seeds_uv.device),
    "edge_alpha": empty,
    "edge_seed_pair": torch.empty((0, 2), dtype=torch.long, device=seeds_uv.device),
}
        graph = self.build_soft_completed_graph(
            seeds_uv=seeds_uv,
            vertices_uv=vertices_uv,
            alpha=empty,
            triples=triples,
            edge_data=edge_data,
            seed_activity=seed_state["weights"],
            cad_domain=cad_domain,
        )
        diagnostics.update({
            "num_pair_boundary_vertices": int(
                (graph["boundary_source_type"] == 3).sum().item()
            ),
            "num_corner_shell_vertices": int(
                (graph["boundary_source_type"] == 4).sum().item()
            ),
        })
        out: dict[str, Any] = {
            "vertices_uv": vertices_uv,
            "alpha": empty,
            "triple_idx": triples,
            "validity": validity,
            "diagnostics": diagnostics,
            "edges" :edge_data,
            "graph": graph,
            "vertex_type": graph["node_type"],
            "boundary_source_type": graph["boundary_source_type"],
            "boundary_source_name": graph["boundary_source_name"],
            "mode": "soft",
        }
        if cad_domain is not None and want_xyz:
            out["vertices_xyz"] = torch.empty((0, 3), dtype=seeds_uv.dtype, device=seeds_uv.device)
        return out

    def build_soft_edges(
    self,
    vertices_uv: torch.Tensor,
    alpha: torch.Tensor,
    triples: torch.Tensor,
    min_edge_alpha: float = 0.0,
):
        T = triples.shape[0]
        if T < 2:
            return {
                "edge_index": torch.empty((0, 2), dtype=torch.long, device=vertices_uv.device),
                "edge_alpha": torch.empty((0,), dtype=vertices_uv.dtype, device=vertices_uv.device),
                "edge_seed_pair": torch.empty((0, 2), dtype=torch.long, device=vertices_uv.device),
            }

        edges = []
        edge_seed_pairs = []

        for a in range(T):
            set_a = set(map(int, triples[a].detach().cpu().tolist()))
            for b in range(a + 1, T):
                set_b = set(map(int, triples[b].detach().cpu().tolist()))
                shared = sorted(list(set_a.intersection(set_b)))
                if len(shared) == 2:
                    edges.append([a, b])
                    edge_seed_pairs.append(shared)

        if len(edges) == 0:
            return {
                "edge_index": torch.empty((0, 2), dtype=torch.long, device=vertices_uv.device),
                "edge_alpha": torch.empty((0,), dtype=vertices_uv.dtype, device=vertices_uv.device),
                "edge_seed_pair": torch.empty((0, 2), dtype=torch.long, device=vertices_uv.device),
            }

        edge_index = torch.tensor(edges, dtype=torch.long, device=vertices_uv.device)
        edge_seed_pair = torch.tensor(edge_seed_pairs, dtype=torch.long, device=vertices_uv.device)

        edge_alpha = alpha[edge_index[:, 0]] * alpha[edge_index[:, 1]]

        if min_edge_alpha > 0:
            keep = edge_alpha > min_edge_alpha
            edge_index = edge_index[keep]
            edge_alpha = edge_alpha[keep]
            edge_seed_pair = edge_seed_pair[keep]

        return {
            "edge_index": edge_index,
            "edge_alpha": edge_alpha,
            "edge_seed_pair": edge_seed_pair,
        }

    def soft_vertex_degree(self, num_vertices, edge_index, edge_alpha, dtype, device):
        deg = torch.zeros(num_vertices, dtype=dtype, device=device)

        if edge_index.numel() == 0:
            return deg

        deg = deg.scatter_add(0, edge_index[:, 0], edge_alpha)
        deg = deg.scatter_add(0, edge_index[:, 1], edge_alpha)


        return deg

    def exact_vertex_degree(self, num_vertices, edge_index, dtype, device):
        degree = torch.zeros(num_vertices, dtype=dtype, device=device)

        if edge_index.numel() == 0:
            return degree

        one = torch.ones(edge_index.shape[0], dtype=dtype, device=device)

        degree = degree.scatter_add(0, edge_index[:, 0], one)
        degree = degree.scatter_add(0, edge_index[:, 1], one)

        return degree
    
    def plot_scipy_topology_output(
    self,
    seeds_uv,
    out=None,
    cad_domain=None,
    show_degree=True,
    show_boundary_rays=True,
):
        if out is None:
            out = self(
                seeds_uv,
                topology_mode="scipy",
                cad_domain=cad_domain,
                return_xyz=False,
            )

        seeds_np = seeds_uv.detach().cpu().numpy()
        vertices = out["vertices_uv"].detach().cpu().numpy()
        alpha = out["alpha"].detach().cpu().numpy()
        edges = out["edges"]["edge_index"].detach().cpu().numpy()
        edge_types = out["edges"]["edge_type"].detach().cpu().numpy()
        vertex_types = out["vertex_type"].detach().cpu().numpy()
        boundary_sources = out["boundary_source_type"].detach().cpu().numpy()
        boundary_rays = out["boundary_rays"].detach().cpu().numpy()
        boundary_ray_dirs = out.get("boundary_ray_dirs", None)
        degree = out.get("vertex_degree", None)

        if degree is not None:
            degree_np = degree.detach().cpu().numpy()
        else:
            degree_np = None

        fig, ax = plt.subplots(figsize=(8, 8))

        for e, edge_type in zip(edges, edge_types):
            a, b = int(e[0]), int(e[1])
            ax.plot(
                [vertices[a, 0], vertices[b, 0]],
                [vertices[a, 1], vertices[b, 1]],
                color="tab:blue" if edge_type == 2 else "black",
                linestyle="--" if edge_type == 2 else "-",
                linewidth=1.0,
                alpha=0.7,
            )

        if show_boundary_rays and len(boundary_rays) > 0:
            if boundary_ray_dirs is not None:
                boundary_ray_dirs_np = boundary_ray_dirs.detach().cpu().numpy()
            else:
                boundary_ray_dirs_np = None

            for ray_idx, ray in enumerate(boundary_rays):
                v_id = int(ray[0])
                p = vertices[v_id]

                if boundary_ray_dirs_np is not None:
                    direction = boundary_ray_dirs_np[ray_idx]
                else:
                    i = int(ray[1])
                    j = int(ray[2])
                    n = seeds_np[j] - seeds_np[i]
                    direction = np.array([-n[1], n[0]], dtype=float)
                    direction = direction / (np.linalg.norm(direction) + 1e-12)

                q = p + 0.2 * direction

                ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    color="purple",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                )

        interior = vertex_types == 0
        ray_boundary = (vertex_types == 1) & (boundary_sources == 1)
        finite_boundary = (vertex_types == 1) & (boundary_sources == 2)
        pair_boundary = (vertex_types == 1) & (boundary_sources == 3)
        corner_boundary = (vertex_types == 1) & (boundary_sources == 4)
        sc = ax.scatter(
            vertices[interior, 0],
            vertices[interior, 1],
            c=alpha[interior],
            s=90,
            cmap="viridis",
            vmin=0,
            vmax=1,
            label="Voronoi vertices",
            zorder=3,
        )
        ax.scatter(
            vertices[ray_boundary, 0],
            vertices[ray_boundary, 1],
            c=alpha[ray_boundary],
            marker="D",
            edgecolors="purple",
            s=75,
            cmap="viridis",
            vmin=0,
            vmax=1,
            label="Ray-clipped boundary vertices",
            zorder=4,
        )
        ax.scatter(
            vertices[finite_boundary, 0],
            vertices[finite_boundary, 1],
            c=alpha[finite_boundary],
            marker="s",
            edgecolors="tab:cyan",
            s=75,
            cmap="viridis",
            vmin=0,
            vmax=1,
            label="Finite-edge boundary vertices",
            zorder=4,
        )
        ax.scatter(
            vertices[pair_boundary, 0], vertices[pair_boundary, 1],
            c=alpha[pair_boundary], marker="D", edgecolors="tab:orange", s=80,
            cmap="viridis", vmin=0, vmax=1,
            label="Pair-bisector boundary vertices", zorder=4,
        )
        ax.scatter(
            vertices[corner_boundary, 0], vertices[corner_boundary, 1],
            c="gold", marker="s", edgecolors="black", s=85,
            label="Corner shell vertices", zorder=4,
        )

        ax.scatter(
            seeds_np[:, 0],
            seeds_np[:, 1],
            c="red",
            s=60,
            label="Seeds",
            zorder=4,
        )

        if show_degree and degree_np is not None:
            for p, d in zip(vertices, degree_np):
                ax.text(
                    p[0] + 0.01,
                    p[1] + 0.01,
                    f"d={int(round(d))}",
                    fontsize=10,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.85,
                        boxstyle="round,pad=0.2",
                    ),
                    zorder=5,
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title("SciPy topology + differentiable PyTorch vertices")
        ax.legend()
        fig.colorbar(sc, ax=ax, label="alpha")
        plt.show()

        print("mode:", out["mode"])
        print("vertices:", out["vertices_uv"].shape)
        print("edges:", out["edges"]["edge_index"].shape)
        print("boundary rays:", out["boundary_rays"].shape)
        print("clipping diagnostics:", out["diagnostics"])

        return fig, ax

    @staticmethod
    def _generated_graph_edge_style(edge_type: int) -> tuple[str, str, str]:
        styles = {
            0: ("black", "-", "Interior Voronoi edge"),
            1: ("tab:orange", "-", "Clipped interior-boundary Voronoi edge"),
            2: ("0.5", ":", "Reserved edge type"),
            3: ("tab:orange", "-", "Clipped boundary-boundary Voronoi edge"),
            4: ("tab:cyan", "--", "Boundary shell edge"),
        }
        return styles.get(int(edge_type), ("0.5", ":", f"Edge type {edge_type}"))

    def _draw_generated_graph(
        self,
        ax,
        seeds_uv,
        out,
        show_node_ids: bool = True,
        show_edge_ids: bool = False,
        node_id_fontsize: int = 9,
        show_pruned_nodes: bool = False,
        color_by_edge_type: bool = True,
    ):
        """Draw only the compact graph represented by ``out['graph']``."""
        graph = out["graph"]
        nodes = graph["nodes_uv"].detach().cpu().numpy()
        edges = graph["edge_index"].detach().cpu().numpy()
        node_types = graph["node_type"].detach().cpu().numpy()
        source_types = graph.get("boundary_source_type", graph["node_type"])
        source_types = source_types.detach().cpu().numpy()
        edge_types_t = graph.get("edge_type")
        if edge_types_t is None:
            edge_types = np.zeros((len(edges),), dtype=np.int64)
        else:
            edge_types = edge_types_t.detach().cpu().numpy()

        for edge_id, ((source, target), edge_type) in enumerate(zip(edges, edge_types)):
            source, target = int(source), int(target)
            color, linestyle, _ = (
                self._generated_graph_edge_style(int(edge_type))
                if color_by_edge_type
                else ("black", "-", "Graph edge")
            )
            ax.plot(
                [nodes[source, 0], nodes[target, 0]],
                [nodes[source, 1], nodes[target, 1]],
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                alpha=0.85,
                zorder=1,
            )
            if show_edge_ids:
                midpoint = 0.5 * (nodes[source] + nodes[target])
                ax.text(
                    midpoint[0], midpoint[1], f"e{edge_id}", fontsize=node_id_fontsize - 1,
                    color=color, ha="center", va="center", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
                )

        interior = node_types == 0
        coordinate_corners = (
            (np.isclose(nodes[:, 0], 0.0, atol=1e-4) | np.isclose(nodes[:, 0], 1.0, atol=1e-4))
            & (np.isclose(nodes[:, 1], 0.0, atol=1e-4) | np.isclose(nodes[:, 1], 1.0, atol=1e-4))
        )
        corners = (node_types == 1) & ((source_types == 4) | coordinate_corners)
        boundary = (node_types == 1) & ~corners
        ax.scatter(
            nodes[interior, 0], nodes[interior, 1], c="orange", marker="o",
            edgecolors="black", s=80, label="Interior nodes", zorder=3,
        )
        ax.scatter(
            nodes[boundary, 0], nodes[boundary, 1], c="tab:cyan", marker="D",
            edgecolors="black", s=70, label="Boundary nodes", zorder=3,
        )
        ax.scatter(
            nodes[corners, 0], nodes[corners, 1], c="gold", marker="s",
            edgecolors="black", s=85, label="Corner shell nodes", zorder=3,
        )
        seeds_np = seeds_uv.detach().cpu().numpy()
        ax.scatter(
            seeds_np[:, 0], seeds_np[:, 1], c="red", marker="o", s=45,
            label="Seeds", zorder=4,
        )

        if show_node_ids:
            for node_id, (point, node_type) in enumerate(zip(nodes, node_types)):
                if int(node_type) == 0:
                    prefix = "I"
                elif bool(corners[node_id]):
                    prefix = "C"
                else:
                    prefix = "B"
                ax.annotate(
                    f"{prefix}{node_id}", xy=point, xytext=(5, 5),
                    textcoords="offset points", fontsize=node_id_fontsize,
                    color="black", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
                )

        if show_pruned_nodes:
            pruned = out.get("pruned_vertices_uv")
            if pruned is not None and pruned.numel() > 0:
                pruned_np = pruned.detach().cpu().numpy()
                ax.scatter(
                    pruned_np[:, 0], pruned_np[:, 1], marker="x", c="0.45", s=65,
                    label="Pruned nodes", zorder=2,
                )

        edge_handles = []
        for edge_type in range(5):
            color, linestyle, label = self._generated_graph_edge_style(edge_type)
            edge_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, label=label))
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles + edge_handles, labels + [item.get_label() for item in edge_handles])

        num_nodes = len(nodes)
        num_interior = int(interior.sum())
        num_boundary = int((node_types == 1).sum())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(
            f"Soft Geometry Graph \n"
            f"nodes={num_nodes} (interior={num_interior}, "
            f"boundary={num_boundary}), edges={len(edges)}\n"
        )
        return ax

    @staticmethod
    def _print_generated_graph_tables(out) -> None:
        graph = out["graph"]
        nodes = graph["nodes_uv"].detach().cpu().numpy()
        node_types = graph["node_type"].detach().cpu().numpy()
        degrees = graph["node_degree"].detach().cpu().numpy()
        alpha = graph["node_alpha"].detach().cpu().numpy()
        edges = graph["edge_index"].detach().cpu().numpy()
        edge_types = graph["edge_type"].detach().cpu().numpy()
        seed_pairs = graph["edge_seed_pair"].detach().cpu().numpy()

        print("\nNode table")
        print("node_id  node_type       u          v      degree      alpha")
        for node_id, (point, node_type, degree, activity) in enumerate(
            zip(nodes, node_types, degrees, alpha)
        ):
            type_name = "interior" if int(node_type) == 0 else "boundary"
            print(
                f"{node_id:7d}  {type_name:9s}  {point[0]:9.6f}  {point[1]:9.6f}  "
                f"{degree:8.3f}  {activity:9.6f}"
            )

        print("\nEdge table")
        print("edge_id  source  target  edge_type  seed_pair")
        for edge_id, (edge, edge_type, seed_pair) in enumerate(
            zip(edges, edge_types, seed_pairs)
        ):
            print(
                f"{edge_id:7d}  {int(edge[0]):6d}  {int(edge[1]):6d}  "
                f"{int(edge_type):9d}  ({int(seed_pair[0])}, {int(seed_pair[1])})"
            )

    def plot_graph_output(
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
            out = self(seeds_uv, topology_mode="scipy", cad_domain=cad_domain, return_xyz=False)
        fig, ax = plt.subplots(figsize=(8, 8))
        self._draw_generated_graph(
            ax, seeds_uv, out, show_node_ids, show_edge_ids,
            node_id_fontsize, show_pruned_nodes, color_by_edge_type,
        )
        plt.show()
        if print_node_table:
            self._print_generated_graph_tables(out)
        return fig, ax

    def plot_generated_graph_debug(
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
        """Plot the generated graph abstraction without a SciPy background."""
        return self.plot_graph_output(
            seeds_uv=seeds_uv,
            out=out,
            cad_domain=cad_domain,
            show_node_ids=show_node_ids,
            show_edge_ids=show_edge_ids,
            node_id_fontsize=node_id_fontsize,
            print_node_table=print_node_table,
            show_pruned_nodes=show_pruned_nodes,
            color_by_edge_type=color_by_edge_type,
        )

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
            out = self(seeds_uv, topology_mode="scipy", cad_domain=cad_domain, return_xyz=False)

        seeds_np = seeds_uv.detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
        left, middle = axes
        try:
            raw_voronoi = Voronoi(seeds_np)
            voronoi_plot_2d(
                raw_voronoi, ax=left, show_vertices=False, show_points=False,
                line_colors="black", line_width=1.0, line_alpha=0.75, point_size=0,
            )
            if raw_voronoi.vertices.size > 0:
                left.scatter(
                    raw_voronoi.vertices[:, 0], raw_voronoi.vertices[:, 1],
                    marker="x", c="0.35", s=55, label="Raw SciPy vertices", zorder=3,
                )
        except Exception as error:
            left.text(0.5, 0.5, f"SciPy Voronoi unavailable\n{error}", ha="center", va="center")
        left.scatter(seeds_np[:, 0], seeds_np[:, 1], c="red", s=45, label="Seeds", zorder=4)
        left.set_xlim(0, 1)
        left.set_ylim(0, 1)
        left.set_aspect("equal")
        left.set_title( f"VD for {seeds_uv.shape[0]} \n""Raw SciPy Voronoi (UV clipped view)\n")
        left.legend()

        self._draw_generated_graph(
            middle, seeds_uv, out, show_node_ids, show_edge_ids,
            node_id_fontsize, show_pruned_nodes, color_by_edge_type,
        )
        middle.set_facecolor("none")
        middle.patch.set_alpha(0.0)
        plt.show()
        if print_node_table:
            self._print_generated_graph_tables(out)
        return fig, axes

    def _draw_graph_connectivity(
        self,
        ax,
        out,
        show_node_ids: bool = True,
        show_edge_ids: bool = False,
        node_id_fontsize: int = 9,
        node_size: float = 100.0,
        edge_width: float = 2.0,
    ):
        """Draw topology only, preserving the final graph's UV shape."""
        graph = out["graph"]
        nodes_uv = graph["nodes_uv"].detach().cpu().numpy()
        edges = graph["edge_index"].detach().cpu().numpy()
        node_types = graph["node_type"].detach().cpu().numpy()
        num_nodes = len(nodes_uv)

        # Deliberately omit seeds, CAD boundaries, edge classes, and geometric
        # decorations: this panel is only the graph incidence structure.
        for edge_id, (source, target) in enumerate(edges):
            source, target = int(source), int(target)
            ax.plot(
                [nodes_uv[source, 0], nodes_uv[target, 0]],
                [nodes_uv[source, 1], nodes_uv[target, 1]],
                color="black",
                linestyle="-",
                linewidth=edge_width,
                alpha=0.8,
                zorder=1,
            )
            if show_edge_ids:
                midpoint = 0.5 * (nodes_uv[source] + nodes_uv[target])
                ax.text(
                    midpoint[0], midpoint[1], f"e{edge_id}",
                    fontsize=max(node_id_fontsize - 1, 1), color="black",
                    ha="center", va="center", zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.4),
                )

        interior = node_types == 0
        boundary = node_types == 1
        ax.scatter(
            nodes_uv[interior, 0], nodes_uv[interior, 1], c="white",
            marker="o", edgecolors="black", linewidths=1.5,
            s=node_size, label="Interior nodes", zorder=3,
        )
        ax.scatter(
            nodes_uv[boundary, 0], nodes_uv[boundary, 1], c="0.75",
            marker="o", edgecolors="black", linewidths=1.5,
            s=node_size, label="Boundary nodes", zorder=3,
        )

        if show_node_ids:
            for node_id, point in enumerate(nodes_uv):
                ax.text(
                    point[0], point[1], str(node_id),
                    fontsize=node_id_fontsize, ha="center", va="center", zorder=5,
                )

        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            "Topology-only graph\n"
            f"nodes={num_nodes}, edges={len(edges)}"
        )
        return ax



    def plot_voronoi_debug(
        self,
        seeds_uv,
        out=None,
        cad_domain=None,
        alpha_threshold=0.5,
        trim_res=300,
        miss_tol=0.02,
        show_degree_labels=True,
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

        soft_degree = out.get("vertex_degree", None)
        if soft_degree is not None:
            pred_degree_np = soft_degree[keep].detach().cpu().numpy()
            degree_np = soft_degree.detach().cpu().numpy()
        else:
            pred_degree_np = None
            degree_np = None

        vor = Voronoi(seeds_np)
        exact_np = vor.vertices

        inside_exact = (
            (exact_np[:, 0] >= 0.0) & (exact_np[:, 0] <= 1.0) &
            (exact_np[:, 1] >= 0.0) & (exact_np[:, 1] <= 1.0)
        )

        if cad_domain is not None and hasattr(cad_domain, "smooth_inside_activity"):
            exact_t = torch.as_tensor(
                exact_np,
                dtype=seeds_uv.dtype,
                device=seeds_uv.device,
            )
            trim_exact = cad_domain.smooth_inside_activity(
                exact_t,
                tau=self.tau_trim,
            )
            inside_exact = inside_exact & (
                trim_exact.detach().cpu().numpy() > alpha_threshold
            )

        exact_inside_np = exact_np[inside_exact]

        fig, axes = plt.subplots(1, 2, figsize=(24, 8))

        def draw_trim(ax):
            if cad_domain is None or not hasattr(cad_domain, "smooth_inside_activity"):
                return

            u = torch.linspace(
                0,
                1,
                trim_res,
                device=seeds_uv.device,
                dtype=seeds_uv.dtype,
            )
            v = torch.linspace(
                0,
                1,
                trim_res,
                device=seeds_uv.device,
                dtype=seeds_uv.dtype,
            )

            uu, vv = torch.meshgrid(u, v, indexing="xy")
            grid = torch.stack(
                [uu.reshape(-1), vv.reshape(-1)],
                dim=-1,
            )

            activity = cad_domain.smooth_inside_activity(
                grid,
                tau=self.tau_trim,
            )

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

        # ------------------------------------------------------------
        # 1) Predicted vertices colored by alpha
        # ------------------------------------------------------------
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

        ax.scatter(
            seeds_np[:, 0],
            seeds_np[:, 1],
            c="red",
            s=50,
            label="Seeds",
        )

        ax.set_title("Predicted vertices colored by validity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend()
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="alpha")


        # ------------------------------------------------------------
        # 3) Thresholded predicted vertices + degree labels
        # ------------------------------------------------------------
        ax = axes[1]
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

        ax.scatter(
            seeds_np[:, 0],
            seeds_np[:, 1],
            c="red",
            s=50,
            label="Seeds",
        )

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

        if show_degree_labels and pred_degree_np is not None:
            for p, deg in zip(pred_np, pred_degree_np):
                ax.text(
                    p[0] + 0.01,
                    p[1] + 0.01,
                    f"d={deg:.2f}",
                    fontsize=12,
                    color="black",
                    ha="left",
                    va="bottom",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.85,
                        boxstyle="round,pad=0.2",
                    ),
                    zorder=10,
                )

        ax.set_title("Thresholded prediction with soft degree")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend()

        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------
        # Print comparison stats
        # ------------------------------------------------------------
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

            if pred_degree_np is not None:
                print("Predicted soft degrees:")
                print(pred_degree_np)

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

                    if degree_np is not None:
                        print(f"soft_degree     = {degree_np[cand_idx]:.6f}")

                    for name, value in validity.items():
                        if (
                            torch.is_tensor(value)
                            and value.ndim > 0
                            and value.shape[0] > cand_idx
                        ):
                            print(
                                f"gate {name:>12s} = "
                                f"{value[cand_idx].detach().cpu().item():.6e}"
                            )

                    if "vertex_keff" in diagnostics:
                        keff = diagnostics["vertex_keff"]
                        if (
                            torch.is_tensor(keff)
                            and keff.ndim > 0
                            and keff.shape[0] > cand_idx
                        ):
                            print(
                                f"vertex_keff     = "
                                f"{keff[cand_idx].detach().cpu().item():.6f}"
                            )

                    if "alpha_base" in diagnostics:
                        ab = diagnostics["alpha_base"]
                        if (
                            torch.is_tensor(ab)
                            and ab.ndim > 0
                            and ab.shape[0] > cand_idx
                        ):
                            print(
                                f"alpha_base      = "
                                f"{ab[cand_idx].detach().cpu().item():.6e}"
                            )

                    if "vertex_competition_gate" in diagnostics:
                        cg = diagnostics["vertex_competition_gate"]
                        if (
                            torch.is_tensor(cg)
                            and cg.ndim > 0
                            and cg.shape[0] > cand_idx
                        ):
                            print(
                                f"competition     = "
                                f"{cg[cand_idx].detach().cpu().item():.6e}"
                            )
            else:
                print(f"\nNo missing exact vertices with miss_tol={miss_tol}.")

        else:
            print("Cannot compare: one set is empty.")
            print("Predicted active vertices:", len(pred_np))
            print("Exact inside vertices:", len(exact_inside_np))
