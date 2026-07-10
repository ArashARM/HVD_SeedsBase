"""Gradient-validation helpers for the hybrid Voronoi decoder.

The decoder is piecewise differentiable: SciPy/Qhull chooses discrete Voronoi
topology, then PyTorch reconstructs the fixed local geometry branch. These
tests deliberately do not differentiate through SciPy. Frozen-topology checks
validate the differentiable branch; full-decoder checks rebuild topology and
classify topology changes as expected discrete events, not gradient failures.

Training-loop topology monitor example:

    tester = DecoderGradientTester(decoder, cad_domain)
    topology_monitor = TopologyChangeMonitor(tester)

    if step % topology_check_every == 0:
        event = topology_monitor.update(step, seeds_uv)
        if event["changed"]:
            print(f"Topology changed at step {step}: {event['previous_signature']} -> {event['current_signature']}")
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import math
from collections import Counter
from typing import Any, Callable, Iterable

import numpy as np
import torch

try:  # SciPy is optional for recognizing Qhull failures more explicitly.
    from scipy.spatial import QhullError
except Exception:  # pragma: no cover - depends on local SciPy install.
    QhullError = RuntimeError  # type: ignore[misc,assignment]


LossFn = Callable[[dict[str, Any]], torch.Tensor | tuple[torch.Tensor, dict[str, Any]]]


class DecoderGradientTester:
    """Reusable gradient tester for hybrid SciPy/PyTorch Voronoi decoders."""

    TOPOLOGY_BUILDER_NAMES = (
        "build_scipy_voronoi_topology",
        "build_voronoi_topology",
        "compute_voronoi_topology",
        "create_voronoi_topology",
    )
    DISCRETE_SIGNATURE_KEYS = (
        "edges",
        "edge_index",
        "edge_type",
        "edge_types",
        "edge_seed_pairs",
        "edge_seed_pair",
        "edge_seed_pair_original",
        "vertex_seed_triples",
        "vertex_seed_triples_original",
        "scipy_vertex_aug_seed_triples",
        "vertex_type",
        "node_type",
        "boundary_seed_pair",
        "boundary_source_type",
        "node_clip_source_vertices",
        "node_trim_curve_piece",
        "node_trim_curve_segment",
        "boundary_curve_offsets",
        "boundary_curve_loop_id",
    )
    OUTPUT_GRAD_KEYS = (
        "vertices_uv",
        "vertices_xyz",
        "edge_curves_uv",
        "edge_curves_xyz",
        "density",
        "fiber",
        "rho",
        "fiber3d",
    )

    def __init__(
        self,
        decoder: Any,
        cad_domain: Any,
        config: Any | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
        edge_types: Iterable[int] = (0,),
        finite_difference_step: float = 1e-6,
        relative_tolerance: float = 1e-3,
        absolute_tolerance: float = 1e-6,
        topology_change_tolerance: float | None = None,
        verbose: bool = True,
        topology_builder: Callable[..., Any] | None = None,
    ) -> None:
        self.decoder = decoder
        self.cad_domain = cad_domain
        self.config = config
        self.device = torch.device(device) if device is not None else self._infer_device(decoder)
        self.dtype = dtype
        self.edge_types = tuple(int(v) for v in edge_types)
        self.finite_difference_step = float(finite_difference_step)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.topology_change_tolerance = topology_change_tolerance
        self.verbose = bool(verbose)
        self.topology_builder = topology_builder
        self.discovered_methods = self._discover_methods()
        self.last_signature_fields: list[str] = []

    def test_autograd_connectivity(
        self,
        seeds_uv: torch.Tensor,
        decoder_kwargs: dict[str, Any] | None = None,
        loss_fn: LossFn | None = None,
    ) -> dict[str, Any]:
        """Check that decoder outputs and the selected loss connect to seeds."""
        decoder_kwargs = dict(decoder_kwargs or {})
        seeds = self._prepare_seeds(seeds_uv, requires_grad=True)
        self._clear_parameter_grads()
        with self._temporary_decoder_eval_and_dtype():
            decoder_out = self._call_decoder(seeds, decoder_kwargs)
            loss, metrics = self._compute_loss(decoder_out, loss_fn)
            result = {
                "loss": self._scalar(loss),
                "loss_requires_grad": bool(loss.requires_grad),
                "loss_metrics": self._json_safe(metrics),
                "output_grad_summary": self.output_grad_state_summary(decoder_out),
            }
            if not loss.requires_grad:
                result.update(
                    {
                        "seed_grad_is_none": True,
                        "all_finite": False,
                        "grad_norm": 0.0,
                        "grad_max_abs": 0.0,
                        "nonzero_component_count": 0,
                        "per_seed_grad_norm": [],
                        "overall_pass": False,
                        "failure": "Selected loss is detached from autograd.",
                    }
                )
                return result
            grad = torch.autograd.grad(loss, seeds, retain_graph=False, create_graph=False, allow_unused=True)[0]
        self._clear_parameter_grads()
        result.update(self._grad_stats(grad, seeds))
        if result["seed_grad_is_none"]:
            result["failure"] = "Seed gradient is None."
        elif not result["all_finite"]:
            result["failure"] = "Seed gradient contains NaN or Inf."
        result["overall_pass"] = bool(loss.requires_grad and not result["seed_grad_is_none"] and result["all_finite"])
        return result

    def build_topology(self, seeds_uv: torch.Tensor, decoder_kwargs: dict[str, Any] | None = None) -> Any:
        """Build detached discrete topology using the decoder API or supplied builder."""
        kwargs = dict(decoder_kwargs or {})
        seeds = self._prepare_seeds(seeds_uv, requires_grad=False)
        builder = self.topology_builder
        if builder is None:
            for name in self.TOPOLOGY_BUILDER_NAMES:
                candidate = getattr(self.decoder, name, None)
                if callable(candidate):
                    builder = candidate
                    break
        if builder is None:
            raise RuntimeError(
                "No topology builder found. Supply topology_builder or add one of: "
                + ", ".join(self.TOPOLOGY_BUILDER_NAMES)
            )
        call_kwargs = self._filter_kwargs(builder, kwargs)
        if "cad_domain" in inspect.signature(builder).parameters and "cad_domain" not in call_kwargs:
            call_kwargs["cad_domain"] = self.cad_domain
        if "u_periodic" in inspect.signature(builder).parameters and "u_periodic" not in call_kwargs:
            call_kwargs["u_periodic"] = self._decoder_bool("face_u_periodic", False)
        if "v_periodic" in inspect.signature(builder).parameters and "v_periodic" not in call_kwargs:
            call_kwargs["v_periodic"] = self._decoder_bool("face_v_periodic", False)
        with torch.no_grad():
            return builder(seeds, **call_kwargs)

    def topology_signature(self, topology: Any) -> str:
        """Hash discrete connectivity metadata only."""
        hasher = hashlib.sha256()
        used: list[str] = []

        def visit(obj: Any, prefix: str = "") -> None:
            for key in self.DISCRETE_SIGNATURE_KEYS:
                value = self._get_value(obj, key)
                if value is not None:
                    before = hasher.copy().digest()
                    self._hash_discrete_value(hasher, prefix + key, value)
                    after = hasher.copy().digest()
                    if before != after:
                        used.append(prefix + key)
            graph = self._get_value(obj, "graph")
            if graph is not None:
                visit(graph, prefix + "graph.")
            edges = self._get_value(obj, "edges")
            if isinstance(edges, dict):
                visit(edges, prefix + "edges.")

        visit(topology)
        if not used:
            raise RuntimeError(
                "Topology signature could not find discrete fields. Refusing to hash floating coordinates only."
            )
        self.last_signature_fields = sorted(set(used))
        return hasher.hexdigest()

    def seed_adjacency_signature(self, topology: Any) -> str:
        """
        Hash the canonical undirected seed-neighbor graph.

        This ignores Voronoi vertex numbering, edge ordering, edge direction,
        and boundary bookkeeping that do not represent a true adjacency change.
        """
        pairs = self._get_value(topology, "edge_seed_pairs")

        if pairs is None:
            pairs = self._get_value(topology, "edge_seed_pair")

        if pairs is None:
            edges = self._get_value(topology, "edges")

            if isinstance(edges, dict):
                pairs = self._first_present(
                    edges,
                    (
                        "edge_seed_pair_original",
                        "edge_seed_pair",
                        "edge_seed_pairs",
                    ),
                )

        if pairs is None:
            raise RuntimeError(
                "Cannot build seed-adjacency signature: "
                "no edge seed-pair metadata was found."
            )

        pairs = torch.as_tensor(
            pairs,
            dtype=torch.long,
        ).detach().cpu().numpy().reshape(-1, 2)

        # Exclude shell and non-seed edges.
        pairs = pairs[
            (pairs[:, 0] >= 0)
            & (pairs[:, 1] >= 0)
        ]

        # Canonical undirected pairs.
        pairs = np.sort(pairs, axis=1)

        # Remove duplicate adjacency entries and sort rows.
        pairs = np.unique(pairs, axis=0)
        pairs = np.ascontiguousarray(pairs)

        hasher = hashlib.sha256()
        hasher.update(str(pairs.shape).encode("utf-8"))
        hasher.update(pairs.tobytes())

        return hasher.hexdigest()

    def decode_with_frozen_topology(
        self,
        seeds_uv: torch.Tensor,
        topology: Any,
        decoder_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reconstruct differentiable geometry with fixed discrete topology."""
        kwargs = dict(decoder_kwargs or {})
        if isinstance(seeds_uv, torch.Tensor):
            seeds = seeds_uv.to(dtype=self.dtype, device=self.device)
        else:
            seeds = torch.as_tensor(seeds_uv, dtype=self.dtype, device=self.device)
        vertices_fn = getattr(self.decoder, "differentiable_vertices_from_topology", None)
        sample_graph_fn = getattr(self.decoder, "sample_graph_edge_curves_uv", None)
        lift_fn = getattr(self.decoder, "sample_smooth_edge_curves_xyz", None)
        loop_fn = getattr(self.decoder, "build_boundary_loop_edges", None)
        missing = [
            name
            for name, fn in (
                ("differentiable_vertices_from_topology", vertices_fn),
                ("sample_graph_edge_curves_uv", sample_graph_fn),
                ("sample_smooth_edge_curves_xyz", lift_fn),
            )
            if not callable(fn)
        ]
        if missing:
            raise RuntimeError(f"Frozen-topology adapter missing decoder methods: {missing}")

        required = ("vertex_type", "boundary_source_type", "vertices_uv", "node_clip_source_vertices", "scipy_vertex_aug_seed_triples", "guard_seeds_uv", "edges", "edge_seed_pairs", "edge_type")
        missing_fields = [key for key in required if self._get_value(topology, key) is None]
        if missing_fields:
            raise RuntimeError(f"Frozen-topology adapter missing topology fields: {missing_fields}")

        vertices_uv = vertices_fn(
            seeds_uv=seeds,
            vertex_type=self._tensor_like(self._get_value(topology, "vertex_type"), dtype=torch.long, device=seeds.device),
            boundary_source_type=self._tensor_like(self._get_value(topology, "boundary_source_type"), dtype=torch.long, device=seeds.device),
            topology_vertices_uv=self._tensor_like(self._get_value(topology, "vertices_uv"), dtype=seeds.dtype, device=seeds.device),
            node_clip_source_vertices=self._tensor_like(self._get_value(topology, "node_clip_source_vertices"), dtype=torch.long, device=seeds.device),
            scipy_vertex_aug_seed_triples=self._tensor_like(self._get_value(topology, "scipy_vertex_aug_seed_triples"), dtype=torch.long, device=seeds.device),
            guard_seeds_uv=self._tensor_like(self._get_value(topology, "guard_seeds_uv"), dtype=seeds.dtype, device=seeds.device),
            node_trim_segment_uv=self._tensor_like(self._get_value(topology, "node_trim_segment_uv"), dtype=seeds.dtype, device=seeds.device),
        )
        vertex_type = self._tensor_like(self._get_value(topology, "vertex_type"), dtype=torch.long, device=seeds.device)
        edges = self._tensor_like(self._get_value(topology, "edges"), dtype=torch.long, device=seeds.device).reshape(-1, 2)
        edge_type = self._tensor_like(self._get_value(topology, "edge_type"), dtype=torch.long, device=seeds.device).reshape(-1)
        edge_seed_pairs = self._tensor_like(self._get_value(topology, "edge_seed_pairs"), dtype=torch.long, device=seeds.device).reshape(-1, 2)
        graph: dict[str, Any] = {
            "nodes_uv": vertices_uv,
            "node_type": vertex_type,
            "edge_index": edges,
            "edge_seed_pair": edge_seed_pairs,
            "edge_type": edge_type,
            "boundary_source_type": self._tensor_like(self._get_value(topology, "boundary_source_type"), dtype=torch.long, device=seeds.device),
        }
        for key in ("node_trim_curve_piece", "node_trim_curve_segment", "node_trim_curve_fraction", "node_trim_segment_uv"):
            value = self._get_value(topology, key)
            if value is not None:
                graph[key] = self._tensor_like(value, dtype=seeds.dtype if key.endswith("uv") or key.endswith("fraction") else torch.long, device=seeds.device)

        if callable(loop_fn):
            loop_edges, loop_edge_type, boundary_data = loop_fn(vertices_uv, vertex_type, cad_domain=self.cad_domain)
            edges = torch.cat((edges, loop_edges.to(device=seeds.device)), dim=0)
            edge_type = torch.cat((edge_type, loop_edge_type.to(device=seeds.device)), dim=0)
            loop_seed_pairs = torch.full((loop_edges.shape[0], 2), -1, dtype=torch.long, device=seeds.device)
            edge_seed_pairs = torch.cat((edge_seed_pairs, loop_seed_pairs), dim=0)
            graph.update({"edge_index": edges, "edge_type": edge_type, "edge_seed_pair": edge_seed_pairs})
            if isinstance(boundary_data, dict):
                for key in ("boundary_curve_uv", "boundary_curve_offsets", "boundary_curve_loop_id"):
                    if key in boundary_data:
                        graph[key] = self._tensor_like(boundary_data[key], dtype=seeds.dtype if key == "boundary_curve_uv" else torch.long, device=seeds.device)

        n_samples = int(kwargs.get("n_samples", kwargs.get("edge_curve_samples", getattr(self.decoder, "tube_curve_samples", 64))))
        n_samples = max(n_samples, 2)
        curves_uv = sample_graph_fn(
            seeds_uv=seeds,
            graph=graph,
            n_samples=n_samples,
            u_periodic=bool(kwargs.get("u_periodic", self._decoder_bool("face_u_periodic", False))),
            v_periodic=bool(kwargs.get("v_periodic", self._decoder_bool("face_v_periodic", False))),
        )
        curves_xyz = lift_fn(self.cad_domain, curves_uv)
        return {
            "vertices_uv": vertices_uv,
            "edge_curves_uv": curves_uv,
            "edge_curves_xyz": curves_xyz,
            "edge_types": edge_type,
            "edge_type": edge_type,
            "edges": {"edge_index": edges, "edge_seed_pair": edge_seed_pairs, "edge_type": edge_type},
            "graph": graph,
            "mode": "frozen_topology",
        }

    def test_frozen_topology_gradient(
        self,
        seeds_uv: torch.Tensor,
        seed_indices: Iterable[int] | None = None,
        coordinate_indices: Iterable[int] | None = None,
        decoder_kwargs: dict[str, Any] | None = None,
        loss_fn: LossFn | None = None,
        step: float | None = None,
    ) -> dict[str, Any]:
        """Authoritative finite-difference test on one fixed topology branch."""
        decoder_kwargs = dict(decoder_kwargs or {})
        h = float(step or self.finite_difference_step)
        topology = self.build_topology(seeds_uv, decoder_kwargs)
        signature = self.seed_adjacency_signature(topology)
        seeds = self._prepare_seeds(seeds_uv, requires_grad=True)
        with self._temporary_decoder_eval_and_dtype():
            out = self.decode_with_frozen_topology(seeds, topology, decoder_kwargs)
            loss, _ = self._compute_loss(out, loss_fn)
            grad = torch.autograd.grad(loss, seeds, allow_unused=False)[0]
        components = self._component_indices(seeds, seed_indices, coordinate_indices)
        details = []
        for seed_i, coord_i in components:
            plus = seeds.detach().clone()
            minus = seeds.detach().clone()
            plus[seed_i, coord_i] += h
            minus[seed_i, coord_i] -= h
            f_plus = self._eval_frozen_loss(plus, topology, decoder_kwargs, loss_fn)
            f_minus = self._eval_frozen_loss(minus, topology, decoder_kwargs, loss_fn)
            fd = (f_plus - f_minus) / (2.0 * h)
            autograd_value = float(grad[seed_i, coord_i].detach().cpu().item())
            abs_error = abs(autograd_value - fd)
            rel_error = abs_error / (abs(autograd_value) + abs(fd) + 1e-30)
            passed = abs_error <= self.absolute_tolerance or rel_error <= self.relative_tolerance
            details.append(
                {
                    "seed_index": int(seed_i),
                    "coordinate_index": int(coord_i),
                    "autograd": autograd_value,
                    "finite_difference": fd,
                    "absolute_error": abs_error,
                    "relative_error": rel_error,
                    "pass": passed,
                }
            )
        return self._gradient_result_summary(signature, details)

    def test_full_decoder_local_gradient(
        self,
        seeds_uv: torch.Tensor,
        seed_indices: Iterable[int] | None = None,
        coordinate_indices: Iterable[int] | None = None,
        decoder_kwargs: dict[str, Any] | None = None,
        loss_fn: LossFn | None = None,
        step: float | None = None,
    ) -> dict[str, Any]:
        """Finite-difference check that rebuilds topology and classifies topology events."""
        decoder_kwargs = dict(decoder_kwargs or {})
        h = float(step or self.finite_difference_step)
        base_topology = self.build_topology(seeds_uv, decoder_kwargs)
        base_sig = self.seed_adjacency_signature(base_topology)
        seeds = self._prepare_seeds(seeds_uv, requires_grad=True)
        with self._temporary_decoder_eval_and_dtype():
            base_out = self._call_decoder(seeds, decoder_kwargs)
            base_loss, _ = self._compute_loss(base_out, loss_fn)
            grad = torch.autograd.grad(base_loss, seeds, allow_unused=False)[0]
        details = []
        for seed_i, coord_i in self._component_indices(seeds, seed_indices, coordinate_indices):
            plus = seeds.detach().clone()
            minus = seeds.detach().clone()
            plus[seed_i, coord_i] += h
            minus[seed_i, coord_i] -= h
            try:
                plus_sig = self.seed_adjacency_signature(self.build_topology(plus, decoder_kwargs))
                minus_sig = self.seed_adjacency_signature(self.build_topology(minus, decoder_kwargs))
                if plus_sig != base_sig or minus_sig != base_sig:
                    if self.verbose:
                        print("Topology change detected; finite difference is not expected to match the derivative of the frozen local branch.")
                    details.append({"seed_index": int(seed_i), "coordinate_index": int(coord_i), "classification": "topology_change", "base_signature": base_sig, "plus_signature": plus_sig, "minus_signature": minus_sig})
                    continue
                f_plus = self._eval_full_loss(plus, decoder_kwargs, loss_fn)
                f_minus = self._eval_full_loss(minus, decoder_kwargs, loss_fn)
            except (QhullError, RuntimeError, ValueError) as exc:
                details.append({"seed_index": int(seed_i), "coordinate_index": int(coord_i), "classification": "evaluation_failure", "error": repr(exc), "base_signature": base_sig})
                continue
            fd = (f_plus - f_minus) / (2.0 * h)
            autograd_value = float(grad[seed_i, coord_i].detach().cpu().item())
            abs_error = abs(autograd_value - fd)
            rel_error = abs_error / (abs(autograd_value) + abs(fd) + 1e-30)
            passed = abs_error <= self.absolute_tolerance or rel_error <= self.relative_tolerance
            details.append({"seed_index": int(seed_i), "coordinate_index": int(coord_i), "classification": "stable_topology", "autograd": autograd_value, "finite_difference": fd, "absolute_error": abs_error, "relative_error": rel_error, "pass": passed, "base_signature": base_sig})
        stable = [d for d in details if d["classification"] == "stable_topology"]
        changes = [d for d in details if d["classification"] == "topology_change"]
        failures = [d for d in details if d["classification"] == "evaluation_failure"]
        return {
            "base_topology_signature": base_sig,
            "stable_comparisons": len(stable),
            "topology_change_events": len(changes),
            "decoder_failures": len(failures),
            "pass_count": sum(1 for d in stable if d.get("pass")),
            "fail_count": sum(1 for d in stable if not d.get("pass")),
            "details": details,
            "overall_pass": all(d.get("pass", True) for d in stable) and not failures,
        }

    def test_step_size_sweep(
        self,
        seeds_uv: torch.Tensor,
        seed_index: int,
        coordinate_index: int,
        steps: Iterable[float] = (1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7),
        decoder_kwargs: dict[str, Any] | None = None,
        loss_fn: LossFn | None = None,
    ) -> dict[str, Any]:
        """Run frozen-topology finite differences for several step sizes."""
        rows = []
        for h in steps:
            result = self.test_frozen_topology_gradient(
                seeds_uv,
                seed_indices=[seed_index],
                coordinate_indices=[coordinate_index],
                decoder_kwargs=decoder_kwargs,
                loss_fn=loss_fn,
                step=float(h),
            )
            detail = result["details"][0]
            rows.append({"step": float(h), **{k: detail[k] for k in ("finite_difference", "autograd", "absolute_error", "relative_error", "pass")}})
        return {"seed_index": int(seed_index), "coordinate_index": int(coordinate_index), "results": rows}

    def test_directional_derivative(
        self,
        seeds_uv: torch.Tensor,
        num_directions: int = 5,
        step: float = 1e-6,
        decoder_kwargs: dict[str, Any] | None = None,
        loss_fn: LossFn | None = None,
        frozen_topology: bool = True,
        random_seed: int = 0,
    ) -> dict[str, Any]:
        """Compare autograd and central finite differences along random directions."""
        decoder_kwargs = dict(decoder_kwargs or {})
        torch_gen = torch.Generator(device="cpu").manual_seed(int(random_seed))
        base = self._prepare_seeds(seeds_uv, requires_grad=True)
        topology = self.build_topology(base, decoder_kwargs) if frozen_topology else None
        with self._temporary_decoder_eval_and_dtype():
            out = self.decode_with_frozen_topology(base, topology, decoder_kwargs) if frozen_topology else self._call_decoder(base, decoder_kwargs)
            loss, _ = self._compute_loss(out, loss_fn)
            grad = torch.autograd.grad(loss, base, allow_unused=False)[0]
        rows = []
        for direction_id in range(int(num_directions)):
            direction = torch.randn(base.shape, generator=torch_gen, dtype=base.dtype).to(base.device)
            direction = direction / direction.norm().clamp_min(1e-30)
            autograd_dir = float((grad * direction).sum().detach().cpu().item())
            plus = base.detach() + float(step) * direction
            minus = base.detach() - float(step) * direction
            if frozen_topology:
                f_plus = self._eval_frozen_loss(plus, topology, decoder_kwargs, loss_fn)
                f_minus = self._eval_frozen_loss(minus, topology, decoder_kwargs, loss_fn)
            else:
                f_plus = self._eval_full_loss(plus, decoder_kwargs, loss_fn)
                f_minus = self._eval_full_loss(minus, decoder_kwargs, loss_fn)
            fd = (f_plus - f_minus) / (2.0 * float(step))
            abs_error = abs(autograd_dir - fd)
            rel_error = abs_error / (abs(autograd_dir) + abs(fd) + 1e-30)
            rows.append({"direction": direction_id, "autograd": autograd_dir, "finite_difference": fd, "absolute_error": abs_error, "relative_error": rel_error, "pass": abs_error <= self.absolute_tolerance or rel_error <= self.relative_tolerance})
        return {"frozen_topology": bool(frozen_topology), "directions": rows, "overall_pass": all(r["pass"] for r in rows)}

    def analyze_seed_gradient_coverage(
        self,
        seeds_uv: torch.Tensor,
        decoder_kwargs: dict[str, Any] | None = None,
        loss_fn: LossFn | None = None,
        zero_threshold: float = 1e-12,
    ) -> dict[str, Any]:
        """Report which seeds receive gradients and whether they participate in selected edges."""
        result = self.test_autograd_connectivity(seeds_uv, decoder_kwargs, loss_fn)
        seeds = self._prepare_seeds(seeds_uv, requires_grad=True)
        out = self._call_decoder(seeds, dict(decoder_kwargs or {}))
        loss, metrics = self._compute_loss(out, loss_fn)
        grad = torch.autograd.grad(loss, seeds, allow_unused=True)[0]
        norms = torch.zeros((seeds.shape[0],), dtype=seeds.dtype) if grad is None else torch.linalg.vector_norm(grad.detach(), dim=1).cpu()
        near_zero = [int(i) for i, v in enumerate(norms.tolist()) if float(v) <= zero_threshold]
        participating = self._participating_selected_seeds(out)
        non_participating = sorted(set(range(seeds.shape[0])) - participating) if participating is not None else None
        selected_counts = self._selected_edge_type_counts(out)
        return {
            "per_seed_grad_norm": [float(v) for v in norms.tolist()],
            "zero_or_near_zero_seed_indices": near_zero,
            "fraction_receiving_gradient": float(sum(v > zero_threshold for v in norms.tolist()) / max(len(norms), 1)),
            "selected_edge_count": int(metrics.get("count", 0)) if isinstance(metrics, dict) else None,
            "selected_edge_type_counts": selected_counts,
            "participating_seed_indices": sorted(participating) if participating is not None else None,
            "nonparticipating_seed_indices": non_participating,
            "zero_because_nonparticipating": sorted(set(near_zero).intersection(non_participating or [])),
            "zero_despite_participation": sorted(set(near_zero).intersection(participating or set())),
            "connectivity": result,
        }

    def test_topology_event_behavior(
        self,
        seeds_uv: torch.Tensor,
        perturbation_scales: Iterable[float] = (1e-6, 1e-5, 1e-4, 1e-3),
        trials_per_scale: int = 20,
        decoder_kwargs: dict[str, Any] | None = None,
        random_seed: int = 0,
    ) -> dict[str, Any]:
        """Characterize local topology-change frequency under random perturbations."""
        decoder_kwargs = dict(decoder_kwargs or {})
        base = self._prepare_seeds(seeds_uv, requires_grad=False)
        base_sig = self.seed_adjacency_signature(self.build_topology(base, decoder_kwargs))
        gen = torch.Generator(device="cpu").manual_seed(int(random_seed))
        rows = []
        for scale in perturbation_scales:
            counts = Counter()
            for _ in range(int(trials_per_scale)):
                direction = torch.randn(base.shape, generator=gen, dtype=base.dtype).to(base.device)
                direction = direction / direction.norm().clamp_min(1e-30)
                try:
                    sig = self.seed_adjacency_signature(self.build_topology(base + float(scale) * direction, decoder_kwargs))
                    counts["unchanged" if sig == base_sig else "changed"] += 1
                except (QhullError, RuntimeError, ValueError):
                    counts["failed"] += 1
            total = max(int(trials_per_scale), 1)
            rows.append({"scale": float(scale), "unchanged": counts["unchanged"], "changed": counts["changed"], "failed": counts["failed"], "change_frequency": counts["changed"] / total})
        return {"base_topology_signature": base_sig, "scales": rows}

    def run_all(
        self,
        seeds_uv: torch.Tensor,
        decoder_kwargs: dict[str, Any] | None = None,
        loss_fn: LossFn | None = None,
        run_full_decoder_test: bool = True,
        run_step_sweep: bool = True,
        run_directional_test: bool = True,
        run_topology_stress: bool = True,
    ) -> dict[str, Any]:
        """Run the common diagnostic suite and return one nested dictionary."""
        decoder_kwargs = dict(decoder_kwargs or {})
        results: dict[str, Any] = {}
        results["connectivity"] = self.test_autograd_connectivity(seeds_uv, decoder_kwargs, loss_fn)
        results["frozen_topology"] = self.test_frozen_topology_gradient(seeds_uv, decoder_kwargs=decoder_kwargs, loss_fn=loss_fn)
        if run_full_decoder_test:
            results["full_decoder"] = self.test_full_decoder_local_gradient(seeds_uv, decoder_kwargs=decoder_kwargs, loss_fn=loss_fn)
        if run_directional_test:
            results["directional"] = self.test_directional_derivative(seeds_uv, decoder_kwargs=decoder_kwargs, loss_fn=loss_fn)
        results["coverage"] = self.analyze_seed_gradient_coverage(seeds_uv, decoder_kwargs, loss_fn)
        if run_step_sweep:
            seed_i = 0
            coord_i = 0
            results["step_sweep"] = self.test_step_size_sweep(
                seeds_uv,
                seed_index=seed_i,
                coordinate_index=coord_i,
                decoder_kwargs=decoder_kwargs,
                loss_fn=loss_fn,
            )
        if run_topology_stress:
            results["topology_stress"] = self.test_topology_event_behavior(seeds_uv, decoder_kwargs=decoder_kwargs)
        if self.verbose:
            self._print_summary(results)
        return results

    def default_curve_length_loss(
        self,
        decoder_out: dict[str, Any],
        edge_types: Iterable[int] | None = None,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Mean squared log-relative polyline length loss for selected edge types."""
        curves = self._first_present(decoder_out, ("edge_curves_xyz", "curves_xyz"))
        if curves is None:
            raise KeyError("Decoder output must contain edge_curves_xyz or curves_xyz.")
        if not isinstance(curves, torch.Tensor) or curves.ndim != 3:
            raise ValueError("Edge curves must be a tensor with shape [E, K, D].")
        types = self._edge_types_from_out(decoder_out)
        selected_types = tuple(self.edge_types if edge_types is None else edge_types)
        if types is not None:
            type_t = torch.as_tensor(types, dtype=torch.long, device=curves.device).reshape(-1)
            mask = torch.zeros((curves.shape[0],), dtype=torch.bool, device=curves.device)
            for edge_type in selected_types:
                mask |= type_t == int(edge_type)
            curves = curves[mask]
        else:
            mask = torch.ones((curves.shape[0],), dtype=torch.bool, device=curves.device)
        if curves.shape[0] < 2:
            raise ValueError(f"Default loss needs at least two selected edges; got {curves.shape[0]}.")
        lengths = torch.linalg.vector_norm(curves[:, 1:] - curves[:, :-1], dim=-1).sum(dim=1).clamp_min(float(eps))
        mean_length = lengths.mean().clamp_min(float(eps))
        loss = torch.square(torch.log(lengths / mean_length)).mean()
        metrics = {
            "count": int(lengths.shape[0]),
            "min": self._scalar(lengths.min()),
            "max": self._scalar(lengths.max()),
            "mean": self._scalar(mean_length),
            "std": self._scalar(lengths.std(unbiased=False)),
            "cv": self._scalar(lengths.std(unbiased=False) / mean_length),
            "max_min_ratio": self._scalar(lengths.max() / lengths.min().clamp_min(float(eps))),
            "edge_types": list(selected_types),
        }
        return loss, metrics

    def output_grad_state_summary(self, decoder_out: Any) -> dict[str, Any]:
        """Summarize grad state for likely differentiable decoder outputs."""
        summary = {}
        for key in self.OUTPUT_GRAD_KEYS:
            value = self._get_value(decoder_out, key)
            if isinstance(value, torch.Tensor):
                summary[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "requires_grad": bool(value.requires_grad),
                    "grad_fn": type(value.grad_fn).__name__ if value.grad_fn is not None else None,
                }
        return summary

    def save_json(self, path: str, results: Any) -> None:
        """Save JSON-safe summaries and finite-difference details."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._json_safe(results), handle, indent=2, sort_keys=True)

    def _call_decoder(self, seeds: torch.Tensor, decoder_kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(decoder_kwargs or {})
        try:
            out = self.decoder(seeds, **kwargs)
        except (TypeError, ValueError) as exc:
            topo_forward = getattr(self.decoder, "forward_scipy_topology", None)
            if not callable(topo_forward):
                raise
            if "w_raw" not in str(exc) and "must be provided" not in str(exc) and "points_3d" not in str(exc):
                raise
            call_kwargs = self._filter_kwargs(topo_forward, kwargs)
            call_kwargs.setdefault("cad_domain", self.cad_domain)
            call_kwargs.setdefault("u_periodic", self._decoder_bool("face_u_periodic", False))
            call_kwargs.setdefault("v_periodic", self._decoder_bool("face_v_periodic", False))
            out = topo_forward(seeds, **call_kwargs)
        if not isinstance(out, dict):
            raise TypeError("Decoder must return a dictionary-like output for gradient testing.")
        return out

    def _compute_loss(self, decoder_out: dict[str, Any], loss_fn: LossFn | None) -> tuple[torch.Tensor, dict[str, Any]]:
        if loss_fn is None:
            return self.default_curve_length_loss(decoder_out)
        value = loss_fn(decoder_out)
        if isinstance(value, tuple):
            loss, metrics = value
        else:
            loss, metrics = value, {}
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError("loss_fn must return a scalar torch.Tensor or (scalar_tensor, metrics).")
        return loss, dict(metrics or {})

    def _eval_frozen_loss(self, seeds: torch.Tensor, topology: Any, decoder_kwargs: dict[str, Any], loss_fn: LossFn | None) -> float:
        with self._temporary_decoder_eval_and_dtype():
            out = self.decode_with_frozen_topology(seeds.detach(), topology, decoder_kwargs)
            loss, _ = self._compute_loss(out, loss_fn)
            return self._scalar(loss)

    def _eval_full_loss(self, seeds: torch.Tensor, decoder_kwargs: dict[str, Any], loss_fn: LossFn | None) -> float:
        with self._temporary_decoder_eval_and_dtype():
            out = self._call_decoder(seeds.detach(), decoder_kwargs)
            loss, _ = self._compute_loss(out, loss_fn)
            return self._scalar(loss)

    def _component_indices(self, seeds: torch.Tensor, seed_indices: Iterable[int] | None, coordinate_indices: Iterable[int] | None) -> list[tuple[int, int]]:
        n = int(seeds.shape[0])
        if seed_indices is None:
            if n <= 20:
                seed_ids = list(range(n))
            else:
                seed_ids = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1]))
        else:
            seed_ids = [int(i) for i in seed_indices]
        coord_ids = [0, 1] if coordinate_indices is None else [int(i) for i in coordinate_indices]
        return [(i, j) for i in seed_ids for j in coord_ids]

    def _gradient_result_summary(self, signature: str, details: list[dict[str, Any]]) -> dict[str, Any]:
        pass_count = sum(1 for d in details if d["pass"])
        fail_count = len(details) - pass_count
        rels = [float(d["relative_error"]) for d in details]
        abss = [float(d["absolute_error"]) for d in details]
        return {
            "topology_signature": signature,
            "tested_component_count": len(details),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "max_absolute_error": max(abss) if abss else 0.0,
            "max_relative_error": max(rels) if rels else 0.0,
            "mean_relative_error": float(np.mean(rels)) if rels else 0.0,
            "details": details,
            "overall_pass": fail_count == 0,
        }

    def _prepare_seeds(self, seeds_uv: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        seeds = torch.as_tensor(seeds_uv, dtype=self.dtype, device=self.device).detach().clone()
        seeds.requires_grad_(requires_grad)
        return seeds

    @contextlib.contextmanager
    def _temporary_decoder_eval_and_dtype(self) -> Any:
        was_training = bool(getattr(self.decoder, "training", False))
        old_param_dtypes = {p: p.dtype for p in self._decoder_parameters()}
        try:
            if hasattr(self.decoder, "eval"):
                self.decoder.eval()
            if hasattr(self.decoder, "to"):
                self.decoder.to(dtype=self.dtype, device=self.device)
            yield
        finally:
            for dtype in set(old_param_dtypes.values()):
                if hasattr(self.decoder, "to"):
                    self.decoder.to(dtype=dtype)
                    break
            if was_training and hasattr(self.decoder, "train"):
                self.decoder.train(True)

    def _decoder_parameters(self) -> list[torch.nn.Parameter]:
        parameters = getattr(self.decoder, "parameters", None)
        if callable(parameters):
            return list(parameters())
        return []

    def _clear_parameter_grads(self) -> None:
        zero_grad = getattr(self.decoder, "zero_grad", None)
        if callable(zero_grad):
            zero_grad(set_to_none=True)

    def _grad_stats(self, grad: torch.Tensor | None, seeds: torch.Tensor) -> dict[str, Any]:
        if grad is None:
            return {"seed_grad_is_none": True, "all_finite": False, "grad_norm": 0.0, "grad_max_abs": 0.0, "nonzero_component_count": 0, "per_seed_grad_norm": [0.0] * int(seeds.shape[0])}
        finite = torch.isfinite(grad)
        per_seed = torch.linalg.vector_norm(grad.detach(), dim=1)
        return {
            "seed_grad_is_none": False,
            "all_finite": bool(finite.all().item()),
            "grad_norm": self._scalar(torch.linalg.vector_norm(grad.detach())),
            "grad_max_abs": self._scalar(grad.detach().abs().max()),
            "nonzero_component_count": int((grad.detach().abs() > 0).sum().item()),
            "per_seed_grad_norm": [float(v) for v in per_seed.cpu().tolist()],
        }

    def _participating_selected_seeds(self, out: dict[str, Any]) -> set[int] | None:
        pairs = self._first_present(out, ("edge_seed_pair_original", "edge_seed_pairs", "edge_seed_pair"))
        if pairs is None:
            edges_obj = self._get_value(out, "edges")
            if isinstance(edges_obj, dict):
                pairs = self._first_present(edges_obj, ("edge_seed_pair_original", "edge_seed_pair", "edge_seed_pairs"))
        types = self._edge_types_from_out(out)
        if pairs is None or types is None:
            return None
        pairs_t = torch.as_tensor(pairs).detach().cpu().reshape(-1, 2)
        types_t = torch.as_tensor(types).detach().cpu().reshape(-1)
        selected = torch.zeros((types_t.shape[0],), dtype=torch.bool)
        for edge_type in self.edge_types:
            selected |= types_t == int(edge_type)
        seeds = set()
        for pair in pairs_t[selected].tolist():
            for value in pair:
                if int(value) >= 0:
                    seeds.add(int(value))
        return seeds

    def _selected_edge_type_counts(self, out: dict[str, Any]) -> dict[int, int]:
        types = self._edge_types_from_out(out)
        if types is None:
            return {}
        types_t = torch.as_tensor(types).detach().cpu().reshape(-1).tolist()
        return {int(k): int(v) for k, v in Counter(int(v) for v in types_t if int(v) in self.edge_types).items()}

    def _edge_types_from_out(self, out: dict[str, Any]) -> Any | None:
        value = self._first_present(out, ("edge_types", "edge_type"))
        if value is not None:
            return value
        graph = self._get_value(out, "graph")
        if graph is not None:
            value = self._first_present(graph, ("edge_types", "edge_type"))
            if value is not None:
                return value
        edges = self._get_value(out, "edges")
        if isinstance(edges, dict):
            return self._first_present(edges, ("edge_types", "edge_type"))
        return None

    def _first_present(self, obj: Any, keys: Iterable[str]) -> Any | None:
        for key in keys:
            value = self._get_value(obj, key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _get_value(obj: Any, key: str) -> Any | None:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _tensor_like(self, value: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.empty((0,), dtype=dtype, device=device)
        if isinstance(value, torch.Tensor):
            return value.detach().to(dtype=dtype, device=device) if not value.is_floating_point() else value.to(dtype=dtype, device=device)
        return torch.as_tensor(value, dtype=dtype, device=device)

    @staticmethod
    def _hash_discrete_value(hasher: Any, name: str, value: Any) -> None:
        if isinstance(value, dict):
            return
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
        else:
            array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.kind not in ("i", "u", "b"):
            return
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.tobytes())

    @staticmethod
    def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in signature.parameters}

    def _decoder_bool(self, name: str, default: bool) -> bool:
        value = getattr(self.decoder, name, default)
        if isinstance(value, torch.Tensor):
            return bool(value.detach().reshape(-1)[0].item()) if value.numel() else default
        if isinstance(value, (list, tuple)):
            return bool(value[0]) if value else default
        return bool(value)

    @staticmethod
    def _infer_device(decoder: Any) -> torch.device:
        parameters = getattr(decoder, "parameters", None)
        if callable(parameters):
            for param in parameters():
                return param.device
        return torch.device("cpu")

    def _discover_methods(self) -> dict[str, bool]:
        names = set(self.TOPOLOGY_BUILDER_NAMES) | {
            "forward_scipy_topology",
            "differentiable_vertices_from_topology",
            "sample_graph_edge_curves_uv",
            "sample_smooth_edge_curves_xyz",
            "build_boundary_loop_edges",
        }
        return {name: callable(getattr(self.decoder, name, None)) for name in sorted(names)}

    @staticmethod
    def _scalar(value: torch.Tensor | np.ndarray | float | int) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        return float(value)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() > 64:
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            if value.size > 64:
                return {"shape": list(value.shape), "dtype": str(value.dtype)}
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _print_summary(self, results: dict[str, Any]) -> None:
        print("\nDECODER AUTOGRAD CONNECTIVITY")
        print(f"  pass={results['connectivity'].get('overall_pass')} grad_norm={results['connectivity'].get('grad_norm')}")
        print("FROZEN-TOPOLOGY GRADIENT CHECK")
        frozen = results["frozen_topology"]
        print(f"  pass={frozen['overall_pass']} tested={frozen['tested_component_count']} fails={frozen['fail_count']} max_rel={frozen['max_relative_error']:.3e}")
        if "full_decoder" in results:
            print("FULL-DECODER LOCAL CHECK")
            full = results["full_decoder"]
            print(f"  stable={full['stable_comparisons']} topology_events={full['topology_change_events']} failures={full['decoder_failures']} stable_fails={full['fail_count']}")
        print("TOPOLOGY CHANGE EVENTS")
        if "topology_stress" in results:
            for row in results["topology_stress"]["scales"]:
                print(f"  scale={row['scale']:.1e} changed={row['changed']} unchanged={row['unchanged']} failed={row['failed']}")
        print("SEED GRADIENT COVERAGE")
        coverage = results["coverage"]
        print(f"  receiving_fraction={coverage['fraction_receiving_gradient']:.3f} near_zero={coverage['zero_or_near_zero_seed_indices']}")


class TopologyChangeMonitor:
    """Lightweight topology-signature monitor for training loops."""

    def __init__(self, tester: DecoderGradientTester) -> None:
        self.tester = tester
        self.reset()

    def update(self, step: int, seeds_uv: torch.Tensor, decoder_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build current topology under no_grad, compare to previous signature, and record an event."""
        event: dict[str, Any] = {"step": int(step), "failed": False}
        try:
            with torch.no_grad():
                topology = self.tester.build_topology(seeds_uv.detach(), decoder_kwargs)
                signature = self.tester.seed_adjacency_signature(topology)
            previous = self.previous_signature
            changed = previous is not None and previous != signature
            event.update(
                {
                    "changed": bool(changed),
                    "previous_signature": previous,
                    "current_signature": signature,
                    "edge_count": self._edge_count(topology),
                    "edge_type_counts": self._edge_type_counts(topology),
                    "minimum_seed_distance": self._minimum_seed_distance(seeds_uv),
                }
            )
            self.previous_signature = signature
            self.total_updates += 1
            if changed:
                self.topology_changes += 1
                self.change_steps.append(int(step))
                self.consecutive_stable_steps = 0
            else:
                self.consecutive_stable_steps += 1
        except (QhullError, RuntimeError, ValueError) as exc:
            event.update({"failed": True, "changed": False, "error": repr(exc), "previous_signature": self.previous_signature, "current_signature": None})
            self.failures.append(event)
            self.total_updates += 1
        self.events.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        """Return aggregate monitor statistics."""
        return {
            "total_updates": self.total_updates,
            "topology_changes": self.topology_changes,
            "change_rate": self.topology_changes / max(self.total_updates, 1),
            "change_steps": list(self.change_steps),
            "consecutive_stable_steps": self.consecutive_stable_steps,
            "failure_count": len(self.failures),
            "last_signature": self.previous_signature,
        }

    def reset(self) -> None:
        self.previous_signature: str | None = None
        self.total_updates = 0
        self.topology_changes = 0
        self.change_steps: list[int] = []
        self.consecutive_stable_steps = 0
        self.events: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []

    @staticmethod
    def _edge_count(topology: Any) -> int:
        edges = DecoderGradientTester._get_value(topology, "edges")
        if isinstance(edges, dict):
            edges = edges.get("edge_index")
        if edges is None:
            edges = DecoderGradientTester._get_value(topology, "edge_index")
        return int(torch.as_tensor(edges).reshape(-1, 2).shape[0]) if edges is not None else 0

    @staticmethod
    def _edge_type_counts(topology: Any) -> dict[int, int]:
        edge_type = DecoderGradientTester._get_value(topology, "edge_type")
        if edge_type is None:
            edge_type = DecoderGradientTester._get_value(topology, "edge_types")
        if edge_type is None:
            return {}
        values = torch.as_tensor(edge_type).detach().cpu().reshape(-1).tolist()
        return {int(k): int(v) for k, v in Counter(int(x) for x in values).items()}

    @staticmethod
    def _minimum_seed_distance(seeds_uv: torch.Tensor) -> float:
        seeds = seeds_uv.detach()
        if seeds.shape[0] < 2:
            return float("inf")
        distances = torch.cdist(seeds.to(dtype=torch.float64), seeds.to(dtype=torch.float64))
        distances = distances.clone()
        distances.fill_diagonal_(float("inf"))
        value = distances.min()
        return float(value.cpu().item()) if torch.isfinite(value) else float("inf")


def example_usage(decoder: Any, cad_domain: Any, seeds_uv: torch.Tensor, config: Any | None = None) -> dict[str, Any]:
    """Run the standard gradient diagnostics for an existing decoder and seeds."""
    tester = DecoderGradientTester(decoder=decoder, cad_domain=cad_domain, config=config)
    return tester.run_all(seeds_uv)


if __name__ == "__main__":
    print(
        "GradTestClass.py provides DecoderGradientTester and TopologyChangeMonitor.\n"
        "Example:\n"
        "    from GradTestClass import DecoderGradientTester\n"
        "    tester = DecoderGradientTester(decoder, cad_domain)\n"
        "    results = tester.run_all(seeds_uv, decoder_kwargs={'w_raw': w_raw})\n"
        "\n"
        "If your decoder.forward requires width inputs, pass them through decoder_kwargs.\n"
        "For curve-only checks, this tester can fall back to forward_scipy_topology when available."
    )
