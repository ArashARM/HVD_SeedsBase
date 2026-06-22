from __future__ import annotations

import torch
import torch.nn as nn

from Decoder_CLasses.ContinuousVoronoiDecoder import ContinuousVoronoiDecoder


class DummyVoronoiSeedTrainer(nn.Module):
    """Optimize seed positions while rebuilding hard SciPy topology each step."""

    def __init__(
        self,
        initial_seeds: torch.Tensor,
        learning_rate: float = 1e-2,
        edge_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if initial_seeds.ndim != 2 or initial_seeds.shape[1] != 2:
            raise ValueError("initial_seeds must have shape [S, 2].")
        if initial_seeds.shape[0] < 3:
            raise ValueError("At least three seeds are required for Delaunay triangles.")

        self.seeds_uv = nn.Parameter(initial_seeds.detach().clone())
        self.decoder = ContinuousVoronoiDecoder(return_xyz=False)
        self.optimizer = torch.optim.Adam([self.seeds_uv], lr=learning_rate)
        self.edge_loss_weight = float(edge_loss_weight)

    def delaunay_equal_area_loss(self, out: dict) -> torch.Tensor:
        triangles_np = out["delaunay_triples_np"]
        if triangles_np.shape[0] == 0:
            raise RuntimeError("SciPy returned no Delaunay triangles.")

        triangles = torch.as_tensor(
            triangles_np,
            dtype=torch.long,
            device=self.seeds_uv.device,
        )
        points = self.seeds_uv[triangles]
        edge_01 = points[:, 1] - points[:, 0]
        edge_02 = points[:, 2] - points[:, 0]
        cross = edge_01[:, 0] * edge_02[:, 1] - edge_01[:, 1] * edge_02[:, 0]
        areas = 0.5 * cross.abs()
        return ((areas - areas.mean()) ** 2).mean()

    @staticmethod
    def voronoi_equal_edge_length_loss(out: dict) -> torch.Tensor:
        graph = out["graph"]
        nodes = graph["nodes_uv"]
        edges = graph["edge_index"]
        if edges.shape[0] == 0:
            raise RuntimeError("Generated Voronoi graph has no edges.")

        edge_vectors = nodes[edges[:, 0]] - nodes[edges[:, 1]]
        edge_lengths = torch.linalg.vector_norm(edge_vectors, dim=1)
        return ((edge_lengths - edge_lengths.mean()) ** 2).mean()

    def training_step(self, step: int) -> dict[str, float | int]:
        self.optimizer.zero_grad(set_to_none=True)

        # SciPy topology is deliberately rebuilt from the current seeds here.
        out = self.decoder(
            self.seeds_uv,
            topology_mode="scipy",
            return_xyz=False,
        )
        area_loss = self.delaunay_equal_area_loss(out)
        loss = area_loss
        if self.edge_loss_weight != 0.0:
            edge_loss = self.voronoi_equal_edge_length_loss(out)
            loss = loss + self.edge_loss_weight * edge_loss

        loss.backward()
        if self.seeds_uv.grad is None:
            raise AssertionError("seeds_uv.grad is None; gradient flow was broken.")
        grad_norm = float(torch.linalg.vector_norm(self.seeds_uv.grad).detach().cpu())
        if not torch.isfinite(self.seeds_uv.grad).all():
            raise AssertionError("seeds_uv.grad contains non-finite values.")
        if grad_norm <= 0.0:
            raise AssertionError("seeds_uv gradient norm must be greater than zero.")

        graph = out["graph"]
        diagnostics = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "num_nodes": int(graph["nodes_uv"].shape[0]),
            "num_edges": int(graph["edge_index"].shape[0]),
            "num_delaunay_triangles": int(out["delaunay_triples_np"].shape[0]),
        }
        print(
            f"step={diagnostics['step']:02d} "
            f"loss={diagnostics['loss']:.8e} "
            f"grad_norm={diagnostics['grad_norm']:.8e} "
            f"nodes={diagnostics['num_nodes']} "
            f"edges={diagnostics['num_edges']} "
            f"delaunay_triangles={diagnostics['num_delaunay_triangles']}"
        )

        self.optimizer.step()
        with torch.no_grad():
            self.seeds_uv.clamp_(0.0, 1.0)
        return diagnostics

    def _decode_current_structure(self) -> dict:
        return self.decoder(
            self.seeds_uv,
            topology_mode="scipy",
            return_xyz=False,
        )

    def _print_and_plot_structure(
        self,
        label: str,
        print_node_table: bool = False,
    ) -> dict:
        out = self._decode_current_structure()
        graph = out["graph"]
        print(f"\n{label} seeds_uv:")
        print(self.seeds_uv.detach().cpu())
        print(
            f"{label} structure: nodes={graph['nodes_uv'].shape[0]}, "
            f"edges={graph['edge_index'].shape[0]}, "
            f"delaunay_triangles={out['delaunay_triples_np'].shape[0]}"
        )
        self.decoder.plot_scipy_vs_generated_graph(
            self.seeds_uv.detach(),
            out=out,
            show_node_ids=True,
            print_node_table=print_node_table,
        )
        return out

    def fit(
        self,
        steps: int = 5,
        plot_first_last: bool = False,
        print_node_table: bool = False,
    ) -> list[dict[str, float | int]]:
        if plot_first_last:
            self._print_and_plot_structure(
                "Initial",
                print_node_table=print_node_table,
            )

        diagnostics = [self.training_step(step) for step in range(int(steps))]

        if plot_first_last:
            self._print_and_plot_structure(
                "Final",
                print_node_table=print_node_table,
            )
        return diagnostics


def irregular_test_seeds(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [0.12, 0.16],
            [0.42, 0.10],
            [0.81, 0.18],
            [0.20, 0.57],
            [0.55, 0.43],
            [0.87, 0.66],
            [0.36, 0.88],
            [0.72, 0.91],
        ],
        dtype=dtype,
    )


def test_scipy_topology_seed_gradients_with_area_loss() -> None:
    trainer = DummyVoronoiSeedTrainer(
        irregular_test_seeds(),
        learning_rate=5e-3,
    )
    diagnostics = trainer.fit(steps=3)

    assert all(item["grad_norm"] > 0.0 for item in diagnostics)
    assert all(item["num_nodes"] > 0 for item in diagnostics)
    assert all(item["num_edges"] > 0 for item in diagnostics)
    assert all(item["num_delaunay_triangles"] > 0 for item in diagnostics)
    assert torch.all((trainer.seeds_uv >= 0.0) & (trainer.seeds_uv <= 1.0))


def test_scipy_reconstructed_graph_geometry_has_seed_gradients() -> None:
    trainer = DummyVoronoiSeedTrainer(
        irregular_test_seeds(),
        learning_rate=2e-3,
        edge_loss_weight=0.25,
    )
    diagnostics = trainer.fit(steps=2)

    assert all(item["grad_norm"] > 0.0 for item in diagnostics)
    assert torch.all((trainer.seeds_uv >= 0.0) & (trainer.seeds_uv <= 1.0))


def test_smooth_edge_curves_are_differentiable_and_handle_invalid_pairs() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    seeds = torch.tensor(
        [[0.2, 0.3], [0.8, 0.3]], dtype=torch.float64, requires_grad=True
    )
    # Make endpoint geometry depend on seeds, as it does in the decoder.
    vertices = torch.stack((seeds[0] + 0.1, seeds[1] - 0.1))
    edges = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    seed_pairs = torch.tensor([[0, 1], [-1, -1]], dtype=torch.long)

    curves = decoder.sample_smooth_edge_curves_uv(
        seeds, vertices, edges, seed_pairs, n_samples=11
    )

    assert curves.shape == (2, 11, 2)
    assert torch.allclose(curves[:, 0], vertices[edges[:, 0]])
    assert torch.allclose(curves[:, -1], vertices[edges[:, 1]])
    # The invalid-pair fallback follows the straight endpoint chord.
    fallback_offsets = curves[1] - vertices[0]
    fallback_cross = (
        fallback_offsets[:, 0] * (vertices[1] - vertices[0])[1]
        - fallback_offsets[:, 1] * (vertices[1] - vertices[0])[0]
    )
    assert torch.allclose(fallback_cross, torch.zeros_like(fallback_cross))

    curves.square().sum().backward()
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert torch.linalg.vector_norm(seeds.grad) > 0


def test_scipy_forward_includes_differentiable_smooth_edge_curves() -> None:
    seeds = irregular_test_seeds().requires_grad_(True)
    decoder = ContinuousVoronoiDecoder(return_xyz=False)

    out = decoder(seeds, topology_mode="scipy", return_xyz=False)
    curves = out["edge_curves_uv"]

    assert curves.shape == (out["graph"]["edge_index"].shape[0], 64, 2)
    curves.square().mean().backward()
    assert seeds.grad is not None
    assert torch.isfinite(seeds.grad).all()
    assert torch.linalg.vector_norm(seeds.grad) > 0


def test_smooth_edge_curves_xyz_uses_torch_cad_evaluator() -> None:
    class DummyTorchCadDomain:
        @staticmethod
        def eval_uv_norm_batch_torch(uv: torch.Tensor) -> torch.Tensor:
            return torch.cat((uv, (uv[:, :1] ** 2) + uv[:, 1:2]), dim=1)

    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    curves_uv = torch.rand((3, 7, 2), dtype=torch.float64, requires_grad=True)
    curves_xyz = decoder.sample_smooth_edge_curves_xyz(
        DummyTorchCadDomain(), curves_uv
    )

    assert curves_xyz.shape == (3, 7, 3)
    curves_xyz.sum().backward()
    assert curves_uv.grad is not None
    assert torch.isfinite(curves_uv.grad).all()


def test_boundary_box_edge_sampling_stays_on_box_boundary() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    dtype = torch.float64

    same_side = decoder.sample_boundary_box_edge_uv(
        torch.tensor([0.0, 0.2], dtype=dtype),
        torch.tensor([0.0, 0.8], dtype=dtype),
        n_samples=17,
    )
    around_corner = decoder.sample_boundary_box_edge_uv(
        torch.tensor([0.0, 0.4], dtype=dtype),
        torch.tensor([0.7, 0.0], dtype=dtype),
        n_samples=18,
    )

    for curve in (same_side, around_corner):
        on_boundary = (
            torch.isclose(curve[:, 0], torch.zeros_like(curve[:, 0]), atol=1e-5)
            | torch.isclose(curve[:, 0], torch.ones_like(curve[:, 0]), atol=1e-5)
            | torch.isclose(curve[:, 1], torch.zeros_like(curve[:, 1]), atol=1e-5)
            | torch.isclose(curve[:, 1], torch.ones_like(curve[:, 1]), atol=1e-5)
        )
        assert on_boundary.all()


def test_graph_edge_curve_sampling_dispatches_only_shell_edges_to_box() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    seeds = torch.tensor(
        [[0.2, 0.3], [0.8, 0.3]], dtype=torch.float64, requires_grad=True
    )
    nodes = torch.tensor(
        [[0.0, 0.4], [0.7, 0.0], [0.5, 0.5]], dtype=torch.float64
    )
    graph = {
        "nodes_uv": nodes,
        "edge_index": torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
        "edge_seed_pair": torch.tensor([[-1, -1], [0, 1]], dtype=torch.long),
        "edge_type": torch.tensor([4, 1], dtype=torch.long),
    }

    curves = decoder.sample_graph_edge_curves_uv(seeds, graph, n_samples=32)
    shell = curves[0]
    shell_on_boundary = (
        torch.isclose(shell[:, 0], torch.zeros_like(shell[:, 0]), atol=1e-5)
        | torch.isclose(shell[:, 0], torch.ones_like(shell[:, 0]), atol=1e-5)
        | torch.isclose(shell[:, 1], torch.zeros_like(shell[:, 1]), atol=1e-5)
        | torch.isclose(shell[:, 1], torch.ones_like(shell[:, 1]), atol=1e-5)
    )
    assert shell_on_boundary.all()
    # Type 1 starts inside the box and must remain a Voronoi/Hermite edge.
    first = curves[1, 0]
    first_on_boundary = (
        torch.isclose(first[0], first.new_tensor(0.0))
        | torch.isclose(first[0], first.new_tensor(1.0))
        | torch.isclose(first[1], first.new_tensor(0.0))
        | torch.isclose(first[1], first.new_tensor(1.0))
    )
    assert not bool(first_on_boundary)


def test_scipy_shell_curves_stay_on_uv_box() -> None:
    seeds = irregular_test_seeds().requires_grad_(True)
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    out = decoder(seeds, topology_mode="scipy", return_xyz=False)
    edge_type = out["graph"]["edge_type"]
    curves = decoder.sample_graph_edge_curves_uv(
        seeds, out["graph"], n_samples=64
    )

    print(torch.bincount(edge_type))
    print(curves.shape)
    shell_curves = curves[edge_type == 4]
    assert shell_curves.shape[0] > 0
    on_boundary = (
        torch.isclose(shell_curves[..., 0], torch.zeros_like(shell_curves[..., 0]), atol=1e-5)
        | torch.isclose(shell_curves[..., 0], torch.ones_like(shell_curves[..., 0]), atol=1e-5)
        | torch.isclose(shell_curves[..., 1], torch.zeros_like(shell_curves[..., 1]), atol=1e-5)
        | torch.isclose(shell_curves[..., 1], torch.ones_like(shell_curves[..., 1]), atol=1e-5)
    )
    assert on_boundary.all()


def test_soft_tube_field_has_curve_and_radius_gradients() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    query_xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.2, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.get_default_dtype(),
    )
    curves_xyz = torch.tensor(
        [[[0.0, 0.1, 0.0], [0.5, 0.1, 0.0], [1.0, 0.1, 0.0]]],
        dtype=torch.get_default_dtype(),
        requires_grad=True,
    )
    log_radius = decoder.make_learnable_radius(0.02)
    radius = torch.nn.functional.softplus(log_radius)

    tube = decoder.soft_tube_occupancy(
        query_xyz=query_xyz,
        curves_xyz=curves_xyz,
        radius=radius,
        tau_distance=0.02,
        tau_occupancy=0.01,
    )
    larger = decoder.soft_tube_occupancy(
        query_xyz=query_xyz,
        curves_xyz=curves_xyz,
        radius=radius + 0.02,
        tau_distance=0.02,
        tau_occupancy=0.01,
    )

    assert tube["distance"].shape == (query_xyz.shape[0],)
    assert torch.all((tube["occupancy"] >= 0.0) & (tube["occupancy"] <= 1.0))
    assert torch.all(larger["occupancy"] >= tube["occupancy"])
    assert torch.allclose(radius.detach(), radius.new_tensor(0.02), atol=1e-6)

    loss = tube["occupancy"].mean() + 0.01 * radius
    loss.backward()
    assert curves_xyz.grad is not None
    assert torch.isfinite(curves_xyz.grad).all()
    assert log_radius.grad is not None
    assert torch.isfinite(log_radius.grad).all()


def test_curve_points_and_tangents_xyz_uses_finite_differences() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    curves = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    points, tangents = decoder.curve_points_and_tangents_xyz(curves)

    assert points.shape == (3, 3)
    assert tangents.shape == (3, 3)
    expected = torch.tensor([1.0, 0.0, 0.0], dtype=curves.dtype).expand_as(tangents)
    assert torch.allclose(tangents, expected)


def test_soft_tube_fem_fields_have_valid_angles_and_gradients() -> None:
    decoder = ContinuousVoronoiDecoder(return_xyz=False)
    elem_centers = torch.tensor(
        [[0.0, 0.1, 0.0], [0.5, 0.2, 0.0], [1.0, 0.1, 0.0]],
        dtype=torch.float64,
    )
    curves = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    log_radius = torch.nn.Parameter(
        decoder.make_learnable_radius(0.15).detach().to(dtype=torch.float64)
    )
    radius = torch.nn.functional.softplus(log_radius)
    fields = decoder.soft_tube_density_and_fiber_to_elements(
        elem_centers_xyz=elem_centers,
        curves_xyz=curves,
        radius=radius,
        tau_distance=0.02,
        tau_density=0.02,
        tau_fiber=0.02,
        rho_min=1e-3,
    )

    num_elements = elem_centers.shape[0]
    assert fields["density"].shape == (num_elements,)
    assert fields["fiber"].shape == (num_elements, 3)
    assert fields["phi"].shape == (num_elements,)
    assert fields["theta"].shape == (num_elements,)
    assert fields["distance"].shape == (num_elements,)
    assert torch.all(fields["density"] >= 1e-3)
    assert torch.all(fields["density"] <= 1.0)
    assert torch.isfinite(fields["phi"]).all()
    assert torch.isfinite(fields["theta"]).all()
    assert torch.allclose(
        torch.linalg.vector_norm(fields["fiber"], dim=1),
        torch.ones(num_elements, dtype=curves.dtype),
    )
    assert torch.allclose(fields["fiber"][:, 0], torch.ones(num_elements, dtype=curves.dtype))

    loss = fields["density"].mean() + fields["phi"].square().mean() + 0.01 * radius
    loss.backward()
    assert curves.grad is not None and torch.isfinite(curves.grad).all()
    assert log_radius.grad is not None and torch.isfinite(log_radius.grad).all()


if __name__ == "__main__":
    test_scipy_topology_seed_gradients_with_area_loss()
    test_scipy_reconstructed_graph_geometry_has_seed_gradients()
