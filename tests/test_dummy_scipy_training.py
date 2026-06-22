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


if __name__ == "__main__":
    test_scipy_topology_seed_gradients_with_area_loss()
    test_scipy_reconstructed_graph_geometry_has_seed_gradients()
