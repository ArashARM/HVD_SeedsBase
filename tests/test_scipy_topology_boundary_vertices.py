import torch
import matplotlib.pyplot as plt

from Decoder_CLasses.ContinuousVoronoiDecoder import ContinuousVoronoiDecoder


def test_scipy_topology_returns_unified_boundary_graph_fields():
    decoder = ContinuousVoronoiDecoder()
    seeds = torch.tensor(
        [
            [0.15, 0.15],
            [0.85, 0.15],
            [0.15, 0.85],
            [0.85, 0.85],
        ],
        dtype=torch.float32,
    )

    out = decoder(seeds, topology_mode="scipy", return_xyz=False)

    assert "vertex_type" in out
    assert "edge_type" in out["edges"]
    assert out["vertices_uv"].shape[0] == out["graph"]["nodes_uv"].shape[0]
    assert out["graph"]["node_type"].shape[0] == out["vertices_uv"].shape[0]
    assert out["edges"]["edge_index"].shape[1] == 2
    assert out["edges"]["edge_type"].shape[0] == out["edges"]["edge_index"].shape[0]


def test_scipy_topology_prunes_unreferenced_vertices_and_reports_final_counts():
    decoder = ContinuousVoronoiDecoder()
    seeds = torch.tensor(
        [
            [0.10, 0.10],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.90, 0.90],
            [0.50, 0.45],
            [0.20, 0.55],
            [0.80, 0.60],
        ],
        dtype=torch.float64,
    )

    out = decoder(seeds, topology_mode="scipy", return_xyz=False)
    diagnostics = out["diagnostics"]
    num_vertices = out["vertices_uv"].shape[0]
    edges = out["edges"]["edge_index"]

    assert torch.all(out["vertex_degree"] > 0)
    assert edges.numel() == 0 or int(edges.max()) < num_vertices
    assert out["isolated_vertices"].numel() == 0
    assert num_vertices == diagnostics["num_final_vertices"]
    assert num_vertices == (
        diagnostics["num_final_interior_vertices"]
        + diagnostics["num_final_boundary_vertices"]
    )
    assert num_vertices == (
        diagnostics["num_raw_scipy_vertices"]
        + diagnostics["num_raw_boundary_vertices"]
        - diagnostics["num_pruned_vertices"]
    )
    assert out["graph"]["num_interior_nodes"] == diagnostics["num_final_interior_vertices"]
    assert out["graph"]["num_boundary_nodes"] == diagnostics["num_final_boundary_vertices"]

    for references in (out["boundary_origin_vertex"], out["boundary_target_vertex"]):
        assert torch.all(
            (references == -1)
            | ((references >= 0) & (references < num_vertices))
        )


def test_scipy_topology_can_keep_isolated_vertices_for_debugging():
    decoder = ContinuousVoronoiDecoder()
    seeds = torch.tensor(
        [
            [0.10, 0.10],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.90, 0.90],
            [0.50, 0.45],
            [0.20, 0.55],
            [0.80, 0.60],
        ],
        dtype=torch.float64,
    )

    out = decoder(
        seeds,
        topology_mode="scipy",
        return_xyz=False,
        keep_isolated_vertices=True,
    )
    diagnostics = out["diagnostics"]

    assert diagnostics["num_pruned_vertices"] == 0
    assert out["vertices_uv"].shape[0] == (
        diagnostics["num_raw_scipy_vertices"]
        + diagnostics["num_raw_boundary_vertices"]
    )


def test_generated_graph_plot_labels_compact_node_and_edge_ids(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    decoder = ContinuousVoronoiDecoder()
    seeds = torch.tensor(
        [
            [0.10, 0.10], [0.90, 0.10], [0.10, 0.90], [0.90, 0.90],
            [0.50, 0.45], [0.20, 0.55], [0.80, 0.60],
        ],
        dtype=torch.float64,
    )
    out = decoder(seeds, topology_mode="scipy", return_xyz=False)

    fig, ax = decoder.plot_graph_output(
        seeds,
        out,
        show_node_ids=True,
        show_edge_ids=True,
        print_node_table=False,
    )
    labels = {text.get_text() for text in ax.texts}
    source_types = out["graph"]["boundary_source_type"]
    expected_node_labels = {
        (
            "I" if int(node_type) == 0
            else "C" if int(source_types[node_id]) == 4
            else "B"
        ) + str(node_id)
        for node_id, node_type in enumerate(out["graph"]["node_type"])
    }
    num_edges = out["graph"]["edge_index"].shape[0]

    assert expected_node_labels <= labels
    assert {f"e{edge_id}" for edge_id in range(num_edges)} <= labels
    assert f"nodes={out['graph']['nodes_uv'].shape[0]}" in ax.get_title()
    plt.close(fig)


def test_box_boundary_shell_is_closed_and_pair_intersections_are_attached():
    decoder = ContinuousVoronoiDecoder()
    seeds = torch.tensor([[0.25, 0.50], [0.75, 0.50]], dtype=torch.float64)

    for mode in ("soft", "scipy"):
        out = decoder(seeds, topology_mode=mode, return_xyz=False)
        graph = out["graph"]
        edges = graph["edge_index"]
        edge_type = graph["edge_type"]
        source_type = graph["boundary_source_type"]
        boundary_ids = torch.nonzero(graph["node_type"] == 1, as_tuple=False).flatten()
        corner_ids = torch.nonzero(source_type == 4, as_tuple=False).flatten()
        pair_ids = torch.nonzero(source_type == 3, as_tuple=False).flatten()

        assert corner_ids.numel() == 4
        assert pair_ids.numel() == 2
        assert all(
            graph["boundary_source_name"][node_id] == "pair_bisector_boundary"
            for node_id in pair_ids.tolist()
        )
        assert int((edge_type == 2).sum()) == boundary_ids.numel()

        for node_id in boundary_ids.tolist():
            incident = (edges == node_id).any(dim=1)
            assert int((incident & (edge_type == 2)).sum()) == 2
            if int(source_type[node_id]) != 4:
                assert int((incident & (edge_type != 2)).sum()) >= 1
