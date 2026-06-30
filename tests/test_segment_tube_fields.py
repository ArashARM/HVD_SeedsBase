import torch

from Decoder_CLasses.ContinuousVoronoiDecoder import ContinuousVoronoiDecoder


def test_straight_segment_along_x_gives_x_fiber():
    decoder = ContinuousVoronoiDecoder(use_spatial_pruning=False)
    curves = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    query = torch.tensor([[0.25, 0.01, 0.0], [0.75, -0.02, 0.0]], dtype=torch.float64)

    out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        curves,
        radius=0.1,
        tau_density=0.02,
        tau_fiber=0.02,
    )

    expected = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(out["fiber"], expected, atol=1e-6)


def test_density_decreases_with_distance_from_segment():
    decoder = ContinuousVoronoiDecoder(use_spatial_pruning=False)
    curves = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    query = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.15, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32)

    out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        curves,
        radius=0.1,
        tau_density=0.03,
        tau_fiber=0.03,
    )

    assert out["density"][0] > out["density"][1] > out["density"][2]
    assert out["distance"][0] < out["distance"][1] < out["distance"][2]


def test_segment_outputs_match_sampled_output_shapes():
    query = torch.tensor(
        [[0.2, 0.0, 0.0], [0.5, 0.1, 0.0], [1.2, 0.0, 0.0]],
        dtype=torch.float32,
    )
    curves = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    segment_decoder = ContinuousVoronoiDecoder(use_segment_distance=True, use_spatial_pruning=False)
    sampled_decoder = ContinuousVoronoiDecoder(use_segment_distance=False)

    segment_out = segment_decoder.soft_tube_density_and_fiber_to_elements(query, curves, radius=0.1)
    sampled_out = sampled_decoder.soft_tube_density_and_fiber_to_elements(query, curves, radius=0.1)

    for key in ("density", "fiber", "phi", "theta", "distance"):
        assert segment_out[key].shape == sampled_out[key].shape


def test_empty_and_degenerate_curves_have_no_nans():
    decoder = ContinuousVoronoiDecoder()
    query = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    fallback = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)

    empty_curves = query.new_empty((0, 2, 3))
    empty_out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        empty_curves,
        radius=0.1,
        fallback_fiber=fallback,
    )

    degenerate_curves = torch.zeros((1, 2, 3), dtype=torch.float64)
    degenerate_out = decoder.soft_tube_density_and_fiber_to_elements(
        query,
        degenerate_curves,
        radius=0.1,
        fallback_fiber=fallback,
    )

    for out in (empty_out, degenerate_out):
        for key in ("density", "fiber", "phi", "theta", "distance"):
            finite_or_inf = torch.isfinite(out[key]) | torch.isinf(out[key])
            assert finite_or_inf.all()
            assert not torch.isnan(out[key]).any()
