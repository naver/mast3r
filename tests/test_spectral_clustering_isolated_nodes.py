"""Normalized spectral bases must remain defined at zero graph degree."""

import pytest
import torch

from mast3r.cloud_opt.sparse_ga import sim_func, spectral_clustering


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_isolated_vertex_preserves_component_spectrum(dtype):
    graph = torch.tensor([[0, 3, 0], [3, 0, 0], [0, 0, 0]], dtype=dtype)
    eigenvalues, eigenvectors = spectral_clustering(graph, normalized_cuts=True)
    expected = torch.tensor([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], dtype=dtype)
    torch.testing.assert_close(eigenvalues, torch.tensor([0, 0, 2], dtype=dtype))
    torch.testing.assert_close(eigenvectors.T @ eigenvectors, torch.eye(3, dtype=dtype))
    torch.testing.assert_close(
        eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T, expected
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("size", [1, 4])
def test_edgeless_graph_has_finite_zero_laplacian(dtype, size):
    graph = torch.eye(size, dtype=dtype)
    eigenvalues, eigenvectors = spectral_clustering(graph, normalized_cuts=True)
    torch.testing.assert_close(eigenvalues, torch.zeros(size, dtype=dtype))
    torch.testing.assert_close(
        eigenvectors.T @ eigenvectors, torch.eye(size, dtype=dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_underflowed_gaussian_affinities_remain_valid(dtype):
    points = torch.tensor([[0, 0, 1], [1000, 0, 1]], dtype=dtype)
    graph = sim_func(points[:, None], points[None, :], gamma=7)
    assert graph[0, 1] == 0
    eigenvalues, eigenvectors = spectral_clustering(graph, normalized_cuts=True)
    torch.testing.assert_close(eigenvalues, torch.zeros(2, dtype=dtype))
    assert torch.isfinite(eigenvectors).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positive_subnormal_degree_is_not_clamped(dtype):
    weight = torch.finfo(dtype).tiny / 16
    graph = torch.tensor([[0, weight], [weight, 0]], dtype=dtype)
    eigenvalues, eigenvectors = spectral_clustering(graph, normalized_cuts=True)
    torch.testing.assert_close(eigenvalues, torch.tensor([0, 2], dtype=dtype))
    assert torch.isfinite(eigenvectors).all()


def test_connected_graph_matches_original_normalized_laplacian():
    graph = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 4.0], [2.0, 4.0, 0.0]], dtype=torch.float64
    )
    inverse_degree = torch.diag(graph.sum(-1).rsqrt())
    expected = inverse_degree @ (torch.diag(graph.sum(-1)) - graph) @ inverse_degree
    eigenvalues, eigenvectors = spectral_clustering(graph.clone(), normalized_cuts=True)
    torch.testing.assert_close(eigenvalues, torch.linalg.eigvalsh(expected))
    torch.testing.assert_close(
        eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T, expected
    )


def test_unnormalized_path_and_truncated_basis_are_preserved():
    graph = torch.tensor(
        [[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    values, vectors = spectral_clustering(graph.clone(), normalized_cuts=False)
    torch.testing.assert_close(values, torch.tensor([0.0, 0.0, 6.0], dtype=graph.dtype))
    values, vectors = spectral_clustering(graph.clone(), k=2, normalized_cuts=True)
    assert values.shape == (2,)
    assert vectors.shape == (3, 2)
    torch.testing.assert_close(values, torch.zeros(2, dtype=graph.dtype))
