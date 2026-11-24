import numpy as np
import pytest
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans

from so_clustering import HybridClustering, ShapeOrientedClustering, wasserstein_matrix

rng = np.random.default_rng(0)
X = rng.random((40, 15))
C = rng.random((4, 15))
bins = np.linspace(-1.0, 2.0, 15)


def test_values_mode_matches_scipy():
    D = wasserstein_matrix(X, C)
    ref = np.array([[wasserstein_distance(x, c) for c in C] for x in X])
    np.testing.assert_allclose(D, ref, rtol=1e-12, atol=1e-12)


def test_bins_mode_matches_scipy():
    D = wasserstein_matrix(X, C, bin_centers=bins)
    ref = np.array([[wasserstein_distance(bins, bins, x, c) for c in C] for x in X])
    np.testing.assert_allclose(D, ref, rtol=1e-12, atol=1e-12)


def _original_hybrid(histograms, n_clusters, max_iter=100, tol=1e-4):
    """Verbatim logic of the original HybridClustering, for regression testing."""
    centroids = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42).fit(histograms).cluster_centers_
    for _ in range(max_iter):
        assignments = np.array([
            np.argmin([wasserstein_distance(h, c) for c in centroids]) for h in histograms
        ])
        new = np.array([
            histograms[assignments == i].mean(axis=0) if np.any(assignments == i) else centroids[i]
            for i in range(n_clusters)
        ])
        shift = np.linalg.norm(centroids - new, axis=1).max()
        centroids = new
        if shift < tol:
            break
    return assignments, centroids


def test_reproduces_original_algorithm():
    labels_ref, cent_ref = _original_hybrid(X, 3)
    model = HybridClustering(n_clusters=3).fit(X)
    np.testing.assert_array_equal(model.assignments, labels_ref)
    np.testing.assert_allclose(model.centroids, cent_ref)


def test_separates_shapes_with_bins():
    edges = np.linspace(-4, 4, 41)
    centers = 0.5 * (edges[:-1] + edges[1:])
    normal = [np.histogram(rng.normal(0, 1, 500), edges)[0] for _ in range(20)]
    skewed = [np.histogram(rng.gamma(2, 0.7, 500) - 1.4, edges)[0] for _ in range(20)]
    H = np.array(normal + skewed, dtype=float)
    labels = ShapeOrientedClustering(2, bin_centers=centers).fit_predict(H)
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1
    assert labels[0] != labels[-1]


def test_predict_and_transform_shapes():
    model = ShapeOrientedClustering(3).fit(X)
    assert model.predict(X).shape == (40,)
    assert model.transform(X).shape == (40, 3)
    np.testing.assert_array_equal(model.predict(X), np.argmin(model.transform(X), axis=1))


def test_input_validation():
    with pytest.raises(ValueError):
        ShapeOrientedClustering(3, bin_centers=bins[::-1]).fit(X)
    with pytest.raises(RuntimeError):
        ShapeOrientedClustering(3).predict(X)
