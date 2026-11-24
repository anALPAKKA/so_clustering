"""Shape-oriented clustering of distributions (histograms).

The algorithm is a k-means-style loop in which the assignment step uses the
Earth Mover's (1-D Wasserstein) distance instead of the Euclidean distance,
so samples are grouped by the *shape* of their distribution rather than by
bin-by-bin differences.

1. Initialise centroids with k-means++ (scikit-learn ``KMeans``).
2. Assign every histogram to the centroid with the smallest Wasserstein
   distance.
3. Update each centroid as the mean histogram of its members (an empty
   cluster keeps its previous centroid).
4. Repeat 2-3 until the largest centroid shift is below ``tol`` or
   ``max_iter`` is reached.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

__all__ = ["ShapeOrientedClustering", "HybridClustering", "wasserstein_matrix"]


def _check_bin_centers(bin_centers, n_bins):
    bin_centers = np.asarray(bin_centers, dtype=float)
    if bin_centers.ndim != 1 or bin_centers.size != n_bins:
        raise ValueError(
            f"bin_centers must be 1-D with {n_bins} entries, got shape {bin_centers.shape}"
        )
    if np.any(np.diff(bin_centers) <= 0):
        raise ValueError("bin_centers must be strictly increasing")
    return bin_centers


def wasserstein_matrix(X, centroids, bin_centers=None):
    """Wasserstein distance between every row of ``X`` and every centroid.

    Parameters
    ----------
    X : array of shape (n_samples, n_bins)
    centroids : array of shape (n_clusters, n_bins)
    bin_centers : array of shape (n_bins,), optional
        * ``None`` (default): each row is treated as a set of sample values,
          i.e. ``scipy.stats.wasserstein_distance(row, centroid)``. This is
          the behaviour of the original ``HybridClustering`` code.
        * array: each row is treated as weights (counts or densities) on the
          given bin positions, i.e.
          ``wasserstein_distance(bin_centers, bin_centers, row, centroid)``.
          Rows are normalised to sum to one.

    Returns
    -------
    D : array of shape (n_samples, n_clusters)

    Both modes are computed in closed form (vectorised) and match
    ``scipy.stats.wasserstein_distance`` to floating-point precision.
    """
    X = np.asarray(X, dtype=float)
    C = np.asarray(centroids, dtype=float)

    if bin_centers is None:
        # Equal-size, equally weighted samples: W1 = mean |sort(u) - sort(v)|
        Xs = np.sort(X, axis=1)
        Cs = np.sort(C, axis=1)
        return np.abs(Xs[:, None, :] - Cs[None, :, :]).mean(axis=2)

    x = _check_bin_centers(bin_centers, X.shape[1])
    checked = []
    for name, A in (("histograms", X), ("centroids", C)):
        scale = max(np.abs(A).max(), 1.0)
        if np.any(A < -1e-9 * scale):
            raise ValueError(f"{name} must be non-negative when bin_centers is given")
        A = np.clip(A, 0.0, None)  # remove floating-point round-off
        if np.any(A.sum(axis=1) <= 0):
            raise ValueError(f"every row of {name} must have a positive sum")
        checked.append(A)
    X, C = checked
    # W1 = integral |CDF_u - CDF_v| dx on the common support
    Xc = np.cumsum(X / X.sum(axis=1, keepdims=True), axis=1)[:, :-1]
    Cc = np.cumsum(C / C.sum(axis=1, keepdims=True), axis=1)[:, :-1]
    dx = np.diff(x)
    return (np.abs(Xc[:, None, :] - Cc[None, :, :]) * dx).sum(axis=2)


class ShapeOrientedClustering:
    """Cluster histograms by distribution shape using the Wasserstein distance.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int, default 100
        Maximum number of assign/update iterations.
    tol : float, default 1e-4
        Convergence threshold on the largest centroid shift (Euclidean norm).
    bin_centers : array-like of shape (n_bins,), optional
        Positions of the histogram bins. If ``None`` (default) the rows are
        compared as sets of values, exactly as in the original code. If
        given, the rows are compared as weights on these bin positions.
        See :func:`wasserstein_matrix`.
    random_state : int or None, default 42
        Seed for the k-means++ initialisation.

    Attributes
    ----------
    centroids : ndarray of shape (n_clusters, n_bins)
        Mean histogram of each cluster.
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each training histogram (also ``assignments``).
    n_iter_ : int
        Number of iterations run.
    converged_ : bool
        Whether the centroid shift fell below ``tol``.
    """

    def __init__(self, n_clusters, max_iter=100, tol=1e-4, bin_centers=None,
                 random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.bin_centers = bin_centers
        self.random_state = random_state
        self.centroids = None

    def _distances(self, X):
        return wasserstein_matrix(X, self.centroids, self.bin_centers)

    def fit(self, histograms):
        """Fit the model to ``histograms`` of shape (n_samples, n_bins)."""
        X = np.asarray(histograms, dtype=float)
        if X.ndim != 2:
            raise ValueError("histograms must be a 2-D array (n_samples, n_bins)")
        if X.shape[0] < self.n_clusters:
            raise ValueError("need at least n_clusters histograms")

        init = KMeans(n_clusters=self.n_clusters, init="k-means++",
                      random_state=self.random_state).fit(X)
        self.centroids = init.cluster_centers_
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            labels = np.argmin(self._distances(X), axis=1)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            shift = np.linalg.norm(self.centroids - new_centroids, axis=1).max()
            self.centroids = new_centroids
            if shift < self.tol:
                self.converged_ = True
                break

        self.n_iter_ = iteration
        self.labels_ = labels
        self.assignments = labels  # name used in the original code
        return self

    def predict(self, histograms):
        """Index of the nearest centroid (Wasserstein distance) for each histogram."""
        if self.centroids is None:
            raise RuntimeError("call fit() before predict()")
        return np.argmin(self._distances(histograms), axis=1)

    def fit_predict(self, histograms):
        """Fit the model and return the cluster labels."""
        return self.fit(histograms).labels_

    def transform(self, histograms):
        """Wasserstein distance from each histogram to each centroid."""
        if self.centroids is None:
            raise RuntimeError("call fit() before transform()")
        return self._distances(histograms)


# Backwards-compatible name used in the original analysis code.
HybridClustering = ShapeOrientedClustering
