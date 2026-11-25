"""Shape-oriented clustering of distribution histograms.

import numpy as np
from sklearn.cluster import KMeans

__all__ = ["ShapeOrientedClustering", "so_clustering", "wasserstein_matrix"]


def wasserstein_matrix(X, centroids):
    """Wasserstein distance between every histogram profile and every centroid.

    Each row is treated as a set of bin heights, so bin location is ignored.
    Equivalent to ``scipy.stats.wasserstein_distance(row, centroid)`` for every
    pair, computed in one step as the mean absolute difference of the sorted
    bin heights.

    Parameters
    ----------
    X : array of shape (n_samples, n_bins)
    centroids : array of shape (n_clusters, n_bins)

    Returns
    -------
    array of shape (n_samples, n_clusters)
    """
    Xs = np.sort(np.asarray(X, dtype=float), axis=1)
    Cs = np.sort(np.asarray(centroids, dtype=float), axis=1)
    return np.abs(Xs[:, None, :] - Cs[None, :, :]).mean(axis=2)


class ShapeOrientedClustering:
    """Cluster histogram profiles by shape using the Wasserstein distance.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int, default 100
        Maximum number of assignment-update iterations.
    tol : float, default 1e-4
        Convergence tolerance on the largest centroid shift.
    random_state : int, default 42
        Seed for the k-means++ initialisation.

    Attributes
    ----------
    centroids : array of shape (n_clusters, n_bins)
        Average histogram profile of each cluster.
    labels_ : array of shape (n_samples,)
        Cluster index of each histogram (also available as ``assignments``).
    n_iter_ : int
        Number of iterations run.
    converged_ : bool
        Whether the centroid shift fell below ``tol``.
    """

    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, histograms):
        """Fit the model to histogram profiles of shape (n_samples, n_bins)."""
        X = np.asarray(histograms, dtype=float)

        # Initialisation
        self.centroids = KMeans(n_clusters=self.n_clusters, init="k-means++",
                                random_state=self.random_state).fit(X).cluster_centers_
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            # Assignment
            labels = np.argmin(wasserstein_matrix(X, self.centroids), axis=1)

            # Update
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            # Convergence
            shift = np.linalg.norm(self.centroids - new_centroids, axis=1).max()
            self.centroids = new_centroids
            if shift < self.tol:
                self.converged_ = True
                break

        self.n_iter_ = iteration
        self.labels_ = self.assignments = labels
        return self

    def predict(self, histograms):
        """Assign histogram profiles to the nearest centroid."""
        return np.argmin(wasserstein_matrix(histograms, self.centroids), axis=1)

    def fit_predict(self, histograms):
        """Fit the model and return the cluster labels."""
        return self.fit(histograms).labels_

    def transform(self, histograms):
        """Wasserstein distance from each histogram profile to each centroid."""
        return wasserstein_matrix(histograms, self.centroids)


# Short alias: so_clustering(n_clusters=3) is the same as ShapeOrientedClustering(n_clusters=3).
so_clustering = ShapeOrientedClustering
