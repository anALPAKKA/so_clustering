"""so_clustering: shape-oriented clustering of distributions."""

from .clustering import HybridClustering, ShapeOrientedClustering, wasserstein_matrix

__all__ = ["ShapeOrientedClustering", "HybridClustering", "wasserstein_matrix"]
__version__ = "0.1.0"
