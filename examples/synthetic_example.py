"""Cluster synthetic histograms of three distribution shapes.

Run from the repository root:  python examples/synthetic_example.py
"""

import matplotlib.pyplot as plt
import numpy as np

from so_clustering import ShapeOrientedClustering

rng = np.random.default_rng(1)
edges = np.linspace(-4, 6, 51)
centers = 0.5 * (edges[:-1] + edges[1:])


def hist(samples):
    h, _ = np.histogram(samples, bins=edges, density=True)
    return h


# 30 histograms of each shape, each built from 400 random draws
groups = {
    "Gaussian": lambda: rng.normal(0, 1, 400),
    "right-skewed": lambda: rng.gamma(2.0, 1.0, 400) - 1.5,
    "bimodal": lambda: np.concatenate([rng.normal(-1.8, 0.5, 200), rng.normal(1.8, 0.5, 200)]),
}
H = np.array([hist(draw()) for draw in groups.values() for _ in range(30)])
truth = np.repeat(np.arange(len(groups)), 30)

model = ShapeOrientedClustering(n_clusters=3, bin_centers=centers).fit(H)
print(f"converged={model.converged_} after {model.n_iter_} iterations")
for k in range(3):
    members = truth[model.labels_ == k]
    print(f"cluster {k}: {len(members)} histograms, true groups {np.bincount(members, minlength=3)}")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for k, ax in enumerate(axes):
    for h in H[model.labels_ == k]:
        ax.step(centers, h, where="mid", color="0.75", lw=0.6)
    ax.step(centers, model.centroids[k], where="mid", color="C3", lw=2, label="centroid")
    ax.set_title(f"cluster {k} (n={np.sum(model.labels_ == k)})")
    ax.set_xlabel("value")
axes[0].set_ylabel("density")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig("examples/synthetic_clusters.png", dpi=150)
print("saved examples/synthetic_clusters.png")
