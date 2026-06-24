from __future__ import annotations

import numpy as np
import torch

from serf_b1k.mapping.utils.visualization import TorchPCA

PCA_CHANNEL_PERMUTATIONS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]

PCA_CHANNEL_LABELS = [
    "RGB (default)",
    "RBG",
    "GRB",
    "GBR",
    "BRG",
    "BGR",
]


def permute_pca_colors(colors: np.ndarray, permutation: tuple[int, int, int] = (0, 1, 2)) -> np.ndarray:
    return colors[:, list(permutation)]


def compute_pca_colors_higher(
    features: torch.Tensor,
    skip_components: int = 3,
    n_components: int = 3,
    whiten: bool = True,
    quantile_low: float = 0.02,
    quantile_high: float = 0.98,
) -> np.ndarray:
    total = skip_components + n_components
    pca = TorchPCA(n_components=total, whiten=whiten)
    all_colors = pca.fit_transform(features.detach())

    colors = all_colors[:, skip_components:total]
    q_low = np.quantile(colors, quantile_low, axis=0)
    q_high = np.quantile(colors, quantile_high, axis=0)
    colors = np.clip(colors, q_low, q_high)
    return (colors - q_low) / (q_high - q_low + 1e-8)

