"""Is a layout meaningful? Permutation test for whether features that belong
together (same semantic group) sit closer on the grid than chance."""
from __future__ import annotations

import itertools

import numpy as np


def feature_positions(feature_map: np.ndarray) -> dict:
    """{feature index: (row, col)} for every placed feature."""
    return {int(f): (i, j) for (i, j), f in np.ndenumerate(feature_map) if f >= 0}


def mean_grid_distance(pos: dict, pairs) -> float:
    return float(np.mean([abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]) for a, b in pairs]))


def grouping_test(feature_map: np.ndarray, groups, n_perm=2000, seed=0) -> dict:
    """Do features with the same group label lie closer together than if the
    same occupied cells were assigned to features at random?

    groups : sequence of length d with a group label per feature (None = no group)
    Returns mean Manhattan distance of same-group pairs, the permutation null
    mean/std, and a one-sided p-value (fraction of shuffles at least as tight).
    """
    pos = feature_positions(feature_map)
    d = len(groups)
    if len(pos) < d:
        raise ValueError(f"layout places {len(pos)} of {d} features; the test needs a lossless layout")
    pairs = [(a, b) for a, b in itertools.combinations(range(d), 2)
             if groups[a] is not None and groups[a] == groups[b]]
    if not pairs:
        raise ValueError("no two features share a group")
    observed = mean_grid_distance(pos, pairs)
    cells = [pos[f] for f in range(d)]
    rng = np.random.default_rng(seed)
    null = np.array([mean_grid_distance({int(p): cells[k] for k, p in enumerate(rng.permutation(d))}, pairs)
                     for _ in range(n_perm)])
    return dict(observed=observed, null_mean=float(null.mean()), null_std=float(null.std()),
                p=float((null <= observed).mean()), n_pairs=len(pairs))


def render_layout(feature_map: np.ndarray, labels, width=6) -> str:
    """Text picture of a layout with a short label per feature."""
    return "\n".join(" ".join(f"{labels[f] if f >= 0 else '.':>{width}s}" for f in row) for row in feature_map)
