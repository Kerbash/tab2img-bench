"""Figures: critical-difference diagram, layout picture, learning curves."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tab2img.analysis.stats import average_ranks, critical_difference


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def cd_diagram(scores: pd.DataFrame, path=None, alpha=0.05, title=None):
    """Demsar-style critical-difference diagram (Nemenyi). Returns the figure."""
    plt = _plt()
    n, k = scores.shape
    r = average_ranks(scores)
    cd = critical_difference(k, n, alpha)
    fig, ax = plt.subplots(figsize=(9, 0.45 * k + 2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, k + 2)
    ax.invert_xaxis()
    ax.axis("off")
    ax.plot([1, k], [k + 1, k + 1], "k-")
    for t in range(1, k + 1):
        ax.plot([t, t], [k + 1, k + 1.15], "k-")
        ax.text(t, k + 1.35, str(t), ha="center")
    mid = (1 + k) / 2
    for i, (m, v) in enumerate(r.items()):
        y = k - i
        left = v < mid
        end = 1 if left else k
        ax.plot([v, v], [k + 1, y], "k-", lw=0.8)
        ax.plot([v, end], [y, y], "k-", lw=0.8)
        ax.text(end + (-0.1 if left else 0.1), y, f"{m} ({v:.2f})", va="center",
                ha="left" if left else "right")
    # tie bars: maximal groups of methods whose rank spread is < CD
    vals = r.values
    groups = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    for g, (i, j) in enumerate(groups):
        y = k + 0.6 - 0.18 * (g + 1)
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [y, y], "-", lw=3, color="#d62728")
    ax.plot([k, k - cd], [k + 1.75, k + 1.75], "k-", lw=2)
    ax.text(k - cd / 2, k + 1.9, f"CD = {cd:.2f}", ha="center")
    ax.set_title(title or f"Critical-difference diagram (N={n} datasets, Nemenyi, alpha={alpha})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def layout_image(feature_map: np.ndarray, groups, names=None, path=None, title=""):
    """Colour each pixel by the group of the feature it shows."""
    plt = _plt()
    labels = sorted({g for g in groups if g is not None})
    cmap = plt.get_cmap("tab10")
    rows, cols = feature_map.shape
    fig, ax = plt.subplots(figsize=(cols * 0.8 + 1, rows * 0.8 + 1))
    for (i, j), f in np.ndenumerate(feature_map):
        g = groups[f] if f >= 0 else None
        color = cmap(labels.index(g) % 10) if g is not None else "#eeeeee"
        ax.add_patch(plt.Rectangle((j, rows - 1 - i), 1, 1, color=color, ec="white"))
        if f >= 0 and names is not None:
            ax.text(j + 0.5, rows - 0.5 - i, str(names[f])[:6], ha="center", va="center", fontsize=7)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def learning_curves(results, path=None, methods=None):
    """Mean F1 vs N for learning-curve runs named `<dataset>_n<N>_s<seed>`."""
    plt = _plt()
    s = results.scores if methods is None else results.scores[methods]
    idx = s.index.to_series().str.extract(r"^(?P<ds>.+)_n(?P<n>\d+)_s\d+$")
    s = s.assign(ds=idx["ds"].values, n=idx["n"].astype(float).values).dropna(subset=["ds"])
    g = s.groupby(["ds", "n"]).mean()
    dsets = g.index.get_level_values(0).unique()
    fig, axes = plt.subplots(1, len(dsets), figsize=(4 * len(dsets), 3.2), squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        sub = g.loc[ds]
        for m in sub.columns:
            ax.plot(sub.index, sub[m], marker="o", label=m)
        ax.set_xscale("log")
        ax.set_title(ds)
        ax.set_xlabel("N (training + test rows)")
    axes[0][0].set_ylabel("weighted F1")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
