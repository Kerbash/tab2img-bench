"""Who wins where: split datasets by a characteristic (default d/N) and
compare methods inside each regime."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tab2img.analysis.stats import average_ranks, compare_to_best, friedman


def split_by(results, column="dN", cutoff=1.0, characteristics: pd.DataFrame | None = None) -> dict:
    """{label: ResultSet} for `column >= cutoff` and `< cutoff`.
    `column` is looked up in `characteristics` if given, else in results.info."""
    src = characteristics if characteristics is not None else results.info
    v = src.loc[results.scores.index, column].astype(float)
    hi, lo = v.index[v >= cutoff], v.index[v < cutoff]
    return {f"{column}>={cutoff:g}": results.select(datasets=hi),
            f"{column}<{cutoff:g}": results.select(datasets=lo)}


def regime_table(results, column="dN", cutoff=1.0, characteristics=None) -> pd.DataFrame:
    """Average rank of each method per regime + pooled, worst-regime rank and
    swing (how much the rank changes between regimes; small = versatile)."""
    parts = split_by(results, column, cutoff, characteristics)
    t = pd.DataFrame({lab: average_ranks(rs.scores) for lab, rs in parts.items()})
    t["pooled"] = average_ranks(results.scores)
    regs = list(parts)
    t["worst_regime"] = t[regs].max(axis=1)
    t["swing"] = (t[regs[0]] - t[regs[1]]).abs()
    return t.sort_values("worst_regime")


def regime_tests(results, column="dN", cutoff=1.0, characteristics=None) -> dict:
    """{regime: (friedman (chi2, p), compare_to_best table)}."""
    return {lab: (friedman(rs.scores), compare_to_best(rs.scores))
            for lab, rs in split_by(results, column, cutoff, characteristics).items()}


def cutoff_sensitivity(results, cutoffs=(0.5, 1.0, 2.0, 4.0), column="dN", top=4,
                       characteristics=None) -> pd.DataFrame:
    """Top methods on the `column >= cutoff` side for several cutoffs."""
    rows = []
    for c in cutoffs:
        rs = list(split_by(results, column, c, characteristics).values())[0]
        r = average_ranks(rs.scores).head(top)
        rows.append(dict(cutoff=c, n=len(rs.scores), **{f"#{i + 1}": f"{m} ({v:.2f})"
                                                         for i, (m, v) in enumerate(r.items())}))
    return pd.DataFrame(rows).set_index("cutoff")


def rank_correlations(results, characteristics: pd.DataFrame,
                      columns=("N", "d", "dN", "classes", "mean_abs_corr", "redundancy"),
                      log_columns=("N", "d", "dN")) -> pd.DataFrame:
    """Spearman rho between each method's per-dataset rank and each dataset
    characteristic (negative rho = the method ranks better as it grows)."""
    ranks = results.scores.rank(axis=1, ascending=False)
    ch = characteristics.loc[ranks.index]
    out = {}
    for col in columns:
        x = np.log10(ch[col].astype(float)) if col in log_columns else ch[col].astype(float)
        out[col] = {m: stats.spearmanr(x, ranks[m]).statistic for m in ranks.columns}
    return pd.DataFrame(out)
