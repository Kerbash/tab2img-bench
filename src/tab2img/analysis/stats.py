"""Statistical comparison of classifiers over many datasets.

Across datasets (one score per method and dataset; Demsar 2006, JMLR):
  friedman, average_ranks, critical_difference (Nemenyi / Bonferroni-Dunn),
  compare_to_best (z-test on ranks vs the top method, Holm-adjusted),
  pairwise_wilcoxon (signed-rank on per-dataset scores, Holm-adjusted).

Within one dataset (paired fold scores):
  corrected_ttest (Nadeau & Bengio 2003; plain paired t-tests over CV folds
  are anti-conservative because training sets overlap), benjamini_hochberg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- p-value adjustment
def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order, m, out, run = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[idx]))
        out[idx] = run
    return out


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, float)
    order, q, prev = np.argsort(p), np.empty(len(p)), 1.0
    for i in range(len(p) - 1, -1, -1):
        prev = min(prev, p[order[i]] * len(p) / (i + 1))
        q[order[i]] = prev
    return q


# ---------------------------------------------------------------- across datasets
def average_ranks(scores: pd.DataFrame, higher_is_better=True) -> pd.Series:
    """Mean rank per method (1 = best; ties get the average rank)."""
    return scores.rank(axis=1, ascending=not higher_is_better, method="average").mean().sort_values()


def friedman(scores: pd.DataFrame) -> tuple[float, float]:
    chi2, p = stats.friedmanchisquare(*[scores[c] for c in scores.columns])
    return float(chi2), float(p)


def rank_se(k: int, n: int) -> float:
    """Standard error of a difference of two average ranks (k methods, n datasets)."""
    return float(np.sqrt(k * (k + 1) / (6.0 * n)))


def critical_difference(k: int, n: int, alpha=0.05, test="nemenyi") -> float:
    """Critical difference in average rank.

    nemenyi          all pairs:   q = studentized_range(1-alpha, k, inf) / sqrt(2)
    bonferroni-dunn  vs a control: q = z(1 - alpha / (2 (k - 1)))
    (these reproduce the tabulated values in Demsar 2006)
    """
    if test == "nemenyi":
        q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    elif test == "bonferroni-dunn":
        q = stats.norm.ppf(1 - alpha / (2 * (k - 1)))
    else:
        raise ValueError(test)
    return float(q * rank_se(k, n))


def compare_to_best(scores: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
    """Each method vs the best-ranked one: rank gap, z, raw and Holm p, verdict."""
    n, k = scores.shape
    r = average_ranks(scores)
    se = rank_se(k, n)
    best = r.index[0]
    rows = [dict(method=best, avg_rank=r[best], rank_diff=0.0, z=0.0, p=np.nan, p_holm=np.nan, verdict="best")]
    others = list(r.index[1:])
    z = (r[others] - r[best]) / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ph = holm(p)
    for m, zz, pp, hh in zip(others, z, p, ph):
        rows.append(dict(method=m, avg_rank=r[m], rank_diff=r[m] - r[best], z=zz, p=pp, p_holm=hh,
                         verdict="tie" if hh >= alpha else "worse"))
    return pd.DataFrame(rows).set_index("method")


def pairwise_wilcoxon(scores: pd.DataFrame, pairs) -> pd.DataFrame:
    """Signed-rank test per (a, b) pair on per-dataset scores, Holm over pairs."""
    rows = []
    for a, b in pairs:
        diff = scores[a] - scores[b]
        p = stats.wilcoxon(scores[a], scores[b], zero_method="zsplit").pvalue if (diff != 0).any() else 1.0
        rows.append(dict(a=a, b=b, mean_diff=diff.mean(), median_diff=diff.median(),
                         wins=int((diff > 0).sum()), losses=int((diff < 0).sum()), p=p))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].values) if len(out) else []
    return out


# ---------------------------------------------------------------- within a dataset
def corrected_ttest(a, b, folds: int) -> tuple[float, float]:
    """Nadeau-Bengio corrected resampled t-test on paired k-fold scores.
    Variance is inflated by (1/k + n_test/n_train) = (1/k + 1/(k-1))."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    k = len(d)
    if k < 2 or np.allclose(d, d[0]):
        return (float(d.mean()) if k else float("nan")), 1.0
    var = d.var(ddof=1) * (1.0 / k + 1.0 / (folds - 1))
    t = d.mean() / np.sqrt(var)
    return float(d.mean()), float(2 * stats.t.sf(abs(t), df=k - 1))


def per_dataset_tests(results, pairs, alpha=0.05) -> pd.DataFrame:
    """Corrected t-test for each (a, b) pair on every dataset, with BH-FDR
    across datasets within each pair. `results` is a ResultSet."""
    rows = []
    for ds in results.scores.index:
        pf, folds = results.per_fold[ds], int(results.info.loc[ds, "folds"])
        for a, b in pairs:
            diff, p = corrected_ttest(pf[a], pf[b], folds)
            rows.append(dict(dataset=ds, pair=f"{a} vs {b}", a=a, b=b, diff=diff, p=p))
    t = pd.DataFrame(rows)
    t["q"] = t.groupby("pair")["p"].transform(benjamini_hochberg)
    for col, src in (("verdict", "p"), ("verdict_fdr", "q")):
        t[col] = np.where(t[src] >= alpha, "tie", np.where(t["diff"] > 0, "win", "loss"))
    return t
