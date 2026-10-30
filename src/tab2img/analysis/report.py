"""One-call text report covering the analyses in the paper."""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pandas as pd

from tab2img.analysis.regimes import cutoff_sensitivity, regime_table, regime_tests
from tab2img.analysis.stats import (compare_to_best, critical_difference, friedman,
                                    pairwise_wilcoxon, per_dataset_tests)

pd.set_option("display.width", 200)


def _rank_block(title, scores):
    chi2, p = friedman(scores)
    n, k = scores.shape
    print(f"\n=== {title}: {n} datasets, {k} methods ===")
    print(f"Friedman chi2 = {chi2:.2f}, p = {p:.2e}   Nemenyi CD = {critical_difference(k, n):.2f}")
    print(compare_to_best(scores).round(3).to_string())


def full_report(results, focus="SOM2IMG", characteristics=None, cutoff=1.0, path=None) -> str:
    """Overall ranking, regime analysis, per-dataset tests and T2I-vs-baselines.

    results : ResultSet (from load_results)
    focus   : method compared head-to-head against the others
    characteristics : optional table with a `dN` column (else uses results.info)
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        t2i = results.select(methods=results.t2i_methods())
        _rank_block("T2I methods, all datasets", t2i.scores)

        if len(t2i.scores) >= 4:
            print(f"\n=== regimes split at d/N = {cutoff} ===")
            for lab, ((chi2, p), tab) in regime_tests(t2i, cutoff=cutoff, characteristics=characteristics).items():
                print(f"\n-- {lab}: Friedman chi2={chi2:.2f} p={p:.2e}")
                print(tab[["avg_rank", "p_holm", "verdict"]].round(3).to_string())
            print("\n-- cross-regime consistency (lower worst_regime = more versatile)")
            print(regime_table(t2i, cutoff=cutoff, characteristics=characteristics).round(2).to_string())
            print("\n-- sensitivity to the cutoff")
            print(cutoff_sensitivity(t2i, characteristics=characteristics).to_string())

        if focus in t2i.methods:
            others = [m for m in t2i.methods if m != focus]
            print(f"\n=== {focus} vs each T2I method: Wilcoxon over datasets ===")
            print(pairwise_wilcoxon(t2i.scores, [(focus, m) for m in others]).round(4).to_string(index=False))
            t = per_dataset_tests(t2i, [(focus, m) for m in others])
            print(f"\n=== {focus} vs X per dataset (corrected t-test p<.05 | BH-FDR) ===")
            for pair, g in t.groupby("pair", sort=False):
                c, cq = g.verdict.value_counts(), g.verdict_fdr.value_counts()
                print(f"  {pair:34s} win {c.get('win', 0):2d} tie {c.get('tie', 0):2d} loss {c.get('loss', 0):2d}"
                      f"   | FDR win {cq.get('win', 0):2d} loss {cq.get('loss', 0):2d}")

        base = results.baseline_methods()
        if base:
            joint = results.select(methods=t2i.methods + base)
            _rank_block("T2I methods + tabular baselines (same folds)", joint.scores)
            if focus in joint.methods:
                print(f"\n=== {focus} vs each baseline: Wilcoxon over datasets ===")
                print(pairwise_wilcoxon(joint.scores, [(focus, b) for b in base]).round(4).to_string(index=False))
    text = buf.getvalue()
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
