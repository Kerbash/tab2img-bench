"""Analysis of benchmark results: rankings, significance, regimes, layouts."""
from tab2img.analysis.characteristics import characteristics_table, dataset_characteristics
from tab2img.analysis.layout import grouping_test, render_layout
from tab2img.analysis.regimes import (cutoff_sensitivity, rank_correlations, regime_table,
                                      regime_tests, split_by)
from tab2img.analysis.report import full_report
from tab2img.analysis.results import ResultSet, load_results
from tab2img.analysis.stats import (average_ranks, benjamini_hochberg, compare_to_best,
                                    corrected_ttest, critical_difference, friedman, holm,
                                    pairwise_wilcoxon, per_dataset_tests)

__all__ = ["load_results", "ResultSet", "average_ranks", "friedman", "critical_difference",
           "compare_to_best", "pairwise_wilcoxon", "corrected_ttest", "per_dataset_tests",
           "holm", "benjamini_hochberg", "dataset_characteristics", "characteristics_table",
           "split_by", "regime_table", "regime_tests", "cutoff_sensitivity", "rank_correlations",
           "grouping_test", "render_layout", "full_report"]
