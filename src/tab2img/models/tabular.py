"""Standard tabular classifiers used as reference baselines (CPU, scikit-learn)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

BASELINES = ["LogReg_L2", "LogReg_L1_CV", "RandomForest", "HistGB"]


def make_baseline(name: str, tune_l1: bool = True, inner_folds: int = 3, seed: int = 42, n_jobs: int = 1):
    """Build one baseline.

    LogReg_L1_CV tunes C by inner `inner_folds`-fold CV on the training fold
    only. Pass tune_l1=False when the training folds are too small for that
    (smallest class < inner_folds rows); it then uses C=1.0, as in the paper.
    """
    if name == "LogReg_L2":
        return LogisticRegression(max_iter=5000)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=n_jobs)
    if name == "HistGB":
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "LogReg_L1_CV":
        if not tune_l1:
            return LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)
        return LogisticRegressionCV(Cs=np.logspace(-2, 2, 9), cv=inner_folds, penalty="l1", solver="saga",
                                    scoring="f1_weighted", max_iter=5000, random_state=seed, n_jobs=n_jobs)
    raise KeyError(f"unknown baseline {name!r}; available: {BASELINES}")
