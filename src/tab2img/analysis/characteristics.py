"""Describe a dataset by the properties used in the regime analysis."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tab2img.data import Dataset


def dataset_characteristics(ds: Dataset) -> dict:
    """N, d, classes, d/N, imbalance, correlation structure.

    imb            largest class size / (N / classes)   (1 = balanced)
    mean_abs_corr  mean |Pearson r| over feature pairs
    frac_strong    fraction of feature pairs with |r| > 0.5
    n90            principal components (standardised data) for 90 % variance
    redundancy     1 - n90 / d
    """
    X = ds.X.values.astype(float)
    n, d = X.shape
    counts = np.bincount(ds.y)
    counts = counts[counts > 0]
    if d > 1:
        c = np.nan_to_num(np.corrcoef(X, rowvar=False))
        iu = np.triu_indices(d, 1)
        absr = np.abs(c[iu])
        mac, strong = float(absr.mean()), float((absr > 0.5).mean())
    else:
        mac = strong = 0.0
    ev = PCA().fit(StandardScaler().fit_transform(X)).explained_variance_ratio_
    n90 = int(np.searchsorted(np.cumsum(ev), 0.9) + 1)
    return dict(dataset=ds.name, N=n, d=d, classes=len(counts), dN=d / n,
                imb=float(counts.max() / (n / len(counts))), mean_abs_corr=mac,
                frac_strong=strong, n90=n90, redundancy=1 - n90 / d)


def characteristics_table(datasets) -> pd.DataFrame:
    """Characteristics for an iterable of Datasets (or callables returning one)."""
    return pd.DataFrame([dataset_characteristics(ds() if callable(ds) else ds)
                         for ds in datasets]).set_index("dataset")
