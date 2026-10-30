"""Loading a tabular classification dataset from CSV."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#: columns dropped by name because they are identifiers, not features
ID_COLUMNS = {"subject", "id", "index", "idx", "sample", "sample_id"}


@dataclass
class Dataset:
    """A numeric feature matrix `X` (DataFrame), integer labels `y` (0..k-1)."""
    name: str
    X: pd.DataFrame
    y: np.ndarray
    classes: list = field(default_factory=list)
    source: str = ""

    @property
    def n(self) -> int:
        return len(self.X)

    @property
    def d(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y))

    def subsample(self, n: int, seed: int = 42) -> "Dataset":
        """Stratified random subset of `n` rows (used for learning curves)."""
        idx, _ = train_test_split(np.arange(self.n), train_size=n, stratify=self.y, random_state=seed)
        idx = np.sort(idx)
        return Dataset(f"{self.name}_n{n}_s{seed}", self.X.iloc[idx].reset_index(drop=True),
                       self.y[idx], self.classes, self.source)


def from_frame(df: pd.DataFrame, target: str, name: str = "dataset", source: str = "",
               verbose: bool = True) -> Dataset:
    """Prepare a DataFrame exactly as the benchmark does:

    * drop identifier columns by name ("Unnamed: *", id, index, subject, ...)
    * cast boolean (one-hot) columns to int, then keep numeric columns only
    * label-encode the target
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in {list(df.columns)[:10]}...")
    id_like = [c for c in df.columns if c != target and (
        str(c).lower().startswith("unnamed") or str(c).strip().lower() in ID_COLUMNS)]
    X = df.drop(columns=[target] + id_like)
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        X[bool_cols] = X[bool_cols].astype(int)
    kept = X.select_dtypes(include=[np.number]).columns
    dropped = [c for c in X.columns if c not in kept]
    X = X[kept]
    if verbose and (id_like or dropped):
        print(f"  [{name}] dropped id-like columns {id_like}; non-numeric columns {dropped}")
    enc = LabelEncoder()
    y = enc.fit_transform(df[target])
    return Dataset(name, X, y, list(enc.classes_), source)


def load_csv(path, target: str = "target", name: str | None = None, verbose: bool = True) -> Dataset:
    path = Path(path)
    return from_frame(pd.read_csv(path), target, name or path.stem, str(path), verbose)
