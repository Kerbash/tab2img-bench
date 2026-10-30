"""Load benchmark JSON files into tables.

Works on files written by `tab2img.evaluation` and on the original study's
`results_final/` + `results_baselines/` folders (pass both; rows for the same
dataset are merged).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ResultSet:
    scores: pd.DataFrame        # datasets x methods, mean score
    per_fold: dict              # {dataset: {method: [fold scores]}}
    info: pd.DataFrame          # per dataset: n, d, classes, folds, dN
    kinds: dict                 # {method: "t2i" | "baseline"}

    @property
    def methods(self):
        return list(self.scores.columns)

    def select(self, methods=None, datasets=None, complete=True) -> "ResultSet":
        """Subset; complete=True keeps only datasets where every chosen method has a score."""
        s = self.scores
        if methods is not None:
            s = s[[m for m in methods if m in s.columns]]
        if datasets is not None:
            s = s.loc[[d for d in datasets if d in s.index]]
        if complete:
            s = s.dropna()
        pf = {d: {m: self.per_fold[d][m] for m in s.columns if m in self.per_fold.get(d, {})} for d in s.index}
        return ResultSet(s, pf, self.info.loc[s.index], {m: self.kinds.get(m, "t2i") for m in s.columns})

    def t2i_methods(self):
        return [m for m in self.methods if self.kinds.get(m, "t2i") == "t2i"]

    def baseline_methods(self):
        return [m for m in self.methods if self.kinds.get(m) == "baseline"]


def load_results(*dirs, metric: str = "f1", exclude=("NCTD",)) -> ResultSet:
    """Read every `<Name>.json` in `dirs` (files starting with "_" are ignored).

    metric: "f1" | "acc" | "auc" (per-fold scores exist only for F1).
    exclude: methods to leave out (NCTD by default: missing on high-d sets).
    """
    from tab2img.models.tabular import BASELINES
    scores, per_fold, info, kinds = {}, {}, {}, {}
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.basename(path)[:-5]
            if name.startswith("_"):
                continue
            r = json.load(open(path))
            meta = info.setdefault(name, {})
            for key in ("n", "d", "classes", "folds"):
                if key in r and key not in meta:
                    meta[key] = r[key]
            for row in r["results"]:
                m = row["method"]
                if m in exclude:
                    continue
                kinds[m] = row.get("kind", "baseline" if m in BASELINES else "t2i")
                scores.setdefault(name, {})[m] = row.get(f"{metric}_mean", np.nan)
                per_fold.setdefault(name, {})[m] = [np.nan if v is None else v
                                                    for v in row.get("f1_per_fold", [])]
    S = pd.DataFrame(scores).T.sort_index()
    I = pd.DataFrame(info).T.loc[S.index]
    I["dN"] = I["d"].astype(float) / I["n"].astype(float)
    return ResultSet(S, per_fold, I, kinds)
