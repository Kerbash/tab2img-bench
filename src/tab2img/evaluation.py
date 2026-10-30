"""The benchmark protocol: stratified k-fold CV over T2I methods and baselines.

For each fold:
  1. fit a MinMaxScaler on the training fold only, apply it to both folds
  2. for every T2I method: fit the layout on the training fold, build images,
     train the CNN, score the held-out fold
  3. for every tabular baseline: train and score on the same scaled folds

All methods share the same folds, so per-fold scores are paired (needed by
the corrected t-test in `tab2img.analysis`). One JSON file per dataset.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from tab2img.data import Dataset
from tab2img.grid import optimal_grid
from tab2img.methods import PAPER_METHODS
from tab2img.models.tabular import BASELINES, make_baseline
from tab2img.pipeline import T2IClassifier, default_device


@dataclass
class Protocol:
    """Evaluation settings. Defaults = the 51-dataset study."""
    folds: int = 10
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32
    cnn: str = "simple"
    quantize: bool = True
    clip: bool = False            # True fixes the uint8 wrap-around (see pipeline.encode_images)
    min_folds: int = 2            # folds are reduced to the smallest class size, never below this


@dataclass
class MethodSpec:
    """A T2I method to evaluate: registry name + optional parameters + label."""
    method: str
    params: dict = field(default_factory=dict)
    label: str | None = None

    @property
    def name(self):
        return self.label or self.method


def _as_specs(methods):
    out = []
    for m in methods:
        if isinstance(m, MethodSpec):
            out.append(m)
        elif isinstance(m, str):
            out.append(MethodSpec(m))
        elif isinstance(m, dict):
            out.append(MethodSpec(**m))
        else:
            raise TypeError(f"cannot interpret method spec {m!r}")
    return out


def _scores(y_true, pred, proba, k):
    f1 = f1_score(y_true, pred, average="weighted")
    acc = accuracy_score(y_true, pred)
    try:
        auc = roc_auc_score(y_true, proba[:, 1]) if k == 2 else \
            roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")
    return float(f1), float(acc), float(auc)


def evaluate(dataset: Dataset, methods=PAPER_METHODS, baselines=BASELINES,
             protocol: Protocol | None = None, out_json=None, verbose=True) -> dict:
    """Run the protocol on one dataset. Returns (and optionally writes) a dict
    with per-method mean/std and per-fold scores."""
    p = protocol or Protocol()
    specs = _as_specs(methods)
    baselines = list(baselines or [])
    X, y = dataset.X, dataset.y
    d, k = X.shape[1], int(len(np.unique(y)))
    counts = np.bincount(y)
    folds = p.folds
    if counts[counts > 0].min() < folds:
        folds = max(p.min_folds, min(folds, int(counts[counts > 0].min())))
        if verbose:
            print(f"  [note] smallest class has {counts[counts > 0].min()} rows; using {folds} folds")
    grid = optimal_grid(d)
    device = default_device()
    # inner 3-fold CV for LogReg_L1_CV only if every class keeps >= 3 training rows
    tune_l1 = counts[counts > 0].min() * (folds - 1) / folds >= 3
    if verbose:
        print(f"{dataset.name}: N={len(X)} d={d} classes={k} grid={grid[0]}x{grid[1]} "
              f"folds={folds} device={device}", flush=True)

    names = [s.name for s in specs] + baselines
    scores = {n: {"f1": [], "acc": [], "auc": []} for n in names}
    placed = {s.name: [] for s in specs}
    t0 = time.time()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=p.seed)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        sc = MinMaxScaler().fit(X.iloc[tr])
        Xtr = pd.DataFrame(sc.transform(X.iloc[tr]), columns=X.columns)
        Xte = pd.DataFrame(sc.transform(X.iloc[te]), columns=X.columns)
        ytr, yte = y[tr], y[te]

        for s in specs:
            try:
                clf = T2IClassifier(s.method, method_params=s.params, image_shape=grid, cnn=p.cnn,
                                    epochs=p.epochs, batch_size=p.batch_size, seed=p.seed,
                                    quantize=p.quantize, clip=p.clip).fit(Xtr, ytr)
                placed[s.name].append(clf.transformer_.n_features_placed_)
                proba = clf.predict_proba(Xte)
                pred = (proba[:, 1] > 0.5).astype(int) if k == 2 else proba.argmax(1)
                clf.release()
                f1, acc, auc = _scores(yte, pred, proba, k)
            except Exception as e:
                print(f"  [WARN] {s.name} fold {fold + 1}: {e}")
                traceback.print_exc()
                f1 = acc = auc = float("nan")
            for key, v in zip(("f1", "acc", "auc"), (f1, acc, auc)):
                scores[s.name][key].append(v)

        for b in baselines:
            try:
                clf = make_baseline(b, tune_l1, seed=p.seed).fit(Xtr.values, ytr)
                proba = clf.predict_proba(Xte.values)
                f1, acc, auc = _scores(yte, clf.predict(Xte.values), proba, k)
            except Exception as e:
                print(f"  [WARN] {b} fold {fold + 1}: {e}")
                f1 = acc = auc = float("nan")
            for key, v in zip(("f1", "acc", "auc"), (f1, acc, auc)):
                scores[b][key].append(v)
        if verbose:
            print(f"  fold {fold + 1}/{folds} done ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for n in names:
        f1a = np.array(scores[n]["f1"], float)
        rows.append({
            "method": n,
            "kind": "baseline" if n in baselines else "t2i",
            "f1_mean": float(np.nanmean(f1a)),
            "f1_std": float(np.nanstd(f1a)),
            "acc_mean": float(np.nanmean(scores[n]["acc"])),
            "auc_mean": float(np.nanmean(scores[n]["auc"])),
            "features_placed_mean": float(np.mean(placed[n])) if placed.get(n) else None,
            "features_total": int(d),
            "f1_per_fold": [None if np.isnan(v) else round(float(v), 4) for v in f1a],
        })
    rows.sort(key=lambda r: -np.nan_to_num(r["f1_mean"], nan=-1))
    out = {"dataset": dataset.name, "source": dataset.source, "n": int(len(X)), "d": int(d),
           "classes": k, "folds": folds, "grid": [int(grid[0]), int(grid[1])],
           "protocol": asdict(p), "device": str(device),
           "runtime_sec": round(time.time() - t0, 1), "results": rows}
    if verbose:
        print(format_summary(out))
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(out, f, indent=1)
    return out


def format_summary(res: dict) -> str:
    lines = [f"{'method':18s} {'F1':>7s} {'std':>6s} {'acc':>6s} {'auc':>6s}  features"]
    for r in res["results"]:
        feat = f"{r['features_placed_mean']:.1f}/{r['features_total']}" if r["features_placed_mean"] else "all"
        lines.append(f"{r['method']:18s} {r['f1_mean']:7.4f} {r['f1_std']:6.4f} "
                     f"{r['acc_mean']:6.3f} {r['auc_mean']:6.3f}  {feat}")
    return "\n".join(lines)


def run_benchmark(datasets, out_dir, methods=PAPER_METHODS, baselines=BASELINES,
                  protocol: Protocol | None = None, exclude: dict | None = None,
                  overwrite=False, verbose=True) -> list:
    """Evaluate many datasets; one `<out_dir>/<name>.json` each.

    datasets : iterable of `Dataset` objects, or callables returning one
               (so large files are only loaded when needed)
    exclude  : {dataset_name: [method names to skip]} e.g. NCTD on high-d sets
    Datasets whose JSON already exists are skipped unless overwrite=True, so
    an interrupted run can simply be restarted.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for ds in datasets:
        ds = ds() if callable(ds) else ds
        path = out_dir / f"{ds.name}.json"
        if path.exists() and not overwrite:
            if verbose:
                print(f"skip {ds.name} (already in {out_dir})")
            done.append(path)
            continue
        skip = set((exclude or {}).get(ds.name, []))
        ms = [m for m in _as_specs(methods) if m.name not in skip]
        evaluate(ds, ms, [b for b in baselines if b not in skip], protocol, path, verbose)
        done.append(path)
    return done


def learning_curve(dataset: Dataset, sizes, out_dir, seeds_small=(42, 43, 44), seeds_large=(42,),
                   small_cutoff=500, methods=("SOM2IMG", "CorrelationBased", "Random"),
                   baselines=(), protocol: Protocol | None = None, verbose=True) -> list:
    """Re-run the protocol on stratified subsamples of increasing size.
    Sizes <= `small_cutoff` are drawn with several seeds (noisier)."""
    subsets = []
    for n in sizes:
        for s in (seeds_small if n <= small_cutoff else seeds_large):
            subsets.append(lambda n=n, s=s: dataset.subsample(n, s))
    return run_benchmark(subsets, out_dir, list(methods), list(baselines), protocol, verbose=verbose)
