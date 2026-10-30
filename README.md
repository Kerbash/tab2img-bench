# tab2img

Tabular-to-image (T2I) methods, a CNN classification pipeline, and the
benchmark + statistical analysis used in the SOM2IMG study
(*Which Tabular-to-Image Method, for Which Data? A Regime Map from 51 Datasets*).

* **Methods are objects.** `SOM2IMG` and the eight comparison methods share one
  scikit-learn style interface: `fit(X, y)` learns a feature-to-pixel layout,
  `transform(X)` returns a batch of images.
* **The pipeline is a classifier.** `T2IClassifier` = T2I method + CNN, usable
  like any scikit-learn estimator.
* **The benchmark is one function or one command.** Stratified k-fold CV in
  which every T2I method and every tabular baseline sees the same folds.
* **The analysis reproduces the paper** (Friedman / Nemenyi / Holm,
  per-dataset corrected t-tests, d/N regimes, layout semantics). The study's
  raw results are bundled in `results/paper_51/`.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt  (exact paper versions)
```

Python 3.10+. A CUDA GPU is used automatically when available (install the
CUDA build of PyTorch from pytorch.org first). `pip install -e .[resnet]` adds
torchvision for the pretrained ResNet-18 head.

## Quick start

```python
import tab2img as t2i

# 1) a T2I method on its own
som = t2i.create("SOM2IMG")             # names: t2i.available()
som.fit(X_train)                        # X: DataFrame or array, scaled to [0, 1]
images = som.transform(X_test)          # (n, rows, cols), values in [0, 1]
som.feature_map_                        # (rows, cols) feature index per pixel, -1 = empty

# methods are plain classes too, with their parameters in the constructor
from tab2img.methods import SOM2IMG
som = SOM2IMG(image_shape=(8, 8), sigma=1.0, n_iterations=5000, assignment="hungarian")

# 2) the image + CNN pipeline as a classifier
clf = t2i.T2IClassifier("SOM2IMG", cnn="simple", epochs=30).fit(X_train, y_train)
clf.predict(X_test); clf.predict_proba(X_test)

# 3) the benchmark protocol on one dataset
ds = t2i.load_csv("data/my_data.csv", target="label")
res = t2i.evaluate(ds, methods=t2i.PAPER_METHODS, baselines=["LogReg_L2", "HistGB"],
                   out_json="results/my_data.json")
```

`examples/quickstart.py` runs all of this on scikit-learn's Breast Cancer data.

## Methods

| name | layout idea | keeps every feature | uses labels |
|---|---|---|---|
| `SOM2IMG` | Self-Organizing Map trained on the feature vectors; feature at its best-matching neuron, spill-over to the nearest free pixel | yes | no |
| `SOM2IMG_tuned` | `SOM2IMG(grid_mult=3.0)`: grid with ~3d cells | yes | no |
| `CorrelationBased` | average-linkage clustering on 1 − abs(Pearson r), snake fill | yes | no |
| `IGTD` | clustering on Jensen-Shannon distance between feature histograms, snake fill | yes | no |
| `DeepInsight` | t-SNE of the features, rasterised (collisions overwrite) | no | no |
| `REFINED` | polar coordinates from correlations with an anchor and a reference feature | no | no |
| `SuperTML` | top-k ANOVA features, t-SNE of samples, best feature per occupied cell | no | yes |
| `Random` | each pixel a random feature (with replacement), per-row scaling | no | no |
| `SnakeRandom` | every feature once, random order, snake fill (control: lossless, no structure) | yes | no |
| `NCTD` | circulant 2d x 2d image (not in the 51-dataset study: too large at high d) | yes | no |

`n_features_placed_` reports how many features a fitted layout actually shows.
The comparison methods are simplified re-implementations of each method's core
idea (e.g. `IGTD` uses clustering, not the original iterative pixel swapping);
they are not the authors' reference code.

`SOM2IMG` options: `sigma`, `learning_rate`, `n_iterations`, `grid_mult`,
`assignment` (`"spillover"` | `"hungarian"`), `init` (`"random"` | `"pca"`),
`topology` (`"rectangular"` | `"hexagonal"`), `whiten`.

### Adding a method

```python
from tab2img.methods import FeatureMapMethod, register, snake_fill

class VarianceOrder(FeatureMapMethod):
    name = "VarianceOrder"
    def _layout(self, X, y):                 # X: training rows (float ndarray)
        return snake_fill(X.var(axis=0).argsort()[::-1], self.image_shape_)

register("VarianceOrder", VarianceOrder)     # now usable in configs and evaluate()
```

Subclass `TabularToImage` (implement `_fit` and `_transform`) for methods that
are not a fixed pixel layout.

## The benchmark protocol

Per dataset (`tab2img.evaluate`):

1. Numeric features only; identifier columns (`Unnamed: *`, `id`, `index`,
   `subject`, …) are dropped by name; boolean one-hot columns are cast to int.
2. `StratifiedKFold(10, shuffle=True, random_state=42)`; folds are reduced to
   the smallest class size when it is below 10.
3. In every fold a `MinMaxScaler` is fit on the training fold only. Each T2I
   layout is also fit on the training fold only.
4. Images: grid = smallest near-square rectangle with ≥ d cells; pixels stored
   as 8-bit, replicated to 3 channels.
5. CNN (`simple`): conv(16) → conv(32) → adaptive 4×4 average pool → dense 128
   → dropout 0.3 → output; Adam, batch 32, 30 epochs, seed 42. Other heads:
   `deep`, `resnet18`, `linear`, `tiny_gap`, `tiny_flat`.
6. Baselines on the same scaled folds: `LogReg_L2`, `LogReg_L1_CV` (C tuned
   by inner 3-fold CV on the training fold), `RandomForest` (500 trees), `HistGB`.
7. Weighted F1 (primary), accuracy and AUC per fold.

Output: one JSON per dataset with mean/std and per-fold scores per method.

## Command line

```bash
python -m tab2img methods
python -m tab2img run configs/example.yaml              # skips datasets already done
python -m tab2img characteristics configs/example.yaml -o chars.csv
python -m tab2img analyze results/example --report report.txt --cd cd.png
```

See `configs/example.yaml` for the config format (methods with parameters and
labels, baselines, per-dataset exclusions).

## Analysis

```python
from tab2img.analysis import load_results, compare_to_best, regime_table, per_dataset_tests, full_report

rs = load_results("results/paper_51")
t2i_only = rs.select(methods=rs.t2i_methods())
compare_to_best(t2i_only.scores)           # avg rank, Holm p vs the best method
regime_table(t2i_only, cutoff=1.0)         # ranks for d/N >= 1 and < 1, worst regime, swing
per_dataset_tests(t2i_only, [("SOM2IMG", "SnakeRandom")])   # Nadeau-Bengio t-test + BH-FDR
print(full_report(rs))
```

Also: `dataset_characteristics` (N, d, d/N, imbalance, correlation, redundancy),
`rank_correlations` (method rank vs characteristic), `cutoff_sensitivity`,
`grouping_test` (do semantically related features sit together on the grid?),
and plots in `tab2img.analysis.plots` (CD diagram, layout picture, learning curves).

In `full_report`, the "T2I + baselines" ranking pools all T2I methods with the
baselines, so its average ranks differ from a pool of only the top-3 T2I methods.

## Reproducing the paper

**Statistics only (no GPU):** the original per-fold results of all 51 datasets
are in `results/paper_51/` (T2I and baseline rows merged per dataset, plus
`_dataset_characteristics.csv`).

```bash
python -m tab2img analyze results/paper_51 --cd cd.png
```

gives Friedman χ² = 70.67 (p = 3.6e-12), Nemenyi CD = 1.68, CorrelationBased
3.24 and SOM2IMG 3.54 average rank (Holm p = 0.58, tie), and the d/N regime
tables. `tests/test_tab2img.py::test_reproduces_paper_ranking` checks this.

**Full re-run:** `configs/paper_51.yaml` lists the 51 datasets, targets and the
expected file layout under `data/`. The datasets are not redistributed here;
see the paper for their sources.

```bash
python -m tab2img run configs/paper_51.yaml
```

### Reproducibility notes

* With the same library versions, layouts and images are bit-identical to the
  code that produced the paper results, and on CPU the CV scores are identical
  fold for fold (checked on Sonar: 5 methods x 10 folds). CNN training on a GPU
  is not deterministic (CUDA conv / pooling kernels): re-running the *original*
  code on Sonar moved SOM2IMG's mean F1 from 0.792 to 0.807, and single folds
  can change by a few points. Treat per-dataset T2I differences of about 0.01-0.02
  F1 as within run-to-run noise. Tabular baselines are deterministic.
  For exact reruns use CPU (`CUDA_VISIBLE_DEVICES=-1`).
* `clip=False` (default, as in the paper): a test-fold value outside the
  training range is cast to 8 bit without clipping and **wraps around**
  (1.02 → 260 → 4, a near-black pixel). This affects every method that scales
  values (all except `Random`). Set `protocol: {clip: true}` (or
  `T2IClassifier(clip=True)`) for new work.
* `Random` scales each row by its own min/max; `IGTD` quantises to 32 levels
  and expects inputs already in [0, 1]. Both follow the original implementations.

## Layout

```
src/tab2img/
  methods/      one file per T2I method + base classes and registry
  models/       cnn.py (PyTorch heads), tabular.py (baselines)
  pipeline.py   T2IClassifier, encode_images
  evaluation.py Protocol, evaluate, run_benchmark, learning_curve
  data.py       CSV loading and preparation
  grid.py       image size rule
  analysis/     results loading, stats, regimes, characteristics, layout test, plots, report
  cli.py        command line
configs/        example.yaml, paper_51.yaml
results/        paper_51/  (bundled study results)
examples/       quickstart.py
tests/          pytest suite (CPU, < 1 min)
```
