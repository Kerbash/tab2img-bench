"""Fast tests (CPU, a few seconds):  pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tab2img as t2i
from tab2img.analysis import (compare_to_best, corrected_ttest, critical_difference, friedman, holm,
                              load_results)

ROOT = Path(__file__).resolve().parents[1]
LOSSLESS = {"SOM2IMG", "SOM2IMG_tuned", "IGTD", "CorrelationBased", "SnakeRandom", "NCTD"}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.random((80, 12))
    X[:, 1] = X[:, 0] + 0.05 * rng.random(80)          # a correlated pair
    y = (X[:, 0] + X[:, 2] > 1).astype(int)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]), y


@pytest.mark.parametrize("name", t2i.available())
def test_every_method_fits_and_transforms(name, data):
    X, y = data
    m = t2i.create(name, n_iterations=200) if name.startswith("SOM2IMG") else t2i.create(name)
    imgs = m.fit(X, y).transform(X.iloc[:5])
    assert imgs.shape == (5, *m.image_shape_)
    assert np.isfinite(imgs).all()
    assert imgs.min() >= 0 and imgs.max() <= 1 + 1e-9      # training rows stay in [0, 1]
    if name in LOSSLESS:
        assert m.n_features_placed_ == X.shape[1]


def test_layout_is_deterministic(data):
    X, y = data
    a = t2i.create("SOM2IMG", n_iterations=300).fit(X).feature_map_
    b = t2i.create("SOM2IMG", n_iterations=300).fit(X).feature_map_
    assert np.array_equal(a, b)


def test_grid_matches_original_rule():
    assert t2i.optimal_grid(30) == (5, 6)
    assert t2i.optimal_grid(20) == (4, 5)
    for d in range(1, 300):
        r, c = t2i.optimal_grid(d)
        assert r * c >= d


def test_encode_images_wrap_and_clip():
    x = np.array([[[1.02]]])
    assert t2i.encode_images(x)[0, 0, 0, 0] == pytest.approx(4 / 255)      # paper behaviour
    assert t2i.encode_images(x, clip=True)[0, 0, 0, 0] == pytest.approx(1.0)


def test_classifier_runs_on_cpu(data):
    X, y = data
    clf = t2i.T2IClassifier("CorrelationBased", epochs=2, device="cpu").fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(1), 1)


def test_evaluate_small(data, tmp_path):
    X, y = data
    ds = t2i.Dataset("toy", X, y)
    res = t2i.evaluate(ds, methods=["SnakeRandom", {"method": "SOM2IMG", "params": {"n_iterations": 100},
                                                    "label": "SOM_fast"}],
                       baselines=["LogReg_L2"], protocol=t2i.Protocol(folds=3, epochs=1),
                       out_json=tmp_path / "toy.json", verbose=False)
    assert {r["method"] for r in res["results"]} == {"SnakeRandom", "SOM_fast", "LogReg_L2"}
    assert all(len(r["f1_per_fold"]) == 3 for r in res["results"])
    rs = load_results(tmp_path)
    assert rs.baseline_methods() == ["LogReg_L2"]


def test_critical_difference_matches_demsar_table():
    se = np.sqrt(10 * 11 / (6.0 * 51))
    assert critical_difference(10, 51) == pytest.approx(3.164 * se, abs=1e-3)
    assert critical_difference(10, 51, test="bonferroni-dunn") == pytest.approx(2.773 * se, abs=1e-3)


def test_holm_and_corrected_ttest():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    diff, p = corrected_ttest([0.9] * 10, [0.9] * 10, 10)
    assert diff == 0 and p == 1.0


@pytest.mark.skipif(not (ROOT / "results" / "paper_51").exists(), reason="bundled results missing")
def test_reproduces_paper_ranking():
    rs = load_results(ROOT / "results" / "paper_51")
    t2i_only = rs.select(methods=rs.t2i_methods())
    chi2, p = friedman(t2i_only.scores)
    assert t2i_only.scores.shape == (51, 9)
    assert chi2 == pytest.approx(70.67, abs=0.01)
    tab = compare_to_best(t2i_only.scores)
    assert tab.index[0] == "CorrelationBased"
    assert tab.loc["SOM2IMG", "avg_rank"] == pytest.approx(3.539, abs=1e-3)
    assert tab.loc["SOM2IMG", "verdict"] == "tie"
