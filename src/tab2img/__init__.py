"""tab2img: tabular-to-image methods (SOM2IMG and baselines), a CNN pipeline,
and the benchmark + statistical analysis used in the SOM2IMG study.

Quick start
-----------
    import tab2img as t2i

    som = t2i.create("SOM2IMG")                 # any name in t2i.available()
    images = som.fit(X_train).transform(X_test) # (n, rows, cols)

    clf = t2i.T2IClassifier("SOM2IMG").fit(X_train, y_train)
    clf.predict(X_test)

    ds = t2i.load_csv("data.csv", target="label")
    t2i.evaluate(ds, out_json="results/data.json")   # full CV protocol
"""
from tab2img.data import Dataset, from_frame, load_csv
from tab2img.evaluation import MethodSpec, Protocol, evaluate, learning_curve, run_benchmark
from tab2img.grid import optimal_grid
from tab2img.methods import PAPER_METHODS, TabularToImage, available, create, register
from tab2img.pipeline import T2IClassifier, encode_images

__version__ = "0.1.0"

__all__ = ["create", "available", "register", "PAPER_METHODS", "TabularToImage", "T2IClassifier",
           "encode_images", "Dataset", "load_csv", "from_frame", "Protocol", "MethodSpec",
           "evaluate", "run_benchmark", "learning_curve", "optimal_grid"]
