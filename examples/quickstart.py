"""Quick tour on scikit-learn's Breast Cancer data (no download needed).

    python examples/quickstart.py          (from the repository root, after `pip install -e .`)
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tab2img as t2i
from tab2img.analysis import grouping_test, render_layout

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target.values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
sc = MinMaxScaler().fit(Xtr)                       # scale on the training split only
Xtr = pd.DataFrame(sc.transform(Xtr), columns=X.columns)
Xte = pd.DataFrame(sc.transform(Xte), columns=X.columns)

# 1. A T2I method is an object: fit a layout, transform rows into images
print("methods:", t2i.available())
som = t2i.create("SOM2IMG")                        # or t2i.methods.SOM2IMG(sigma=1.0, ...)
som.fit(Xtr)
images = som.transform(Xte)
print("image batch:", images.shape, "features placed:", som.n_features_placed_, "/", X.shape[1])

# 2. Is the layout meaningful? Group features by what they measure
short = {"radius": "RAD", "perimeter": "PER", "area": "ARE", "compactness": "CMP", "concavity": "CCV",
         "concave points": "CPT", "texture": "TEX", "smoothness": "SMO", "symmetry": "SYM",
         "fractal dimension": "FRA"}
measure = [next(k for k in sorted(short, key=len, reverse=True) if k in c) for c in X.columns]
print(render_layout(som.feature_map_, [short[m] for m in measure]))
size_shape = {"radius": "size", "perimeter": "size", "area": "size",
              "compactness": "shape", "concavity": "shape", "concave points": "shape"}
print("same-group grid distance vs random:", grouping_test(som.feature_map_, [size_shape.get(m) for m in measure]))

# 3. The full pipeline (image + CNN) is a scikit-learn classifier
clf = t2i.T2IClassifier("SOM2IMG", epochs=30).fit(Xtr, ytr)
print("test accuracy:", (clf.predict(Xte) == yte).mean())

# 4. The benchmark protocol (10-fold CV, all methods share folds) on one dataset
ds = t2i.from_frame(pd.concat([X, data.target.rename("target")], axis=1), "target", name="BreastCancer")
res = t2i.evaluate(ds, methods=["SOM2IMG", "CorrelationBased", "SnakeRandom"],
                   baselines=["LogReg_L2"], protocol=t2i.Protocol(folds=5, epochs=10))
