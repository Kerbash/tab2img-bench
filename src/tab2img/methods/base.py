"""Base classes for tabular-to-image (T2I) methods.

Every method is a scikit-learn style transformer:

    method = SOM2IMG(image_shape=(5, 6))
    method.fit(X_train, y_train)          # learn the feature -> pixel layout
    images = method.transform(X_test)     # (n_samples, H, W) float array

`X` is any 2-D array-like (NumPy array or pandas DataFrame) of shape
(n_samples, n_features). Images hold values on the method's own [0, 1] scale;
values from unseen data may fall slightly outside [0, 1] (see
`tab2img.pipeline.encode_images` for how they are turned into CNN input).

`image_shape` is (rows, cols). If it is None, the near-square grid with at
least one cell per feature is chosen at fit time (`tab2img.grid.optimal_grid`).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from tab2img.grid import optimal_grid


def as_2d_float(X) -> np.ndarray:
    """Convert a DataFrame / array / single row to a float64 (n, d) array."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array of shape (n_samples, n_features), got {arr.shape}")
    return arr


class TabularToImage(TransformerMixin, BaseEstimator):
    """Common interface for all T2I methods.

    Subclasses implement `_fit(X, y)` (X is a float64 ndarray) and
    `_transform(X) -> (n, H, W)` array. Fitted attributes end with `_`:

    image_shape_      (rows, cols) of the produced images
    n_features_in_    number of input features seen in fit
    """

    #: short registry name, set by subclasses
    name: str = "base"
    #: True if fit() needs the labels y
    supervised: bool = False

    def __init__(self, image_shape=None, random_state=42):
        self.image_shape = image_shape
        self.random_state = random_state

    # ---- public API -------------------------------------------------------
    def fit(self, X, y=None):
        X = as_2d_float(X)
        if self.supervised and y is None:
            raise ValueError(f"{type(self).__name__} is supervised and needs y")
        self.n_features_in_ = X.shape[1]
        self.image_shape_ = tuple(self.image_shape) if self.image_shape is not None \
            else optimal_grid(self.n_features_in_)
        self._fit(X, None if y is None else np.asarray(y))
        return self

    def transform(self, X) -> np.ndarray:
        self._check_fitted()
        X = as_2d_float(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        return self._transform(X)

    @property
    def n_features_placed_(self) -> int:
        """How many distinct input features appear somewhere in the image."""
        return self.n_features_in_

    # ---- for subclasses ---------------------------------------------------
    def _fit(self, X: np.ndarray, y):
        raise NotImplementedError

    def _transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _check_fitted(self):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() first")


class FeatureMapMethod(TabularToImage):
    """A method whose output is a fixed layout: each pixel shows one feature.

    Subclasses set `self.feature_map_`, an int array of shape `image_shape_`
    holding the feature index shown at each pixel, or -1 for an empty pixel.
    Pixel values come from a per-feature min-max scaler fit on the training
    data (as in the original SOM2IMG benchmark).
    """

    def _fit(self, X, y):
        self.value_scaler_ = MinMaxScaler().fit(X)
        self.feature_map_ = self._layout(X, y)
        if self.feature_map_.shape != tuple(self.image_shape_):
            # a method may choose its own grid size (e.g. expanded-grid SOM)
            self.image_shape_ = tuple(self.feature_map_.shape)

    def _layout(self, X, y) -> np.ndarray:
        raise NotImplementedError

    def _scaled(self, X):
        return self.value_scaler_.transform(X)

    def _transform(self, X):
        return paint(self.feature_map_, self._scaled(X))

    @property
    def n_features_placed_(self) -> int:
        fm = self.feature_map_
        return int(len(np.unique(fm[fm >= 0])))


def paint(feature_map: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fill each pixel with the value of the feature it shows (0 if empty).

    feature_map: (H, W) int, -1 = empty.  values: (n, d).  Returns (n, H, W).
    """
    idx = np.where(feature_map >= 0, feature_map, 0)
    out = values[:, idx]
    out[:, feature_map < 0] = 0.0
    return out


def snake_fill(order, shape) -> np.ndarray:
    """Place features row by row in `order`, reversing every other row
    (boustrophedon), so consecutive features are always neighbours."""
    rows, cols = shape
    order = list(order)
    if len(order) > rows * cols:
        raise ValueError(f"image {rows}x{cols}={rows * cols} is too small for {len(order)} features")
    fm = np.full((rows, cols), -1, dtype=int)
    k = 0
    for i in range(rows):
        js = range(cols) if i % 2 == 0 else reversed(range(cols))
        for j in js:
            if k < len(order):
                fm[i, j] = int(order[k])
                k += 1
    return fm
