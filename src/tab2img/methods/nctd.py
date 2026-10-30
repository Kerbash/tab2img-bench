"""NCTD: circulant 2d x 2d image (Sci. Rep. 2025, doi:10.1038/s41598-025-01568-0)."""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tab2img.methods.base import TabularToImage


class NCTD(TabularToImage):
    """Row i of a d x d block is the scaled feature vector rotated left by i;
    the block is mirrored left/right and up/down into a 2d x 2d image.

    The image has 4*d^2 pixels, so it is impractical above a few hundred
    features (`image_shape` is ignored).
    """

    name = "NCTD"

    def _fit(self, X, y):
        d = X.shape[1]
        self.value_scaler_ = MinMaxScaler().fit(X)
        self.image_shape_ = (2 * d, 2 * d)
        self._roll_idx = (np.arange(d)[:, None] + np.arange(d)[None, :]) % d

    def _transform(self, X):
        v = self.value_scaler_.transform(X)
        block = v[:, self._roll_idx]                         # (n, d, d)
        top = np.concatenate([block, block[:, :, ::-1]], axis=2)
        return np.concatenate([top, top[:, ::-1, :]], axis=1)
