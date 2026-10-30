"""DeepInsight: t-SNE embedding of the features, rasterised onto the grid."""
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from tab2img.methods.base import FeatureMapMethod


def rasterise(coords: np.ndarray, shape) -> np.ndarray:
    """Min-max scale 2-D points onto a (rows, cols) grid and round to pixels.
    Features that round to the same pixel overwrite each other (later wins),
    so this layout can drop features."""
    rows, cols = shape
    pix = np.round(MinMaxScaler().fit_transform(coords) * np.array([rows - 1, cols - 1])).astype(int)
    fm = np.full((rows, cols), -1, dtype=int)
    for i, (r, c) in enumerate(pix):
        fm[r, c] = i
    return fm


class DeepInsight(FeatureMapMethod):
    """t-SNE (PCA init) on the feature vectors, then rasterise. Lossy:
    features that collide on a pixel are dropped (see `n_features_placed_`)."""

    name = "DeepInsight"

    def __init__(self, image_shape=None, perplexity=30, max_iter=1000, random_state=42):
        super().__init__(image_shape, random_state)
        self.perplexity = perplexity
        self.max_iter = max_iter

    def _layout(self, X, y):
        tsne = TSNE(n_components=2, random_state=self.random_state,
                    perplexity=min(self.perplexity, X.shape[1] - 1),
                    max_iter=self.max_iter, init="pca", learning_rate="auto")
        return rasterise(tsne.fit_transform(X.T), self.image_shape_)
