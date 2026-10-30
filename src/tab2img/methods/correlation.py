"""CorrelationBased: hierarchical clustering on feature correlations."""
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from tab2img.methods.base import FeatureMapMethod, snake_fill


class CorrelationBased(FeatureMapMethod):
    """Order features by average-linkage clustering on 1 - |Pearson r|, then
    snake-fill the grid in that order (lossless)."""

    name = "CorrelationBased"

    def _layout(self, X, y):
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num((corr + corr.T) / 2)
        dist = 1 - np.abs(corr)
        np.fill_diagonal(dist, 0)
        order = leaves_list(linkage(squareform(dist), method="average"))
        return snake_fill(order, self.image_shape_)
