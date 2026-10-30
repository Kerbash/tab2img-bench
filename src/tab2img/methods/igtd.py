"""IGTD-style layout: cluster features by the similarity of their value
distributions (Jensen-Shannon distance between histograms)."""
import numpy as np
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import jensenshannon, squareform

from tab2img.methods.base import FeatureMapMethod, paint, snake_fill


def _jsd_row(i, dists):
    return [jensenshannon(dists[i], dists[j]) for j in range(len(dists))]


class IGTD(FeatureMapMethod):
    """Histogram each feature into `n_bins` bins, cluster features (average
    linkage) on pairwise Jensen-Shannon distance, snake-fill in that order.

    Expects inputs already scaled to [0, 1] (the benchmark pipeline does this);
    pixel values are the inputs quantised to `n_bins` levels.
    """

    name = "IGTD"

    def __init__(self, image_shape=None, n_bins=32, n_jobs=-1, random_state=42):
        super().__init__(image_shape, random_state)
        self.n_bins = n_bins
        self.n_jobs = n_jobs

    def _layout(self, X, y):
        q = np.round(X * (self.n_bins - 1))
        dists = np.array([np.histogram(q[:, i], bins=self.n_bins, range=(0, self.n_bins), density=True)[0]
                          for i in range(X.shape[1])])
        jsd = np.array(Parallel(n_jobs=self.n_jobs)(delayed(_jsd_row)(i, dists) for i in range(len(dists))))
        jsd = np.nan_to_num((jsd + jsd.T) / 2)
        np.fill_diagonal(jsd, 0)
        order = leaves_list(linkage(squareform(jsd), method="average"))
        return snake_fill(order, self.image_shape_)

    def _transform(self, X):
        return paint(self.feature_map_, np.round(X * (self.n_bins - 1)) / (self.n_bins - 1.0))
