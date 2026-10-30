"""REFINED-style layout: polar coordinates from feature correlations."""
import numpy as np

from tab2img.methods.base import FeatureMapMethod
from tab2img.methods.deepinsight import rasterise


class REFINED(FeatureMapMethod):
    """Place each feature by its correlation with an anchor feature (radius)
    and with a reference feature (angle), then rasterise. Lossy on collisions.

    Anchor = feature with the highest mean |r|; reference = feature least
    correlated with the anchor.
    """

    name = "REFINED"

    def _layout(self, X, y):
        c = np.corrcoef(X, rowvar=False)
        c = (c + c.T) / 2
        np.fill_diagonal(c, 0)
        c = np.nan_to_num(c)
        a = int(np.argmax(np.mean(np.abs(c), axis=1)))
        ref = int(np.argmin(c[a, :]))

        d = X.shape[1]
        r = np.zeros(d)
        theta = np.zeros(d)
        for k in range(d):
            r[k] = 1 - c[a, k]
            c_ik, c_jk, c_ij = c[a, k], c[ref, k], c[a, ref]
            den = np.sqrt((1 - c_ik ** 2) * (1 - c_ij ** 2))
            if den > 1e-8:
                theta[k] = np.arccos(np.clip((c_jk - c_ik * c_ij) / den, -1.0, 1.0))
        xy = np.array([r * np.cos(theta), r * np.sin(theta)]).T
        return rasterise(xy, self.image_shape_)
