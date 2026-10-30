"""SOM2IMG: lay features out on the grid of a Self-Organizing Map.

Each feature is a point in sample space (its column of training values). A SOM
whose neurons form the image grid is trained on those points, so features with
similar profiles land on nearby neurons. Each feature is drawn at its best
matching unit (BMU); if that pixel is taken it spills over to the nearest free
pixel (Manhattan distance), so no feature is ever dropped.
"""
import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA

from tab2img.grid import optimal_grid
from tab2img.methods.base import FeatureMapMethod


class SOM2IMG(FeatureMapMethod):
    """Self-Organizing-Map feature layout.

    Parameters
    ----------
    image_shape : (rows, cols) or None
        Grid size; None = smallest near-square grid holding all features.
    sigma, learning_rate, n_iterations :
        MiniSom training settings (paper: 1.0, 0.5, 5000).
    grid_mult : float
        Use a grid of about `grid_mult * d` cells instead of `image_shape`
        (fewer BMU collisions). The paper's "SOM2IMG_tuned" is grid_mult=3.
    assignment : "spillover" | "hungarian"
        How features are put on neurons. "spillover" (paper) = BMU, else the
        nearest free cell; "hungarian" = optimal one-to-one assignment.
    init : "random" | "pca"         SOM weight initialisation.
    topology : "rectangular" | "hexagonal"
    whiten : bool
        PCA-whiten the feature vectors (to <= 64 dims) before training.
    random_state : int
    """

    name = "SOM2IMG"

    def __init__(self, image_shape=None, sigma=1.0, learning_rate=0.5, n_iterations=5000,
                 grid_mult=1.0, assignment="spillover", init="random",
                 topology="rectangular", whiten=False, random_state=42):
        super().__init__(image_shape, random_state)
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.grid_mult = grid_mult
        self.assignment = assignment
        self.init = init
        self.topology = topology
        self.whiten = whiten

    def _layout(self, X, y):
        feats = self._scaled(X).T                      # (d, n): one row per feature
        d, n = feats.shape
        if self.whiten and n > 3 and d > 3:
            feats = PCA(n_components=min(n, d, 64), whiten=True,
                        random_state=self.random_state).fit_transform(feats)

        if self.grid_mult != 1.0:
            rows, cols = optimal_grid(max(d, int(round(self.grid_mult * d))))
        else:
            rows, cols = self.image_shape_

        som = MiniSom(x=rows, y=cols, input_len=feats.shape[1],
                      sigma=min(self.sigma, rows, cols), learning_rate=self.learning_rate,
                      random_seed=self.random_state, topology=self.topology)
        if self.init == "pca" and feats.shape[1] >= 2:
            try:
                som.pca_weights_init(feats)
            except Exception:   # MiniSom refuses PCA init in some degenerate cases
                pass
        som.train_random(feats, self.n_iterations)
        self.som_ = som

        fm = np.full((rows, cols), -1, dtype=int)
        if self.assignment == "hungarian":
            from scipy.optimize import linear_sum_assignment
            from scipy.spatial.distance import cdist
            weights = som.get_weights().reshape(rows * cols, -1)
            r, c = linear_sum_assignment(cdist(feats, weights))
            for i, cell in zip(r, c):
                fm[cell // cols, cell % cols] = int(i)
        elif self.assignment == "spillover":
            for i, v in enumerate(feats):
                bx, by = som.winner(v)
                if fm[bx, by] == -1:
                    fm[bx, by] = i
                    continue
                free = np.argwhere(fm == -1)
                if free.size == 0:                     # grid smaller than d: overwrite
                    fm[bx, by] = i
                    continue
                dist = np.abs(free[:, 0] - bx) + np.abs(free[:, 1] - by)
                fx, fy = free[int(np.argmin(dist))]
                fm[fx, fy] = i
        else:
            raise ValueError(f"unknown assignment {self.assignment!r}")
        return fm
