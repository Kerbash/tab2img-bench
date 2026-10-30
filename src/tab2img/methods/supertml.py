"""SuperTML-style supervised layout."""
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from tab2img.methods.base import FeatureMapMethod


class SuperTML(FeatureMapMethod):
    """Supervised: keep the top-k features by ANOVA F (k = number of pixels
    by default), embed the *samples* with t-SNE onto the grid, and give each
    occupied cell the selected feature with the highest mean among the
    samples landing there. Cells with no samples stay empty, so this layout
    usually drops many features.
    """

    name = "SuperTML"
    supervised = True

    def __init__(self, image_shape=None, n_features_to_select="auto", random_state=42):
        super().__init__(image_shape, random_state)
        self.n_features_to_select = n_features_to_select

    def _layout(self, X, y):
        rows, cols = self.image_shape_
        k = rows * cols if self.n_features_to_select == "auto" else self.n_features_to_select
        k = min(k, X.shape[1])
        sel = SelectKBest(f_classif, k=k).fit(X, y)
        idx = sel.get_support(indices=True)
        Xs = X[:, idx]

        tsne = TSNE(n_components=2, random_state=self.random_state, init="pca",
                    learning_rate="auto", perplexity=min(30, Xs.shape[0] - 1))
        # both axes are scaled to [0, rows-1], as in the original implementation
        pos = MinMaxScaler(feature_range=(0, rows - 1)).fit_transform(tsne.fit_transform(Xs)).astype(int)

        fm = np.full((rows, cols), -1, dtype=int)
        for i in range(rows):
            for j in range(cols):
                mask = (pos[:, 0] == i) & (pos[:, 1] == j)
                if mask.any():
                    fm[i, j] = idx[int(np.argmax(Xs[mask].mean(axis=0)))]
        return fm
