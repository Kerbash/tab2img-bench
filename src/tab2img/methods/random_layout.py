"""Control layouts with no learned structure."""
import numpy as np

from tab2img.methods.base import FeatureMapMethod, paint, snake_fill


class RandomPixels(FeatureMapMethod):
    """Every pixel shows a feature drawn uniformly at random (with replacement),
    so some features repeat and some are missing ("Random" in the paper).

    Unlike the other methods, pixel values are min-max scaled *within each
    sample* (across that row's features), as in the original implementation.
    """

    name = "Random"

    def _layout(self, X, y):
        rows, cols = self.image_shape_
        return np.random.RandomState(self.random_state).randint(0, X.shape[1], size=(rows, cols))

    def _transform(self, X):
        lo = X.min(axis=1, keepdims=True)
        rng = X.max(axis=1, keepdims=True) - lo
        rng[rng == 0] = 1.0
        return paint(self.feature_map_, (X - lo) / rng)


class SnakeRandom(FeatureMapMethod):
    """Lossless random control: every feature placed exactly once, in a random
    order, filled row by row in a snake pattern. Separates "does the order
    matter?" from "is every feature kept?"."""

    name = "SnakeRandom"

    def _layout(self, X, y):
        order = np.random.RandomState(self.random_state).permutation(X.shape[1])
        return snake_fill(order, self.image_shape_)
