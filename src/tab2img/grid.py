"""Choosing the image grid size."""
import math


def optimal_grid(n_features: int) -> tuple[int, int]:
    """Near-square (rows, cols) grid with at least `n_features` cells.

    Same rule as the original benchmark: start from floor(sqrt(d)) rows and
    try +-2 rows around it, keeping the most square rectangle.
    """
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    s = int(math.sqrt(n_features))
    best = (s, math.ceil(n_features / s))
    for w in range(max(1, s - 2), s + 3):
        h = math.ceil(n_features / w)
        if abs(w - h) < abs(best[0] - best[1]):
            best = (w, h)
    return best
