"""Tabular-to-image methods and a name -> constructor registry.

    from tab2img.methods import create, available
    som = create("SOM2IMG", image_shape=(5, 6))
    tuned = create("SOM2IMG_tuned")          # preset: SOM2IMG(grid_mult=3.0)

Add your own method by subclassing `FeatureMapMethod` (implement `_layout`)
or `TabularToImage` (implement `_fit` / `_transform`) and calling
`register("MyName", MyClass)`.
"""
from functools import partial

from tab2img.methods.base import FeatureMapMethod, TabularToImage, paint, snake_fill
from tab2img.methods.correlation import CorrelationBased
from tab2img.methods.deepinsight import DeepInsight
from tab2img.methods.igtd import IGTD
from tab2img.methods.nctd import NCTD
from tab2img.methods.random_layout import RandomPixels, SnakeRandom
from tab2img.methods.refined import REFINED
from tab2img.methods.som import SOM2IMG
from tab2img.methods.supertml import SuperTML

_REGISTRY = {}


def register(name: str, factory):
    """Register a class (or any callable returning a TabularToImage) under `name`."""
    _REGISTRY[name] = factory
    return factory


def create(name: str, **params) -> TabularToImage:
    """Build a method by registry name; keyword arguments override defaults."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; available: {available()}") from None
    return factory(**params)


def available() -> list:
    return list(_REGISTRY)


for _cls in (SOM2IMG, RandomPixels, SnakeRandom, IGTD, DeepInsight,
             CorrelationBased, REFINED, SuperTML, NCTD):
    register(_cls.name, _cls)
register("SOM2IMG_tuned", partial(SOM2IMG, grid_mult=3.0))   # expanded-grid SOM from the paper

#: the nine methods compared in the 51-dataset study (NCTD was excluded there:
#: its 2d x 2d image is infeasible at high d)
PAPER_METHODS = ["SOM2IMG", "SOM2IMG_tuned", "Random", "IGTD", "DeepInsight",
                 "CorrelationBased", "REFINED", "SuperTML", "SnakeRandom"]

__all__ = ["TabularToImage", "FeatureMapMethod", "paint", "snake_fill", "register", "create",
           "available", "PAPER_METHODS", "SOM2IMG", "RandomPixels", "SnakeRandom", "IGTD",
           "DeepInsight", "CorrelationBased", "REFINED", "SuperTML", "NCTD"]
