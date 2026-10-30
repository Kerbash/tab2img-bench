"""Classifiers: CNNs for T2I images and standard tabular baselines."""
from tab2img.models.cnn import ARCHITECTURES
from tab2img.models.tabular import BASELINES, make_baseline

__all__ = ["ARCHITECTURES", "BASELINES", "make_baseline"]
