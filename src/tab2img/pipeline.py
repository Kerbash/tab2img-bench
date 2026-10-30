"""The T2I classification pipeline: tabular row -> image -> CNN -> label.

    from tab2img import T2IClassifier
    clf = T2IClassifier("SOM2IMG", cnn="simple", epochs=30)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

`method` is a registry name (see `tab2img.methods.available()`) or an
already-built `TabularToImage` object (cloned on every fit). Inputs should be
scaled to [0, 1] (the benchmark fits a MinMaxScaler on the training fold).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from torch.utils.data import DataLoader, TensorDataset

from tab2img.methods import TabularToImage, create
from tab2img.models.cnn import ARCHITECTURES


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_images(images: np.ndarray, quantize: bool = True, clip: bool = False,
                  channels: int = 3) -> np.ndarray:
    """Turn method output (n, H, W) into CNN input (n, H, W, channels) float32.

    quantize=True stores pixels as 8-bit (0..255) and divides by 255, as the
    original benchmark did when it produced PNG-style uint8 images.

    clip=False reproduces the paper exactly. Note: values outside [0, 1]
    (test rows beyond the training range) then *wrap around* in the uint8
    cast (e.g. 1.02 -> 260 -> 4/255, a near-black pixel). clip=True clips to
    [0, 1] first, which is the sensible choice for new work.
    """
    x = np.clip(images, 0.0, 1.0) if clip else images
    if quantize:
        x = (x * 255).astype(np.uint8)
        x = np.repeat(x[..., None], channels, axis=-1).astype(np.float32) / 255.0
    else:
        x = np.repeat(x[..., None], channels, axis=-1).astype(np.float32)
    return x


class T2IClassifier(ClassifierMixin, BaseEstimator):
    """A T2I method followed by a CNN, as one scikit-learn classifier.

    Parameters
    ----------
    method : str or TabularToImage
    method_params : dict, extra arguments when `method` is a name
    image_shape : (rows, cols) or None (auto from the number of features)
    cnn : one of tab2img.models.cnn.ARCHITECTURES ("simple" = paper)
    epochs, batch_size, seed : training settings (paper: 30, 32, 42)
    quantize, clip : see `encode_images`
    device : "cuda" / "cpu" / None (auto)
    """

    def __init__(self, method="SOM2IMG", method_params=None, image_shape=None, cnn="simple",
                 epochs=30, batch_size=32, seed=42, quantize=True, clip=False, device=None):
        self.method = method
        self.method_params = method_params
        self.image_shape = image_shape
        self.cnn = cnn
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.quantize = quantize
        self.clip = clip
        self.device = device

    # -- helpers ---------------------------------------------------------
    def _build_method(self) -> TabularToImage:
        if isinstance(self.method, TabularToImage):
            m = clone(self.method)
            if self.image_shape is not None:
                m.set_params(image_shape=self.image_shape)
            return m
        params = dict(self.method_params or {})
        if self.image_shape is not None:
            params.setdefault("image_shape", self.image_shape)
        return create(self.method, **params)

    def _images(self, X) -> torch.Tensor:
        imgs = encode_images(self.transformer_.transform(X), self.quantize, self.clip)
        return torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()

    # -- sklearn API -----------------------------------------------------
    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, np.arange(len(self.classes_))):
            raise ValueError("y must be integer labels 0..k-1 (use LabelEncoder)")
        k = len(self.classes_)
        self.device_ = torch.device(self.device) if self.device else default_device()

        self.transformer_ = self._build_method().fit(X, y)
        Xtr = self._images(X)

        torch.manual_seed(self.seed)
        model = ARCHITECTURES[self.cnn](k).to(self.device_)
        with torch.no_grad():
            model(Xtr[:2].to(self.device_))       # materialise LazyLinear layers
        if self.cnn == "resnet18":                # gentle fine-tuning of the pretrained net
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        else:
            opt = torch.optim.Adam(model.parameters())
        crit = nn.BCEWithLogitsLoss() if k == 2 else nn.CrossEntropyLoss()

        loader = DataLoader(TensorDataset(Xtr, torch.as_tensor(y, dtype=torch.long)),
                            batch_size=self.batch_size, shuffle=True)
        model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out.squeeze(1), yb.float()) if k == 2 else crit(out, yb)
                loss.backward()
                opt.step()
        model.eval()
        self.model_ = model
        return self

    def predict_proba(self, X) -> np.ndarray:
        with torch.no_grad():
            out = self.model_(self._images(X).to(self.device_))
            if len(self.classes_) == 2:
                p1 = torch.sigmoid(out.squeeze(1)).cpu().numpy()
                return np.column_stack([1 - p1, p1])
            return torch.softmax(out, dim=1).cpu().numpy()

    def predict(self, X) -> np.ndarray:
        p = self.predict_proba(X)
        if len(self.classes_) == 2:
            return (p[:, 1] > 0.5).astype(int)    # same threshold rule as the paper
        return p.argmax(1)

    def release(self):
        """Free the CNN (and GPU memory) once predictions are no longer needed."""
        self.model_ = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
