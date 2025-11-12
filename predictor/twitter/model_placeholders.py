"""Small importable placeholders for scaler/classifier used as safe fallbacks.

Placing these in a proper module path ensures pickles containing instances
can be unpickled by other modules (instead of relying on __main__).
"""
from __future__ import annotations

import numpy as np


class SimpleScaler:
    """Minimal scaler with a no-op transform."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)


class SimpleClassifier:
    """Minimal classifier that predicts no-jump (zeros)."""
    def predict(self, X):
        X = np.asarray(X)
        return np.zeros(X.shape[0], dtype=int)
