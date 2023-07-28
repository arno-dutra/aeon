
__all__ = [
    "ExtractedEncoder",
]
__author__ = ["Arno Dutra"]

from aeon.base import BaseEstimator
from aeon.utils.validation._dependencies import _check_estimator_deps

class ExtractedEncoder(BaseEstimator):
    def __init__(self, model, autoencoder=None):
        super(ExtractedEncoder, self).__init__()
        _check_estimator_deps(self)

        self.model = model
        self.autoencoder = autoencoder
        self._is_fitted = True

        self._estimator_type = "encoder"


    def __call__(self, X, *args, use_transpose=True, **kwargs):
        if use_transpose:
            X = X.transpose(0, 2, 1)
        X = self.model(X, *args, **kwargs)
        X = X.numpy()
        if use_transpose:
            X = X.transpose(0, 2, 1)
        return X