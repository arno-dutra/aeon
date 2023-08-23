
__all__ = [
    "ExtractedEncoder",
]
__author__ = ["Arno Dutra"]

from aeon.base import BaseEstimator
from aeon.utils.validation._dependencies import _check_estimator_deps
import tensorflow as tf

class ExtractedEncoder(BaseEstimator):
    def __init__(self, model=None, autoencoder=None):
        super(ExtractedEncoder, self).__init__()
        _check_estimator_deps(self)

        self.model = model
        self.autoencoder = autoencoder
        self.__name__ = autoencoder.__name__.replace("AutoEncoder", "Encoder")
        self._is_fitted = True
        self._estimator_type = "encoder"

    def __call__(self, X, *args, use_transpose=True, **kwargs):
        if use_transpose:
            X = X.transpose(0, 2, 1)
        X = self.model(X, *args, **kwargs)
        X = X.numpy()
        return X
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self._is_fitted = True