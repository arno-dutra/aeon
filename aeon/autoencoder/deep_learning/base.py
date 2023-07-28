# -*- coding: utf-8 -*-
"""
Abstract base class for the Keras neural network classifiers.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags, _predict and _predict_proba
"""
__author__ = [
    "Arno Dutra",
    "James-Large",
    "ABostrom",
    "TonyBagnall",
    "aurunmpegasus",
    "achieveordie",
    "hadifawaz1999",
]
__all__ = ["BaseDeepAutoEncoder"]

from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state

from aeon.autoencoder.base import BaseAutoEncoder


class BaseDeepAutoEncoder(BaseAutoEncoder, ABC):
    """Abstract base class for deep learning time series classifiers.

    The base classifier provides a deep learning default method for
    _predict and _predict_proba, and provides a new abstract method for building a
    model.

    Parameters
    ----------
    batch_size : int, default = 40
        training batch size for the model
    last_file_name      : str, default = "last_model"
        The name of the file of the last model, used
        only if save_last_model_to_file is used

    Arguments
    ---------
    self.model = None

    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "algorithm_type": "deeplearning",
        "non-deterministic": True,
        "cant-pickle": True,
        "python_dependencies": "tensorflow",
    }

    def __init__(
        self,
        batch_size=40,
        random_state=None,
        last_file_name="last_model",
    ):
        super(BaseDeepAutoEncoder, self).__init__()

        self.batch_size = batch_size
        self.random_state = random_state
        self.last_file_name = last_file_name
        self.model_ = None

    @abstractmethod
    def build_model(self, input_shape, n_classes):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        n_classes: int
            The number of classes, which shall become the size of the output
            layer

        Returns
        -------
        A compiled Keras Model
        """
        ...

    def summary(self):
        """
        Summary function to return the losses/metrics for model fit.

        Returns
        -------
        history: dict or None,
            Dictionary containing model's train/validation losses and metrics

        """
        return self.history.history if self.history is not None else None

    def _predict(self, X):
        X = X.transpose((0, 2, 1))
        xhat = self.model_.predict(X, self.batch_size)

        return xhat.transpose((0, 2, 1))

    def save_last_model_to_file(self, file_path="./"):
        """Save the last epoch of the trained deep learning model.

        Parameters
        ----------
        file_path : str, default = "./"
            The directory where the model will be saved

        Returns
        -------
        None
        """
        self.model_.save(file_path + self.last_file_name + ".hdf5")
