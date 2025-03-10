# -*- coding: utf-8 -*-
"""Dummy time series regressor."""

__author__ = ["Badr-Eddine Marani"]
__all__ = ["DummyRegressor"]

import numpy as np
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor

from aeon.regression.base import BaseRegressor


class DummyRegressor(BaseRegressor):
    """DummyRegressor makes predictions that ignore the input features.

    This regressor serves as a simple baseline to compare against other more
    complex regressors.
    The specific behavior of the baseline is selected with the `strategy`
    parameter.

    All strategies make predictions that ignore the input feature values passed
    as the `X` argument to `fit` and `predict`. The predictions, however,
    typically depend on values observed in the `y` parameter passed to `fit`.

    Function-identical to `sklearn.dummy.DummyRegressor`, which is called
    inside.

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant"}, default="mean"
        Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
        provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
        the user.

    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.
    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:missing_values": True,
        "capability:unequal_length": False,
        "capability:multivariate": True,
    }

    def __init__(self, strategy="mean", constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile
        self.sklearn_dummy_regressor = SklearnDummyRegressor(
            strategy=strategy, constant=constant, quantile=quantile
        )
        super(DummyRegressor, self).__init__()

    def _fit(self, X, y) -> np.ndarray:
        """Fit the dummy regressor.

        Parameters
        ----------
        X : 3D np.array of shape [n_instances, n_channels, series_length]
        y : array-like, shape = [n_instances] - the target values

        Returns
        -------
        self : reference to self.
        """
        self.sklearn_dummy_regressor.fit(None, y)
        return self

    def _predict(self, X) -> np.ndarray:
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : 3D np.array of shape [n_instances, n_channels, series_length]

        Returns
        -------
        y : predictions of target values for X, np.ndarray
        """
        return self.sklearn_dummy_regressor.predict(np.zeros(X.shape))
