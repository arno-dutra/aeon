# -*- coding: utf-8 -*-
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)
"""
Abstract base class for time series regressors.

    class name: BaseRegressor

Defining methods:
    fitting         - fit(self, X, y)
    predicting      - predict(self, X)

Inherited inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__all__ = [
    "BaseRegressor",
]
__author__ = ["MatthewMiddlehurst", "TonyBagnll", "mloning", "fkiraly"]
import time
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
import pandas as pd

from aeon.base import BaseEstimator
from aeon.datatypes import check_is_scitype, convert_to
from aeon.utils.sklearn import is_sklearn_transformer
from aeon.utils.validation import check_n_jobs


class BaseRegressor(BaseEstimator, ABC):
    """Abstract base class for time series regressors.

    The base regressor specifies the methods and method signatures that all
    regressors have to implement. Attributes with a underscore suffix are set in the
    method fit.

    Parameters
    ----------
    fit_time_           : integer, time (in milliseconds) for fit to run.
    _class_dictionary   : dictionary mapping classes_ onto integers 0...n_classes_-1.
    _n_jobs     : number of threads to use in fit as determined by n_jobs.
    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
    }

    def __init__(self):
        self.fit_time_ = 0
        self._class_dictionary = {}
        self._threads_to_use = 1
        self._X_metadata = {}

        # required for compatability with some sklearn interfaces
        # i.e. CalibratedRegressorCV
        self._estimator_type = "regressor"

        super(BaseRegressor, self).__init__()

    def __rmul__(self, other):
        """Magic * method, return concatenated RegressorPipeline, transformers on left.

        Overloaded multiplication operation for regressors. Implemented for `other`
        being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `aeon` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        RegressorPipeline object, concatenation of `other` (first) with `self` (last).
        """
        from aeon.regression.compose import RegressorPipeline
        from aeon.transformations.base import BaseTransformer
        from aeon.transformations.compose import TransformerPipeline
        from aeon.transformations.series.adapt import TabularToSeriesAdaptor

        # behaviour is implemented only if other inherits from BaseTransformer
        #  in that case, distinctions arise from whether self or other is a pipeline
        if isinstance(other, BaseTransformer):
            # RegressorPipeline already has the dunder method defined
            if isinstance(self, RegressorPipeline):
                return other * self
            # if other is a TransformerPipeline but self is not, first unwrap it
            elif isinstance(other, TransformerPipeline):
                return RegressorPipeline(regressor=self, transformers=other.steps)
            # if neither self nor other are a pipeline, construct a RegressorPipeline
            else:
                return RegressorPipeline(regressor=self, transformers=[other])
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    def fit(self, X, y):
        """Fit time series regressor to training data.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape (n_instances, n_channels, n_timepoints)
            or 2D np.array (univariate, equal length series)
                of shape (n_instances, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_instances], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i
            other types are allowed and converted into one of the above.
        y : 1D np.array of float, of shape (n_instances) - regression labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        start = int(round(time.time() * 1000))
        # convenience conversions to allow user flexibility:
        # if X is 2D array, convert to 3D, if y is pd.Series, convert to numpy
        X, y = self._internal_convert(X, y)
        if isinstance(y, np.ndarray):
            y = y.astype("float")
        self._X_metadata = _check_regressor_input(X, y)
        missing = self._X_metadata["has_nans"]
        multivariate = not self._X_metadata["is_univariate"]
        unequal = not self._X_metadata["is_equal_length"]
        # Check this regressor can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the regressor tags
        X = self._convert_X(X)
        multithread = self.get_tag("capability:multithreading")
        if multithread:
            try:
                self._threads_to_use = check_n_jobs(self.n_jobs)
            except NameError:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )

        self._fit(X, y)
        self.fit_time_ = int(round(time.time() * 1000)) - start
        # this should happen last
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape (n_instances, n_channels, n_timepoints)
            or 2D np.array (univariate, equal length series)
                of shape (n_instances, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_instances], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i
            other types are allowed and converted into one of the above.

        Returns
        -------
        y : 1D np.array of float, of shape (n_instances) - predicted regression labels
            indices correspond to instance indices in X
        """
        self.check_is_fitted()

        # boilerplate input checks for predict-like methods
        X = self._check_convert_X_for_predict(X)

        return self._predict(X)

    def score(self, X, y) -> float:
        """Scores predicted labels against ground truth labels on X.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape (n_instances, n_channels, n_timepoints)
            or 2D np.array (univariate, equal length series)
                of shape (n_instances, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_instances], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i
            other types are allowed and converted into one of the above.
        y : 1D np.array of float, of shape (n_instances) - regression labels (ground
        truth)
            indices correspond to instance indices in X

        Returns
        -------
        float, R-squared score of predict(X) vs y
        """
        from sklearn.metrics import r2_score

        self.check_is_fitted()
        if isinstance(y, pd.Series):
            y = pd.Series.to_numpy(y)
        y = y.astype("float")

        return r2_score(y, self.predict(X))

    @abstractmethod
    def _fit(self, X, y):
        """Fit time series regressor to training data.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = (n_instances, n_channels, n_timepoints)
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
        y : 1D np.array of float, of shape (n_instances) - regression labels for
        fitting indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes ending in "_"
        """
        ...

    @abstractmethod
    def _predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Abstract method, must be implemented.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = (n_instances, n_channels, n_timepoints)
            for list of other mtypes, see datatypes.SCITYPE_REGISTER

        Returns
        -------
        y : 1D np.array of float, of shape (n_instances) - predicted regression labels
            indices correspond to instance indices in X
        """
        ...

    def _check_convert_X_for_predict(self, X):
        """Input checks, capability checks, repeated in all predict/score methods.

        Parameters
        ----------
        X : any object (to check/convert)
            should be of a supported input type including 2D numpy.ndarray

        Returns
        -------
        X: an object of a supported input type, numpy3D if X was a 2D numpy.ndarray

        Raises
        ------
        ValueError if X is of invalid input data type, or there is not enough data
        ValueError if the capabilities in self._tags do not handle the data.
        """
        X = self._internal_convert(X)
        X_metadata = _check_regressor_input(X)
        missing = X_metadata["has_nans"]
        multivariate = not X_metadata["is_univariate"]
        unequal = not X_metadata["is_equal_length"]
        # Check this regressor can handle characteristics
        self._check_capabilities(missing, multivariate, unequal)
        # Convert data as dictated by the regressor tags
        X = self._convert_X(X)

        return X

    def _check_capabilities(self, missing, multivariate, unequal):
        """Check whether this regressor can handle the data characteristics.

        Parameters
        ----------
        missing : boolean, does the data passed to fit contain missing values?
        multivariate : boolean, does the data passed to fit contain missing values?
        unequal : boolea, do the time series passed to fit have variable lengths?

        Raises
        ------
        ValueError if the capabilities in self._tags do not handle the data.
        """
        allow_multivariate = self.get_tag("capability:multivariate")
        allow_missing = self.get_tag("capability:missing_values")
        allow_unequal = self.get_tag("capability:unequal_length")

        self_name = type(self).__name__

        # identify problems, mismatch of capability and inputs
        problems = []
        if missing and not allow_missing:
            problems += ["missing values"]
        if multivariate and not allow_multivariate:
            problems += ["multivariate series"]
        if unequal and not allow_unequal:
            problems += ["unequal length series"]

        # construct error message
        problems_and = " and ".join(problems)
        problems_or = " or ".join(problems)
        msg = (
            f"Data seen by {self_name} instance has {problems_and}, "
            f"but this {self_name} instance cannot handle {problems_or}. "
            f"Calls with {problems_or} may result in error or unreliable results."
        )

        # raise exception or warning with message
        # if self is composite, raise a warning, since passing could be fine
        #   see discussion in PR 2366 why
        if len(problems) > 0:
            if self.is_composite():
                warn(msg)
            else:
                raise ValueError(msg)

    def _convert_X(self, X):
        """Convert equal length series from DataFrame to numpy array or vice versa.

        Parameters
        ----------
        self : this regressor
        X : pd.DataFrame or np.ndarray. Input attribute data

        Returns
        -------
        X : input X converted to type in "X_inner_mtype" tag, default 3D np.ndarray
            Checked and possibly converted input data
        """
        inner_type = self.get_tag("X_inner_mtype")
        # convert pd.DataFrame
        X = convert_to(
            X,
            to_type=inner_type,
            as_scitype="Panel",
        )
        return X


def _check_regressor_input(
    X,
    y=None,
    enforce_min_instances=1,
):
    """Check whether input X and y are valid formats with minimum data.

    Raises a ValueError if the input is not valid.

    Parameters
    ----------
    X : check whether X is a valid input type
    y : check whether a pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        check there are a minimum number of instances.

    Returns
    -------
    metadata : dict with metadata for X returned by datatypes.check_is_scitype

    Raises
    ------
    ValueError
        If y or X is invalid input data type, or there is not enough data
    """
    # Check X is valid input type and recover the data characteristics
    X_valid, _, X_metadata = check_is_scitype(X, scitype="Panel", return_metadata=True)
    if not X_valid:
        raise TypeError(
            f"X is not of a supported input data type."
            f"X must be in a supported type format for Regression, found {type(X)}"
            f"Use datatypes.check_is_mtype to check conformance with specifications."
        )
    n_cases = X_metadata["n_instances"]
    if n_cases < enforce_min_instances:
        raise ValueError(
            f"Minimum number of cases required is {enforce_min_instances} but X "
            f"has : {n_cases}"
        )

    # Check y if passed
    if y is not None:
        # Check y valid input
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError(
                f"y must be a np.array or a pd.Series, but found type: {type(y)}"
            )
        # Check matching number of labels
        n_labels = y.shape[0]
        if n_cases != n_labels:
            raise ValueError(
                f"Mismatch in number of cases. Number in X = {n_cases} nos in y = "
                f"{n_labels}"
            )
        if isinstance(y, np.ndarray):
            if y.ndim > 1:
                raise ValueError(
                    f"np.ndarray y must be 1-dimensional, "
                    f"but found {y.ndim} channels"
                )
    return X_metadata
