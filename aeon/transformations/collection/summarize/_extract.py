# -*- coding: utf-8 -*-
"""Sequence feature extraction transformers."""
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit

from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection.segment import RandomIntervalSegmenter


@njit(fastmath=True, cache=True)
def _der(x: np.ndarray):
    """Loop based Derivative Slope transform."""
    m = len(x)
    der = np.zeros(m)
    for i in range(1, m - 1):
        der[i] = ((x[i] - x[i - 1]) + ((x[i + 1] - x[i - 1]) / 2.0)) / 2.0
    der[0] = der[1]
    der[m - 1] = der[m - 2]
    return der


def series_slope_derivative(X: np.ndarray) -> np.ndarray:
    """Find the slope derivative of collection of time series.

    takes any shape numpy array and finds the derivative of the last axis, padding
    the first and last values so that the length stays the same

    Parameters
    ----------
    X: np.ndarray

    Returns
    -------
    np.ndarray same shape as X

    Example
    -------
    >>> from aeon.transformations.collection.summarize import series_slope_derivative
    >>> x=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    >>> series_slope_derivative(x)
    array([ 1. ,  1. ,  1. ,  1. ,  0.5, -1. , -1. , -1. , -1. ])
    >>> x=np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    >>> series_slope_derivative(x)
    array([[ 1.,  1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.]])
    >>> x=np.random.random((10,3,50)) # 10 time series, 3 channels, length 50
    >>> x2 = series_slope_derivative(x)
    """
    return np.apply_along_axis(_der, axis=-1, arr=X)


class PlateauFinder(BaseTransformer):
    """Plateau finder transformer.

    Transformer that finds segments of the same given value, plateau in
    the time series, and returns the starting indices and lengths.

    Parameters
    ----------
    value : {int, float, np.nan, np.inf}
        Value for which to find segments
    min_length : int
        Minimum lengths of segments with same value to include.
        If min_length is set to 1, the transformer can be used as a value
        finder.
    """

    _tags = {
        "fit_is_empty": True,
        "univariate-only": True,
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
    }

    def __init__(self, value=np.nan, min_length=2):
        self.value = value
        self.min_length = min_length
        super(PlateauFinder, self).__init__(_output_convert=False)

    def _transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X : numpy3D array shape (n_cases, 1, series_length)

        Returns
        -------
        X : pandas data frame
        """
        _starts = []
        _lengths = []

        # find plateaus (segments of the same value)
        for x in X[:, 0]:
            # find indices of transition
            if np.isnan(self.value):
                i = np.where(np.isnan(x), 1, 0)

            elif np.isinf(self.value):
                i = np.where(np.isinf(x), 1, 0)

            else:
                i = np.where(x == self.value, 1, 0)

            # pad and find where segments transition
            transitions = np.diff(np.hstack([0, i, 0]))

            # compute starts, ends and lengths of the segments
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            lengths = ends - starts

            # filter out single points
            starts = starts[lengths >= self.min_length]
            lengths = lengths[lengths >= self.min_length]

            _starts.append(starts)
            _lengths.append(lengths)

        # put into dataframe
        Xt = pd.DataFrame()
        column_prefix = "%s_%s" % (
            "channel_",
            "nan" if np.isnan(self.value) else str(self.value),
        )
        Xt["%s_starts" % column_prefix] = pd.Series(_starts)
        Xt["%s_lengths" % column_prefix] = pd.Series(_lengths)

        Xt = Xt.applymap(lambda x: pd.Series(x))
        return Xt


class DerivativeSlopeTransformer(BaseTransformer):
    """Derivative slope transformer.

    Transformer that finds the slope derivate by simply calling the method
    `series_slope_derivative`. This function keeps the series the same length by
    copying the first and last values with the second and last but one.
    """

    _tags = {
        "fit_is_empty": True,
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
    }

    def _transform(self, X, y=None):
        """Transform X."""
        return series_slope_derivative(X)


def _check_features(features):
    if features is None:
        return [np.mean]
    elif isinstance(features, list) and all([callable(func) for func in features]):
        return features
    else:
        raise ValueError(
            "Features must be list containing only functions (callables) "
            "to be applied to the data columns"
        )


class RandomIntervalFeatureExtractor(BaseTransformer):
    """Random interval feature extractor transform.

    Transformer that segments time-series into random intervals
    and subsequently extracts series-to-primitives features from each interval.

    n_intervals: str{'sqrt', 'log', 'random'}, int or float, optional (
    default='sqrt')
        Number of random intervals to generate, where m is length of time
        series:
        - If "log", log of m is used.
        - If "sqrt", sqrt of m is used.
        - If "random", random number of intervals is generated.
        - If int, n_intervals intervals are generated.
        - If float, int(n_intervals * m) is used with n_intervals giving the
        fraction of intervals of the
        time series length.

        For all arguments relative to the length of the time series,
        the generated number of intervals is
        always at least 1.

    features: list of functions, optional (default=None)
        Applies each function to random intervals to extract features.
        If None, the mean is extracted.

    random_state: : int, RandomState instance, optional (default=None)
        - If int, random_state is the seed used by the random number generator;
        - If RandomState instance, random_state is the random number generator;
        - If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    _tags = {
        "fit_is_empty": False,
        "univariate-only": True,
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "pd_Series_Table",
    }

    def __init__(
        self,
        n_intervals="sqrt",
        min_length=None,
        max_length=None,
        features=None,
        random_state=None,
    ):
        self.n_intervals = n_intervals
        self.min_length = min_length
        self.max_length = max_length
        self.random_state = random_state
        self.features = features
        super(RandomIntervalFeatureExtractor, self).__init__(_output_convert=False)

    def _fit(self, X, y=None):
        """
        Fit transformer, generating random interval indices.

        Parameters
        ----------
        X: np.ndarray shape (n_time_series, 1, series_length)
            The training input samples.
        y : arraylike, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self :
            This estimator
        """
        self._interval_segmenter = RandomIntervalSegmenter(
            self.n_intervals, self.min_length, self.max_length, self.random_state
        )
        self._interval_segmenter.fit(X, y)
        self.intervals_ = self._interval_segmenter.intervals_
        self.input_shape_ = self._interval_segmenter.input_shape_
        self._time_index = self._interval_segmenter._time_index
        return self

    def _transform(self, X, y=None):
        """Transform X.

        Transform X, segments time-series in each column into random
        intervals using interval indices generated
        during `fit` and extracts features from each interval.

        Parameters
        ----------
        X: np.ndarray shape (n_time_series, 1, series_length)
            The training input samples.

        Returns
        -------
        Xt : pandas.DataFrame
          Transformed pandas DataFrame with n_instances rows and one
            column for each generated interval.
        """
        # Check input of feature calculators, i.e list of functions to be
        # applied to time-series
        features = _check_features(self.features)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError(
                "Number of columns of input is different from what was seen in `fit`"
            )
        # Input validation
        # if not all([np.array_equal(fit_idx, trans_idx) for trans_idx,
        # fit_idx in zip(check_equal_index(X),
        #     raise ValueError('Indexes of input time-series are different
        #     from what was seen in `fit`')

        n_instances, _, _ = X.shape
        n_features = len(features)

        intervals = self.intervals_
        n_intervals = len(intervals)

        # Compute features on intervals.
        Xt = np.zeros((n_instances, n_features * n_intervals))  # Allocate output array
        # for transformed data
        columns = []

        i = 0
        drop_list = []
        for func in features:
            for start, end in intervals:
                interval = X[:, :, start:end]

                # Try to use optimised computations over axis if possible,
                # otherwise iterate over rows.
                try:
                    Xt[:, i] = func(interval, axis=-1).squeeze()
                except TypeError as e:
                    if (
                        str(e) == f"{func.__name__}() got an unexpected "
                        f"keyword argument 'axis'"
                    ):
                        Xt[:, i] = np.apply_along_axis(
                            func, axis=2, arr=interval
                        ).squeeze()
                    else:
                        raise
                new_col_name = f"{start}_{end}_{func.__name__}"
                if new_col_name in columns:
                    drop_list += [i]
                else:
                    columns = columns + [new_col_name]
                i += 1

        Xt = pd.DataFrame(Xt)
        Xt = Xt.drop(columns=Xt.columns[drop_list])
        Xt.columns = columns

        return Xt


class FittedParamExtractor(BaseTransformer):
    """Fitted parameter extractor.

    Extract parameters of a fitted forecaster as features for a subsequent
    tabular learning task.
    This class first fits a forecaster to the given time series and then
    returns the fitted parameters.
    The fitted parameters can be used as features for a tabular estimator
    (e.g. classification).

    Parameters
    ----------
    forecaster : estimator object
        aeon estimator to extract features from
    param_names : str
        Name of parameters to extract from the forecaster.
    n_jobs : int, optional (default=None)
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    """

    _tags = {
        "fit_is_empty": True,
        "univariate-only": True,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
    }

    def __init__(self, forecaster, param_names, n_jobs=None):
        self.forecaster = forecaster
        self.param_names = param_names
        self.n_jobs = n_jobs
        super(FittedParamExtractor, self).__init__(_output_convert=True)

    def _transform(self, X, y=None):
        """Transform X.

        Parameters
        ----------
        X: np.ndarray shape (n_time_series, 1, series_length)
            The training input samples.
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.DataFrame
            Extracted parameters; columns are parameter values
        """
        param_names = self._check_param_names(self.param_names)
        n_instances = X.shape[0]

        def _fit_extract(forecaster, x, param_names):
            forecaster.fit(x)
            params = forecaster.get_fitted_params()
            return np.hstack([params[name] for name in param_names])

        def _get_instance(X, key):
            # assuming univariate data
            if isinstance(X, pd.DataFrame):
                return X.iloc[key, 0]
            else:
                return pd.Series(X[key, 0])

        # iterate over rows
        extracted_params = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_extract)(
                self.forecaster.clone(), _get_instance(X, i), param_names
            )
            for i in range(n_instances)
        )

        return pd.DataFrame(extracted_params, columns=param_names)

    @staticmethod
    def _check_param_names(param_names):
        if isinstance(param_names, str):
            param_names = [param_names]
        elif isinstance(param_names, (list, tuple)):
            for param in param_names:
                if not isinstance(param, str):
                    raise ValueError(
                        f"All elements of `param_names` must be strings, "
                        f"but found: {type(param)}"
                    )
        else:
            raise ValueError(
                f"`param_names` must be str, or a list or tuple of strings, "
                f"but found: {type(param_names)}"
            )
        return param_names

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from aeon.forecasting.trend import TrendForecaster

        # accessing a nested parameter
        params = [
            {
                "forecaster": TrendForecaster(),
                "param_names": ["regressor__intercept"],
            }
        ]
        return params
