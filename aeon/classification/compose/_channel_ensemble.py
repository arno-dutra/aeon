# -*- coding: utf-8 -*-
"""ChannelEnsembleClassifier: For Multivariate Time Series Classification.

Builds classifiers on each channel (dimension) independently.
"""

__author__ = ["abostrom", "TonyBagnall"]
__all__ = ["ChannelEnsembleClassifier"]

from itertools import chain

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from aeon.base import _HeterogenousMetaEstimator
from aeon.classification.base import BaseClassifier


class _BaseChannelEnsembleClassifier(_HeterogenousMetaEstimator, BaseClassifier):
    """Base Class for channel ensemble."""

    _tags = {
        "capability:multivariate": True,
    }

    def __init__(self, estimators, verbose=False):
        self.verbose = verbose
        self.estimators = estimators
        self.remainder = "drop"
        super(_BaseChannelEnsembleClassifier, self).__init__()
        self._anytagis_then_set(
            "capability:unequal_length", False, True, self._estimators
        )
        self._anytagis_then_set(
            "capability:missing_values", False, True, self._estimators
        )

    @property
    def _estimators(self):
        return [(name, estimator) for name, estimator, _ in self.estimators]

    @_estimators.setter
    def _estimators(self, value):
        self.estimators = [
            (name, estimator, col)
            for ((name, estimator), (_, _, col)) in zip(value, self.estimators)
        ]

    def _validate_estimators(self):
        if not self.estimators:
            return

        names, estimators, _ = zip(*self.estimators)

        self._check_names(names)

        # validate estimators
        for t in estimators:
            if t == "drop":
                continue
            if not (hasattr(t, "fit") or hasattr(t, "predict_proba")):
                raise TypeError(
                    "All estimators should implement fit and predict proba"
                    "or can be 'drop' "
                    "specifiers. '%s' (type %s) doesn't." % (t, type(t))
                )

    def _validate_channel_callables(self, X):
        """Convert callable channel specifications."""
        channels = []
        for _, _, channel in self.estimators:
            if callable(channel):
                channel = channel(X)
            channels.append(channel)
        self._channels = channels

    def _validate_remainder(self, X):
        """Validate ``remainder`` and defines ``_remainder``."""
        is_estimator = hasattr(self.remainder, "fit") or hasattr(
            self.remainder, "predict_proba"
        )
        if self.remainder != "drop" and not is_estimator:
            raise ValueError(
                "The remainder keyword needs to be 'drop', '%s' was passed "
                "instead" % self.remainder
            )

        n_channels = X.shape[1]
        cols = []
        for channels in self._channels:
            cols.extend(_get_channel_indices(X, channels))
        remaining_idx = sorted(list(set(range(n_channels)) - set(cols))) or None

        self._remainder = ("remainder", self.remainder, remaining_idx)

    def _iter(self, replace_strings=False):
        """Generate (name, estimator, channel) tuples.

        If fitted=True, use the fitted transformations, else use the
        user specified transformations updated with converted channel names
        and potentially appended with transformer for remainder.
        """
        if self.is_fitted:
            estimators = self.estimators_
        else:
            # interleave the validated channel specifiers
            estimators = [
                (name, estimator, channel)
                for (name, estimator, _), channel in zip(
                    self.estimators, self._channels
                )
            ]

        # add transformer tuple for remainder
        if self._remainder[2] is not None:
            estimators = chain(estimators, [self._remainder])

        for name, estimator, channel in estimators:
            if replace_strings and (
                estimator == "drop"
                or estimator != "drop"
                and _is_empty_channel_selection(channel)
            ):
                continue
            yield name, estimator, channel

    def _fit(self, X, y):
        """Fit all estimators, fit the data.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]

        y : array-like, shape = [n_instances]
            The class labels.

        """
        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError(
                "Invalid `estimators` attribute, `estimators`"
                " should be a list of (string, estimator)"
                " tuples"
            )

        self._validate_estimators()
        self._validate_channel_callables(X)
        self._validate_remainder(X)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        estimators_ = []
        for name, estimator, channel in self._iter(replace_strings=True):
            estimator = estimator.clone()
            estimator.fit(_get_channel(X, channel), transformed_y)
            estimators_.append((name, estimator, channel))

        self.estimators_ = estimators_
        return self

    def _collect_probas(self, X):
        return np.asarray(
            [
                estimator.predict_proba(_get_channel(X, channel))
                for (name, estimator, channel) in self._iter(replace_strings=True)
            ]
        )

    def _predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X using 'soft' voting."""
        return np.average(self._collect_probas(X), axis=0)

    def _predict(self, X) -> np.ndarray:
        maj = np.argmax(self.predict_proba(X), axis=1)
        return self.le_.inverse_transform(maj)


class ChannelEnsembleClassifier(_BaseChannelEnsembleClassifier):
    """Applies estimators to channels of an array.

    This estimator allows different channels or channel subsets of the input
    to be transformed separately and the features generated by each
    transformer will be ensembled to form a single output.

    Parameters
    ----------
    estimators : list of tuples
        List of (name, estimator, channel(s)) tuples specifying the transformer
        objects to be applied to subsets of the data.

        name : string
            Like in Pipeline and FeatureUnion, this allows the
            transformer and its parameters to be set using ``set_params`` and searched
            in grid search.
        estimator :  or {'drop'}
            Estimator must support `fit` and `predict_proba`. Special-cased
            strings 'drop' and 'passthrough' are accepted as well, to
            indicate to drop the channels.
        channels(s) : array-like of int, slice, boolean mask array. Integer channels
        are indexed from 0

    remainder : {'drop', 'passthrough'} or estimator, default 'drop'
        By default, only the specified channels in `transformations` are
        transformed and combined in the output, and the non-specified
        channels are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining channels
        that were not specified in `transformations` will be automatically passed
        through. This subset of channels is concatenated with the output of
        the transformations.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified channels will use the ``remainder`` estimator. The
        estimator must support `fit` and `transform`.

    Examples
    --------
    >>> from aeon.classification.dictionary_based import ContractableBOSS
    >>> from aeon.classification.interval_based import CanonicalIntervalForest
    >>> from aeon.datasets import load_basic_motions
    >>> X_train, y_train = load_basic_motions(split="train")
    >>> X_test, y_test = load_basic_motions(split="test")
    >>> cboss = ContractableBOSS(
    ...     n_parameter_samples=4, max_ensemble_size=2, random_state=0
    ... )
    >>> cif = CanonicalIntervalForest(
    ...     n_estimators=2, n_intervals=4, att_subsample_size=4, random_state=0
    ... )
    >>> estimators = [("cBOSS", cboss, 5), ("CIF", cif, [3, 4])]
    >>> channel_ens = ChannelEnsembleClassifier(estimators=estimators)
    >>> channel_ens.fit(X_train, y_train)
    ChannelEnsembleClassifier(...)
    >>> y_pred = channel_ens.predict(X_test)
    """

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_estimators"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "estimators_"

    def __init__(self, estimators, remainder="drop", verbose=False):
        self.remainder = remainder
        super(ChannelEnsembleClassifier, self).__init__(estimators, verbose=verbose)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            ChannelEnsembleClassifier provides the following special sets:
                 "results_comparison" - used in some classifiers to compare against
                    previously generated results where the default set of parameters
                    cannot produce suitable probability estimates

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from aeon.classification.dictionary_based import ContractableBOSS
        from aeon.classification.interval_based import CanonicalIntervalForest
        from aeon.classification.interval_based import (
            TimeSeriesForestClassifier as TSFC,
        )

        if parameter_set != "results_comparison":
            return {
                "estimators": [
                    ("tsf1", TSFC(n_estimators=2), 0),
                    ("tsf2", TSFC(n_estimators=2), 0),
                ]
            }
        cboss = ContractableBOSS(
            n_parameter_samples=4, max_ensemble_size=2, random_state=0
        )
        cif = CanonicalIntervalForest(
            n_estimators=2, n_intervals=4, att_subsample_size=4, random_state=0
        )
        return {"estimators": [("cBOSS", cboss, 5), ("CIF", cif, [3, 4])]}


def _get_channel(X, key):
    """
    Get time series channel(s) from input data X.

    Supported input types (X): numpy arrays

    Supported key types (key):
    - scalar: output is 1D
    - lists, slices, boolean masks: output is 2D
    - callable that returns any of the above

    Supported key data types:

    - integer or boolean mask (positional):
        - supported for arrays and sparse matrices
    - string (key-based):
        - only supported for dataframes
        - So no keys other than strings are allowed (while in principle you
          can use any hashable object as key).

    """
    # check whether we have string channel names or integers
    if _check_key_type(key, int):
        channel_names = False
    elif hasattr(key, "dtype") and np.issubdtype(key.dtype, np.bool_):
        # boolean mask
        channel_names = True
    else:
        raise ValueError(
            "No valid specification of the channels. Only a "
            "scalar, list or slice of all integers or all "
            "strings, or boolean mask is allowed"
        )

    if isinstance(key, (int, str)):
        key = [key]

    if not channel_names:
        return X[:, key] if isinstance(X, np.ndarray) else X.iloc[:, key]
    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            f"X must be a pd.DataFrame if channel names are "
            f"specified, but found: {type(X)}"
        )
    return X.loc[:, key]


def _check_key_type(key, superclass):
    """
    Check that scalar, list or slice is of a certain type.

    This is only used in _get_channel and _get_channel_indices to check
    if the `key` (channel specification) is fully integer or fully string-like.

    Parameters
    ----------
    key : scalar, list, slice, array-like
        The channel specification to check
    superclass : int or str
        The type for which to check the `key`

    """
    if isinstance(key, superclass):
        return True
    if isinstance(key, slice):
        return isinstance(key.start, (superclass, type(None))) and isinstance(
            key.stop, (superclass, type(None))
        )
    if isinstance(key, list):
        return all(isinstance(x, superclass) for x in key)
    if hasattr(key, "dtype"):
        if superclass is int:
            return key.dtype.kind == "i"
        else:
            # superclass = str
            return key.dtype.kind in ("O", "U", "S")
    return False


def _get_channel_indices(X, key):
    """
    Get feature channel indices for input data X and key.

    For accepted values of `key`, see the docstring of _get_channel

    """
    n_channels = X.shape[1]

    if (
        _check_key_type(key, int)
        or hasattr(key, "dtype")
        and np.issubdtype(key.dtype, np.bool_)
    ):
        # Convert key into positive indexes
        idx = np.arange(n_channels)[key]
        return np.atleast_1d(idx).tolist()
    elif _check_key_type(key, str):
        try:
            all_columns = list(X.columns)
        except AttributeError as e:
            raise ValueError(
                "Specifying the columns using strings is only "
                "supported for pandas DataFrames"
            ) from e
        if isinstance(key, str):
            columns = [key]
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is not None:
                start = all_columns.index(start)
            if stop is not None:
                # pandas indexing with strings is endpoint included
                stop = all_columns.index(stop) + 1
            else:
                stop = n_channels + 1
            return list(range(n_channels)[slice(start, stop)])
        else:
            columns = list(key)

        return [all_columns.index(col) for col in columns]
    else:
        raise ValueError(
            "No valid specification of the columns. Only a "
            "scalar, list or slice of all integers or all "
            "strings, or boolean mask is allowed"
        )


def _is_empty_channel_selection(column):
    """Check if column selection is empty.

    Both an empty list or all-False boolean array are considered empty.
    """
    if hasattr(column, "dtype") and np.issubdtype(column.dtype, np.bool_):
        return not column.any()
    elif hasattr(column, "__len__"):
        return len(column) == 0
    else:
        return False
