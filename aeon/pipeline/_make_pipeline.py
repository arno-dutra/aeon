# -*- coding: utf-8 -*-
"""Pipeline making utility."""
# copyright: aeon developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]


def make_pipeline(*steps):
    """Create a pipeline from estimators of any type.

    Parameters
    ----------
    steps : tuple of aeon estimators
        in same order as used for pipeline construction

    Returns
    -------
    pipe : aeon pipeline containing steps, in order
        always a descendant of BaseObject, precise object determined by scitype
        equivalent to result of step[0] * step[1] * ... * step[-1]

    Examples
    --------
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()

    Example 1: forecaster pipeline
    >>> from aeon.datasets import load_airline
    >>> from aeon.forecasting.trend import PolynomialTrendForecaster
    >>> from aeon.pipeline import make_pipeline
    >>> from aeon.transformations.series.exponent import ExponentTransformer
    >>> y = load_airline()
    >>> pipe = make_pipeline(ExponentTransformer(), PolynomialTrendForecaster())
    >>> type(pipe).__name__
    'TransformedTargetForecaster'

    Example 2: classifier pipeline
    >>> from aeon.classification.feature_based import Catch22Classifier
    >>> from aeon.pipeline import make_pipeline
    >>> from aeon.transformations.series.exponent import ExponentTransformer
    >>> pipe = make_pipeline(ExponentTransformer(), Catch22Classifier())
    >>> type(pipe).__name__
    'ClassifierPipeline'

    Example 3: transformer pipeline
    >>> from aeon.pipeline import make_pipeline
    >>> from aeon.transformations.series.exponent import ExponentTransformer
    >>> pipe = make_pipeline(ExponentTransformer(), ExponentTransformer())
    >>> type(pipe).__name__
    'TransformerPipeline'
    """
    pipe = steps[0]
    for i in range(1, len(steps)):
        pipe = pipe * steps[i]

    return pipe
