# -*- coding: utf-8 -*-
"""Tests the VAR model."""
__author__ = ["thayeylolu"]
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from aeon.forecasting.base import ForecastingHorizon
from aeon.forecasting.model_selection import temporal_train_test_split

#
from aeon.forecasting.var import VAR
from aeon.utils.validation._dependencies import _check_soft_dependencies

index = pd.date_range(start="2005", end="2006-12", freq="M")
df = pd.DataFrame(
    np.random.randint(0, 100, size=(23, 2)),
    columns=list("AB"),
    index=pd.PeriodIndex(index),
)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_VAR_against_statsmodels():
    """Compares aeon's and Statsmodel's VAR."""
    from statsmodels.tsa.api import VAR as _VAR

    train, test = temporal_train_test_split(df)
    aeon_model = VAR()
    fh = ForecastingHorizon([1, 3, 4, 5, 7, 9])
    aeon_model.fit(train)
    y_pred = aeon_model.predict(fh=fh)

    stats = _VAR(train)
    stats_fit = stats.fit()
    fh_int = fh.to_relative(train.index[-1])
    lagged = stats_fit.k_ar
    y_pred_stats = stats_fit.forecast(train.values[-lagged:], steps=fh_int[-1])
    new_arr = []
    for i in fh_int:
        new_arr.append(y_pred_stats[i - 1])
    assert_allclose(y_pred, new_arr)
