# -*- coding: utf-8 -*-
"""Test function of DummyRegressor."""

from aeon.datasets import load_unit_test
from aeon.regression import DummyRegressor


def test_dummy_regressor():
    """Test function for DummyRegressor."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy3D")
    X_test, _ = load_unit_test(split="test", return_type="numpy3D")
    dummy = DummyRegressor()
    dummy.fit(X_train, y_train)
    pred = dummy.predict(X_test)
    assert (pred == 1.5).all()
