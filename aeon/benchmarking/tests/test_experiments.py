# -*- coding: utf-8 -*-
"""Functions to test the functions in experiments.py."""

from aeon.benchmarking.experiments import (
    run_classification_experiment,
    run_clustering_experiment,
)
from aeon.classification import DummyClassifier
from aeon.clustering.k_means import TimeSeriesKMeans
from aeon.datasets import load_unit_test


def test_run_clustering_experiment(tmp_path):
    """Test running and saving results for clustering.

    Currently it just checks the files have been created, then deletes them.
    """
    dataset = "UnitTest"
    train_X, train_Y = load_unit_test("TRAIN")
    test_X, test_Y = load_unit_test("TEST")
    run_clustering_experiment(
        train_X,
        TimeSeriesKMeans(n_clusters=2),
        results_path=tmp_path,
        trainY=train_Y,
        testX=test_X,
        testY=test_Y,
        cls_name="kmeans",
        dataset_name=dataset,
        resample_id=0,
    )
    test_path = tmp_path.joinpath(f"kmeans/Predictions/{dataset}/testResample0.csv")
    train_path = tmp_path.joinpath(f"kmeans/Predictions/{dataset}/trainResample0.csv")
    assert test_path.is_file()
    assert train_path.is_file()
    # remove files
    test_path.unlink()
    train_path.unlink()


def test_run_classification_experiment(tmp_path):
    """Test running and saving results for classifiers.

    Currently it just checks the files have been created, then deletes them.
    """
    dataset = "UnitTest"
    train_X, train_Y = load_unit_test("TRAIN")
    test_X, test_Y = load_unit_test("TEST")
    run_classification_experiment(
        train_X,
        train_Y,
        test_X,
        test_Y,
        DummyClassifier(),
        str(tmp_path),
        cls_name="DummyClassifier",
        dataset="UnitTest",
        resample_id=0,
        train_file=True,
    )
    test_path = tmp_path.joinpath(
        f"DummyClassifier/Predictions/{dataset}/testResample0.csv"
    )
    train_path = tmp_path.joinpath(
        f"DummyClassifier/Predictions/{dataset}/trainResample0.csv"
    )
    assert test_path.is_file()
    assert train_path.is_file()
    # remove files
    test_path.unlink()
    train_path.unlink()
