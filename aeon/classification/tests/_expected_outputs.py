# -*- coding: utf-8 -*-
"""Dictionaries of expected outputs of classifier predict runs."""

import numpy as np

# predict_proba results on unit test data
unit_test_proba = dict()

# predict_proba results on basic motions data
basic_motions_proba = dict()


unit_test_proba["BOSSEnsemble"] = np.array(
    [
        [0.5, 0.5],
        [0.75, 0.25],
        [0.25, 0.75],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["ContractableBOSS"] = np.array(
    [
        [0.30211169, 0.69788831],
        [0.88932421, 0.11067579],
        [0.07261438, 0.92738562],
        [1.0, 0.0],
        [0.83795958, 0.16204042],
        [1.0, 0.0],
        [0.18329017, 0.81670983],
        [0.07261438, 0.92738562],
        [0.41278748, 0.58721252],
        [0.88932421, 0.11067579],
    ]
)
unit_test_proba["TemporalDictionaryEnsemble"] = np.array(
    [
        [0.0, 1.0],
        [0.4924, 0.5076],
        [0.0, 1.0],
        [0.9043, 0.0957],
        [0.8016, 0.1984],
        [1.0, 0.0],
        [0.706, 0.294],
        [0.0, 1.0],
        [0.8016, 0.1984],
        [1.0, 0.0],
    ]
)
unit_test_proba["WEASEL"] = np.array(
    [
        [0.20366595, 0.79633405],
        [0.97761497, 0.02238503],
        [0.05127821, 0.94872179],
        [0.81435354, 0.18564646],
        [0.91971316, 0.08028684],
        [0.97877426, 0.02122574],
        [0.16694218, 0.83305782],
        [0.04834253, 0.95165747],
        [0.93156332, 0.06843668],
        [0.97714351, 0.02285649],
    ]
)
unit_test_proba["WEASEL_V2"] = np.array(
    [
        [0.0023, 0.9977],
        [0.986, 0.014],
        [0.0029, 0.9971],
        [0.9974, 0.0026],
        [0.9798, 0.0202],
        [0.9953, 0.0047],
        [0.9602, 0.0398],
        [0.0238, 0.9762],
        [0.9968, 0.0032],
        [0.9986, 0.0014],
    ]
)
unit_test_proba["ElasticEnsemble"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.33333333, 0.66666667],
        [1.0, 0.0],
        [0.66666667, 0.33333333],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["ShapeDTW"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["Catch22Classifier"] = np.array(
    [
        [0.3, 0.7],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.5, 0.5],
        [0.9, 0.1],
        [0.4, 0.6],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.9, 0.1],
    ]
)
unit_test_proba["FreshPRINCEClassifier"] = np.array(
    [
        [0.2, 0.8],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["MatrixProfileClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
)
unit_test_proba["RandomIntervalClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.2, 0.8],
        [0.8, 0.2],
        [1.0, 0.0],
    ]
)
unit_test_proba["SignatureClassifier"] = np.array(
    [
        [0.1, 0.9],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.8, 0.2],
        [0.0, 1.0],
        [0.8, 0.2],
        [1.0, 0.0],
    ]
)
unit_test_proba["SummaryClassifier"] = np.array(
    [
        [0.0, 1.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.9, 0.1],
        [0.9, 0.1],
        [1.0, 0.0],
        [0.8, 0.2],
        [0.6, 0.4],
        [0.9, 0.1],
        [1.0, 0.0],
    ]
)
unit_test_proba["HIVECOTEV1"] = np.array(
    [
        [0.0, 1.0],
        [0.5697, 0.4303],
        [0.0, 1.0],
        [0.8351, 0.1649],
        [0.8881, 0.1119],
        [0.975, 0.025],
        [0.6884, 0.3116],
        [0.0, 1.0],
        [0.7584, 0.2416],
        [0.727, 0.273],
    ]
)
unit_test_proba["HIVECOTEV2"] = np.array(
    [
        [0.0, 1.0],
        [0.4563, 0.5437],
        [0.0379, 0.9621],
        [1.0, 0.0],
        [0.719, 0.281],
        [1.0, 0.0],
        [0.8477, 0.1523],
        [0.0379, 0.9621],
        [0.6902, 0.3098],
        [1.0, 0.0],
    ]
)
unit_test_proba["CanonicalIntervalForest"] = np.array(
    [
        [0.41, 0.59],
        [0.7333, 0.2667],
        [0.1833, 0.8167],
        [0.7667, 0.2333],
        [0.5, 0.5],
        [0.76, 0.24],
        [0.8, 0.2],
        [0.2833, 0.7167],
        [0.86, 0.14],
        [0.7, 0.3],
    ]
)
unit_test_proba["DrCIF"] = np.array(
    [
        [0.0, 1.0],
        [0.8, 0.2],
        [0.2, 0.8],
        [1.0, 0.0],
        [0.7, 0.3],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.3, 0.7],
        [0.8, 0.2],
        [1.0, 0.0],
    ]
)
unit_test_proba["RandomIntervalSpectralEnsemble"] = np.array(
    [
        [0.1, 0.9],
        [0.8, 0.2],
        [0.0, 1.0],
        [0.7, 0.3],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.6, 0.4],
        [0.0, 1.0],
        [0.7, 0.3],
        [0.9, 0.1],
    ]
)
unit_test_proba["SupervisedTimeSeriesForest"] = np.array(
    [
        [0.0, 1.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.1, 0.9],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["TimeSeriesForestClassifier"] = np.array(
    [
        [0.1, 0.9],
        [0.7, 0.3],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.8, 0.2],
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [0.8, 0.2],
        [0.9, 0.1],
    ]
)
unit_test_proba["Arsenal"] = np.array(
    [
        [-0.0, 1.0],
        [1.0, -0.0],
        [-0.0, 1.0],
        [1.0, -0.0],
        [0.9236, 0.0764],
        [1.0, -0.0],
        [0.4506, 0.5494],
        [-0.0, 1.0],
        [1.0, -0.0],
        [1.0, -0.0],
    ]
)
unit_test_proba["RocketClassifier"] = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
unit_test_proba["ShapeletTransformClassifier"] = np.array(
    [
        [0.0, 1.0],
        [0.6, 0.4],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)

basic_motions_proba["ColumnEnsembleClassifier"] = np.array(
    [
        [0.0, 0.08247423, 0.25, 0.66752577],
        [0.25, 0.08247423, 0.66752577, 0.0],
        [0.0, 0.08247423, 0.66752577, 0.25],
        [0.5, 0.08247423, 0.41752577, 0.0],
        [0.0, 0.08247423, 0.5, 0.41752577],
        [0.0, 0.08247423, 0.5, 0.41752577],
        [0.25, 0.33247423, 0.41752577, 0.0],
        [0.0, 0.08247423, 0.91752577, 0.0],
        [0.0, 0.58247423, 0.41752577, 0.0],
        [0.0, 0.33247423, 0.41752577, 0.25],
    ]
)
basic_motions_proba["MUSE"] = np.array(
    [
        [3.67057592e-05, 1.12259557e-03, 6.67246229e-04, 9.98173452e-01],
        [9.93229455e-01, 1.92232324e-04, 2.56248688e-03, 4.01582536e-03],
        [1.73244986e-04, 1.87190456e-04, 9.97716736e-01, 1.92282859e-03],
        [2.59659365e-03, 9.97076299e-01, 7.09934439e-05, 2.56113573e-04],
        [3.19356238e-05, 6.60136189e-03, 2.33211388e-03, 9.91034589e-01],
        [8.50903584e-05, 5.96209341e-04, 3.18223960e-02, 9.67496304e-01],
        [9.81362825e-01, 1.39771640e-03, 1.18616691e-02, 5.37778988e-03],
        [1.55494301e-03, 2.12773041e-04, 9.96621925e-01, 1.61035863e-03],
        [9.59903116e-03, 9.90085747e-01, 7.30870932e-05, 2.42134656e-04],
        [6.40967171e-04, 9.99163067e-01, 5.53240474e-05, 1.40642181e-04],
    ]
)
basic_motions_proba["TemporalDictionaryEnsemble"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.6261, 0.3739, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.7478, 0.0, 0.0, 0.2522],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.7478, 0.2522, 0.0],
        [0.0, 0.7478, 0.2522, 0.0],
    ]
)
basic_motions_proba["Catch22Classifier"] = np.array(
    [
        [0.1, 0.0, 0.1, 0.8],
        [0.3, 0.4, 0.2, 0.1],
        [0.0, 0.2, 0.6, 0.2],
        [0.0, 0.8, 0.1, 0.1],
        [0.1, 0.0, 0.0, 0.9],
        [0.2, 0.0, 0.1, 0.7],
        [0.4, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.6, 0.2],
        [0.1, 0.9, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
basic_motions_proba["FreshPRINCEClassifier"] = np.array(
    [
        [0.0, 0.0, 0.1, 0.9],
        [0.9, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.8, 0.2],
        [0.0, 0.9, 0.0, 0.1],
        [0.0, 0.0, 0.2, 0.8],
        [0.0, 0.0, 0.2, 0.8],
        [0.5, 0.3, 0.0, 0.2],
        [0.0, 0.0, 0.7, 0.3],
        [0.0, 1.0, 0.0, 0.0],
        [0.1, 0.8, 0.0, 0.1],
    ]
)
basic_motions_proba["RandomIntervalClassifier"] = np.array(
    [
        [0.0, 0.0, 0.2, 0.8],
        [0.3, 0.1, 0.1, 0.5],
        [0.0, 0.0, 0.8, 0.2],
        [0.2, 0.7, 0.0, 0.1],
        [0.0, 0.1, 0.4, 0.5],
        [0.0, 0.0, 0.4, 0.6],
        [0.2, 0.3, 0.1, 0.4],
        [0.0, 0.1, 0.9, 0.0],
        [0.1, 0.8, 0.0, 0.1],
        [0.1, 0.7, 0.0, 0.2],
    ]
)
basic_motions_proba["SignatureClassifier"] = np.array(
    [
        [0.0, 0.0, 0.5, 0.5],
        [0.4, 0.0, 0.3, 0.3],
        [0.0, 0.0, 0.9, 0.1],
        [0.2, 0.3, 0.1, 0.4],
        [0.0, 0.0, 0.4, 0.6],
        [0.0, 0.0, 0.7, 0.3],
        [0.1, 0.0, 0.6, 0.3],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.7, 0.1, 0.2],
        [0.2, 0.3, 0.1, 0.4],
    ]
)
basic_motions_proba["SummaryClassifier"] = np.array(
    [
        [0.0, 0.0, 0.3, 0.7],
        [0.5, 0.2, 0.1, 0.2],
        [0.0, 0.0, 0.8, 0.2],
        [0.0, 1.0, 0.0, 0.0],
        [0.1, 0.1, 0.2, 0.6],
        [0.0, 0.0, 0.3, 0.7],
        [0.5, 0.2, 0.1, 0.2],
        [0.0, 0.0, 0.8, 0.2],
        [0.1, 0.9, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
    ]
)
basic_motions_proba["HIVECOTEV2"] = np.array(
    [
        [0.0, 0.0222, 0.0222, 0.9557],
        [0.8065, 0.0701, 0.0, 0.1235],
        [0.0222, 0.0, 0.858, 0.1198],
        [0.0701, 0.2803, 0.3774, 0.2722],
        [0.0222, 0.0, 0.0701, 0.9078],
        [0.0222, 0.0, 0.1144, 0.8634],
        [0.7843, 0.1845, 0.0, 0.0312],
        [0.0222, 0.0, 0.8483, 0.1295],
        [0.0922, 0.7843, 0.0922, 0.0312],
        [0.0, 0.9466, 0.0222, 0.0312],
    ]
)
basic_motions_proba["CanonicalIntervalForest"] = np.array(
    [
        [0.0, 0.0, 0.3, 0.7],
        [0.6, 0.2, 0.2, 0.0],
        [0.0, 0.1, 0.6, 0.3],
        [0.1, 0.5, 0.0, 0.4],
        [0.0, 0.0, 0.3, 0.7],
        [0.0, 0.0, 0.3, 0.7],
        [0.6, 0.2, 0.0, 0.2],
        [0.2, 0.0, 0.6, 0.2],
        [0.0, 0.5, 0.1, 0.4],
        [0.3, 0.7, 0.0, 0.0],
    ]
)
basic_motions_proba["DrCIF"] = np.array(
    [
        [0.1, 0.1, 0.3, 0.5],
        [0.8, 0.2, 0.0, 0.0],
        [0.0, 0.1, 0.7, 0.2],
        [0.3, 0.6, 0.0, 0.1],
        [0.2, 0.0, 0.2, 0.6],
        [0.0, 0.1, 0.4, 0.5],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.2],
        [0.3, 0.7, 0.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
    ]
)
basic_motions_proba["Arsenal"] = np.array(
    [
        [-0.0, 0.158, -0.0, 0.842],
        [1.0, -0.0, -0.0, -0.0],
        [0.6394, 0.3606, -0.0, -0.0],
        [-0.0, -0.0, 0.586, 0.414],
        [-0.0, -0.0, 0.2254, 0.7746],
        [-0.0, -0.0, 0.256, 0.744],
        [0.7771, 0.2229, -0.0, -0.0],
        [0.256, 0.2229, 0.3631, 0.158],
        [-0.0, 0.842, 0.158, -0.0],
        [-0.0, 1.0, -0.0, -0.0],
    ]
)
basic_motions_proba["RocketClassifier"] = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
basic_motions_proba["ShapeletTransformClassifier"] = np.array(
    [
        [0.0, 0.0, 0.2, 0.8],
        [0.2, 0.8, 0.0, 0.0],
        [0.0, 0.2, 0.6, 0.2],
        [0.0, 0.8, 0.2, 0.0],
        [0.0, 0.0, 0.2, 0.8],
        [0.0, 0.0, 0.2, 0.8],
        [0.2, 0.6, 0.0, 0.2],
        [0.0, 0.2, 0.8, 0.0],
        [0.2, 0.6, 0.0, 0.2],
        [0.2, 0.6, 0.0, 0.2],
    ]
)
unit_test_proba["TEASER"] = np.array(
    [
        [0.0, 1.0],
        [0.5, 0.5],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.7, 0.3],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.1, 0.9],
        [0.9, 0.1],
        [1.0, 0.0],
    ]
)
unit_test_proba["ProbabilityThresholdEarlyClassifier"] = np.array(
    [
        [0.0, 1.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ]
)
