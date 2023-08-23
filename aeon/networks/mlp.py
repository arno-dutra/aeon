# -*- coding: utf-8 -*-
"""Multi Layer Perceptron (MLP) (minus the final output layer)."""

__author__ = ["James-Large", "Withington", "AurumnPegasus", "Arno Dutra"]

from aeon.networks.base import BaseDeepNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class MLPNetwork(BaseDeepNetwork):
    """Establish the network structure for a MLP.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    random_state    : int, default = 0
        seed to any needed random actions
    include_input   : bool, default = True
        whether to include the input layer
    units           : list of int, default = [500, 500, 500]
        number of units in each hidden layer
    dropout_rate    : list of float, default = [0.1, 0.2, 0.2, 0.3]
        dropout rate for each layer

    Notes
    -----
    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    References
    ----------
    .. [1]  Network originally defined in:
    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        random_state=0,
        include_input=True,
        units =[500, 500, 500],
        dropout_rate=[0.1, 0.2, 0.2, 0.3],
    ):
        _check_dl_dependencies(severity="error")
        self.random_state = random_state
        self.include_input = include_input
        self.units = units
        self.dropout_rate = dropout_rate
        super(MLPNetwork, self).__init__()

    def build_network(self, input_shape=None, input_layer=None, **kwargs):
        """Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the input layer

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        from tensorflow import keras

        if self.include_input:
            # flattened because multivariate should be on same axis
            input_layer = keras.layers.Input(input_shape)
            input_layer = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(self.dropout_rate[0])(input_layer)
        layer_1 = keras.layers.Dense(self.units[0], activation="relu")(layer_1)

        layer_2 = keras.layers.Dropout(self.dropout_rate[1])(layer_1)
        layer_2 = keras.layers.Dense(self.units[1], activation="relu")(layer_2)

        layer_3 = keras.layers.Dropout(self.dropout_rate[2])(layer_2)
        layer_3 = keras.layers.Dense(self.units[2], activation="relu")(layer_3)

        output_layer = keras.layers.Dropout(self.dropout_rate[3])(layer_3)

        return input_layer, output_layer
