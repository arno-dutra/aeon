# -*- coding: utf-8 -*-
"""Residual Network Decoder (ResNet) (minus the final output layer)."""

__author__ = ["Arno Dutra", "James Large", "Withington", "nilesh05apr", "hadifawaz1999"]

from aeon.networks.base import BaseDeepNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies


class ResNetDecoderNetwork(BaseDeepNetwork):
    """
    Establish the network structure for decoding a ResNet.
    
    Parameters
    ----------
        n_residual_blocks               : int, default = 3,
            the number of residual blocks of ResNet's model
        n_deconv_per_residual_block       : int, default = 3,
            the number of convolution transpose blocks in each residual block
        n_filters                       : int or list of int, default = [128, 64, 64],
            the number of convolution filters for all the convolution transpose layers in the same
            residual block, if not a list, the same number of filters is used in all
            convolutions of all residual blocks.
        kernel_size                    : int or list of int, default = [8, 5, 3],
            the kernel size of all the convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        strides                         : int or list of int, default = 1,
            the strides of convolution transpose kernels in each of the
            convolution layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        dilation_rate                   : int or list of int, default = 1,
            the dilation rate of the convolution transpose layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        padding                         : str or list of str, default = 'padding',
            the type of padding used in the convolution transpose layers
            in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        activation                      : str or list of str, default = 'relu',
            keras activation used in the convolution transpose layers
            in one residual block, if not
            a list, the same kernel size is used in all convolution transpose layers
        use_bias                        : bool or list of bool, default = True,
            condition on wether or not to use bias values in
            the convolution transpose layers in one residual block, if not
            a list, the same kernel size is used in all convolution layers
        random_state                    : int, optional (default = 0)
            The random seed to use random activities.
    """

    _tags = {"python_dependencies": ["tensorflow"]}

    def __init__(
        self,
        n_residual_blocks=3,
        n_deconv_per_residual_block=3,
        n_filters=None,
        kernel_size=None,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=True,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")
        super(ResNetDecoderNetwork, self).__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.n_residual_blocks = n_residual_blocks
        self.n_deconv_per_residual_block = n_deconv_per_residual_block
        self.random_state = random_state

    def _shortcut_layer(
        self, input_tensor, output_tensor, padding="same", use_bias=True
    ):
        import tensorflow as tf

        n_out_filters = int(output_tensor.shape[-1])

        shortcut_layer = tf.keras.layers.Conv1DTranspose(
            filters=n_out_filters, kernel_size=1, padding=padding, use_bias=use_bias
        )(input_tensor)
        shortcut_layer = tf.keras.layers.BatchNormalization()(shortcut_layer)

        return tf.keras.layers.Add()([output_tensor, shortcut_layer])

    def build_network(self, input_layer, **kwargs):
        """
        Construct a network and return its input and output layers.

        Arguments
        ---------
        input_shape : tuple of shape = (series_length (m), n_dimensions (d))
            The shape of the data fed into the input layer.

        Returns
        -------
        input_layer : keras.layers.Input
            The input layer of the network.
        output_layer : keras.layers.Layer
            The output layer of the network.
        """
        import tensorflow as tf

        self._n_filters_ = [1, 64, 128] if self.n_filters is None else self.n_filters
        self._kernel_size_ = [8, 5, 3] if self.kernel_size is None else self.kernel_size
        self._n_filters_ = self._n_filters_[::-1]
        self._kernel_size_ = self._kernel_size_[::-1]

        if isinstance(self._n_filters_, list):
            self._n_filters = self._n_filters_
        else:
            self._n_filters = [self._n_filters_] * self.n_residual_blocks

        if isinstance(self._kernel_size_, list):
            self._kernel_size = self._kernel_size_
        else:
            self._kernel_size = [self._kernel_size_] * self.n_deconv_per_residual_block

        if isinstance(self.strides, list):
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.n_deconv_per_residual_block

        if isinstance(self.dilation_rate, list):
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.n_deconv_per_residual_block

        if isinstance(self.padding, list):
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.n_deconv_per_residual_block

        if isinstance(self.activation, list):
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.n_deconv_per_residual_block

        if isinstance(self.use_bias, list):
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.n_deconv_per_residual_block

        # input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer

        for d in range(self.n_residual_blocks):
            input_block_tensor = x

            for c in range(self.n_deconv_per_residual_block):
                deconv = tf.keras.layers.Conv1DTranspose(  
                    filters=self._n_filters[d],
                    kernel_size=self._kernel_size[c],
                    strides=self._strides[c],
                    padding=self._padding[c],
                    dilation_rate=self._dilation_rate[c],
                )(x)
                deconv = tf.keras.layers.BatchNormalization()(deconv)

                if c == self.n_deconv_per_residual_block - 1:
                    deconv = self._shortcut_layer(
                        input_tensor=input_block_tensor, output_tensor=deconv
                    )

                deconv = tf.keras.layers.Activation(activation=self._activation[c])(deconv)

                x = deconv

        # gap_layer = tf.keras.layers.GlobalAveragePooling1D()(deconv)

        return input_layer, x

