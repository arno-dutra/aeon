# -*- coding: utf-8 -*-
"""Inception Time Classifier."""
__author__ = ["Arno Dutra", "James-Large", "Withington", "TonyBagnall", "hadifawaz1999"]

from aeon.networks.base import BaseDeepNetwork
from aeon.utils.validation._dependencies import _check_dl_dependencies

_check_dl_dependencies(severity="warning")


class InceptionDecoderNetwork(BaseDeepNetwork):
    """InceptionTime Network.

        depth               : int, default = 6,
            the number of inception modules used
        nb_filters          : int or list of int32, default = 32,
            the number of filters used in one inception module, if not a list,
            the same number of filters is used in all inception modules
        nb_conv_per_layer   : int or list of int, default = 3,
            the number of convolution layers in each inception module, if not a list,
            the same number of convolution layers is used in all inception modules
        kernel_size         : int or list of int, default = 40,
            the head kernel size used for each inception module, if not a list,
            the same is used in all inception modules
        use_max_pooling     : bool or list of bool, default = True,
            conditioning whether or not to use max pooling
            layer in inception modules, if not a list,
            the same is used in all inception modules
        max_pool_size       : int or list of int, default = 3,
            the size of the max pooling layer, if not a list,
            the same is used in all inception modules
        strides             : int or list of int, default = 1,
            the strides of kernels in convolution layers for
            each inception module, if not a list,
            the same is used in all inception modules
        dilation_rate       : int or list of int, default = 1,
            the dilation rate of convolutions in each
            inception module, if not a list,
            the same is used in all inception modules
        padding             : str or list of str, default = 'same',
            the type of padding used for convoltuon for
            each inception module, if not a list,
            the same is used in all inception modules
        activation          : str or list of str, default = 'relu',
            the activation function used in each inception
            module, if not a list,
            the same is used in all inception modules
        use_bias            : bool or list of bool, default = False,
            conditioning whether or not convolutions should
            use bias values in each inception
            module, if not a list,
            the same is used in all inception modules
        use_residual        : bool, default = True,
            condition whether or not to use residual connections
            all over Inception
        use_bottleneck      : bool, default = True,
            condition whether or not to use bottlenecks
            all over Inception
        bottleneck_size     : int, default = 32,
            the bottleneck size in case use_bottleneck = True
        use_custom_filters  : bool, default = False,
            condition on whether or not to use custom filters
            in the first inception module.
        random_state        : int, default = 0,

    Adapted from the implementation from Fawaz et. al

    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/
    inception.py

    and

    https://github.com/MSD-IRIMAS/CF-4-TSC/blob/main/classifiers/H_Inception.py
    for the custom filters

    Network originally defined in:

    @article{IsmailFawaz2019inceptionTime, Title                    = {
    InceptionTime: Finding AlexNet for Time Series Classification}, Author
                    = {Ismail Fawaz, Hassan and Lucas, Benjamin and
                    Forestier, Germain and Pelletier, Charlotte and Schmidt,
                    Daniel F. and Weber, Jonathan and Webb, Geoffrey I. and
                    Idoumghar, Lhassane and Muller, Pierre-Alain and
                    Petitjean, François}, journal                  = {
                    ArXiv}, Year                     = {2019} }

    Custom filters defined in:

    @inproceedings{ismail-fawaz2022hccf,
    author = {Ismail-Fawaz, Ali and Devanne, Maxime and Weber,
    Jonathan and Forestier, Germain},
    title = {Deep Learning For Time Series Classification
    Using New Hand-Crafted Convolution Filters},
    booktitle = {2022 IEEE International Conference on
    Big Data (IEEE BigData 2022)},
    city = {Osaka},
    country = {Japan},
    pages = {972-981},
    url = {doi.org/10.1109/BigData55660.2022.10020496},
    year = {2022},
    organization = {IEEE}
    }
    """

    _tags = {"python_dependencies": "tensorflow"}

    def __init__(
        self,
        nb_filters=32,
        nb_conv_per_layer=3,
        kernel_size=40,
        use_max_pooling=True,
        max_pool_size=3,
        strides=1,
        dilation_rate=1,
        padding="same",
        activation="relu",
        use_bias=False,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        use_custom_filters=False,
        use_gap=True,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")

        if padding != "same":
            raise ValueError("""Only padding="same" is supported for decoder""")
        if strides != 1:
            raise ValueError("""Only strides=1 is supported for decoder""")

        self.nb_filters = nb_filters
        self.nb_conv_per_layer = nb_conv_per_layer
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.use_max_pooling = use_max_pooling
        self.max_pool_size = max_pool_size
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.bottleneck_size = bottleneck_size
        self.use_custom_filters = use_custom_filters
        self.use_gap = use_gap
        self.random_state = random_state

        super(InceptionDecoderNetwork, self).__init__()

    def hybrid_layer(self, input_tensor, input_channels, kernel_sizes=None):  # TODO @arno-dutra : OK
        raise NotImplementedError("No hybrid layer implemented for Inception Decoder")

    def _inception_module(  # TODO @arno-dutra : OK
        self,
        input_tensor,
        nb_filters=32,
        dilation_rate=1,
        padding="same",
        strides=1,
        activation="relu",
        use_bias=False,
        kernel_size=40,
        nb_conv_per_layer=3,
        use_max_pooling=True,
        max_pool_size=3,
        use_custom_filters=False,
    ):
        import tensorflow as tf

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1DTranspose(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding=padding,  # = "same"
                activation="linear",
                use_bias=use_bias,
            )(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [kernel_size // (2**i) for i in range(nb_conv_per_layer)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.Conv1DTranspose(
                    filters=nb_filters,
                    kernel_size=kernel_size_s[i],
                    strides=strides,  # = 1
                    dilation_rate=dilation_rate,
                    padding=padding,  # = "same"
                    activation="linear",
                    use_bias=use_bias,
                )(input_inception)
            )

        if use_max_pooling:  # Same code as in the encoder because we want to make a bottleneck
            max_pool_1 = tf.keras.layers.MaxPool1D(
                pool_size=max_pool_size, strides=strides, padding=padding
            )(input_tensor)

            conv_max_pool = tf.keras.layers.Conv1D(
                filters=nb_filters,
                kernel_size=1,
                padding=padding,  # = "same"
                activation="linear",
                use_bias=use_bias,
            )(max_pool_1)

            conv_list.append(conv_max_pool)

        if use_custom_filters:  # NOT IMPLEMENTED
            hybrid_layer = self.hybrid_layer(
                input_tensor=input_tensor, input_channels=int(input_tensor.shape[-1])
            )
            conv_list.append(hybrid_layer)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x

    def _shortcut_layer(self, input_tensor, output_tensor, padding="same", use_bias=False):  # TODO @arno-dutra : OK
        import tensorflow as tf

        n_out_filters = int(output_tensor.shape[-1])

        shortcut_y = tf.keras.layers.Conv1DTranspose(
            filters=n_out_filters,
            kernel_size=1,
            padding=padding,  # = "same"
            use_bias=use_bias,
        )(input_tensor)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        x = tf.keras.layers.Add()([shortcut_y, output_tensor])
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def build_network(self, input_layer, **kwargs):
        """
        Construct a network and return its input and output layers.

        input_layer : a keras layer
            the input layer of the network

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        # not sure of the whole padding thing

        import tensorflow as tf

        if isinstance(self.nb_filters, list):
            self._nb_filters = self.nb_filters
        else:
            self._nb_filters = [self.nb_filters] * self.depth

        if isinstance(self.kernel_size, list):
            self._kernel_size = self.kernel_size
        else:
            self._kernel_size = [self.kernel_size] * self.depth

        if isinstance(self.nb_conv_per_layer, list):
            self._nb_conv_per_layer = self.nb_conv_per_layer
        else:
            self._nb_conv_per_layer = [self.nb_conv_per_layer] * self.depth

        if isinstance(self.strides, list):
            self._strides = self.strides
        else:
            self._strides = [self.strides] * self.depth

        if isinstance(self.dilation_rate, list):
            self._dilation_rate = self.dilation_rate
        else:
            self._dilation_rate = [self.dilation_rate] * self.depth

        if isinstance(self.padding, list):
            self._padding = self.padding
        else:
            self._padding = [self.padding] * self.depth

        if isinstance(self.activation, list):
            self._activation = self.activation
        else:
            self._activation = [self.activation] * self.depth

        if isinstance(self.use_max_pooling, list):
            self._use_max_pooling = self.use_max_pooling
        else:
            self._use_max_pooling = [self.use_max_pooling] * self.depth

        if isinstance(self.max_pool_size, list):
            self._max_pool_size = self.max_pool_size
        else:
            self._max_pool_size = [self.max_pool_size] * self.depth

        if isinstance(self.use_bias, list):
            self._use_bias = self.use_bias
        else:
            self._use_bias = [self.use_bias] * self.depth

        # input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        _use_custom_filters = self.use_custom_filters

        for d in range(self.depth):
            if d > 0:
                _use_custom_filters = False

            x = self._inception_module(
                x,
                nb_filters=self._nb_filters[d],
                dilation_rate=self._dilation_rate[d],
                kernel_size=self._kernel_size[d],
                padding=self._padding[d],
                strides=self._strides[d],
                activation=self._activation[d],
                use_bias=self._use_bias[d],
                use_max_pooling=self._use_max_pooling[d],
                max_pool_size=self._max_pool_size[d],
                nb_conv_per_layer=self._nb_conv_per_layer[d],
                use_custom_filters=_use_custom_filters,
            )

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x, padding=self._padding[d])
                input_res = x

        if self.use_gap:
            x = tf.transpose(x, [0, 2, 1])
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.expand_dims(x, -1)

        return input_layer, x
