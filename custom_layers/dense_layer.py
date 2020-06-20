from typing import Optional
import logging

import tensorflow as tf
from tensorflow.keras import layers


class DenseLayer(layers.Layer):
    def __init__(self, units: int,
                 activation: Optional[str] = None,
                 no_bias: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self._units = units
        self._activation = tf.keras.activations.get(activation)

        self._kernel = None
        self._bias = None

        self._no_bias = no_bias

    def build(self, batch_input_shape):
        logging.debug("Here we are in the dense_layer. batch_input_shape: " + str(batch_input_shape))
        self._kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self._units],
            initializer='glorot_normal')
        self._bias = self.add_weight(
            name='bias', shape=[self._units],
            initializer='zeros')
        super().build(batch_input_shape)

    def call(self, x):
        bias = tf.zeros(tf.shape(x)) if self._no_bias else self._bias
        return self._activation(x @ self._kernel + bias)

    def compute_output_shape(self,
                             batch_input_shape: tf.Tensor):
        return tf.TensorShape([batch_input_shape[: -1]] + [self._units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self._units,
                "activation": tf.keras.activations.serialize(self._activation)}
