from typing import Optional, Tuple
import logging

import tensorflow as tf
from tensorflow.keras import layers

from custom_layers.dense_layer import DenseLayer

BATCH_SIZE = 32


class LSTMCell(layers.AbstractRNNCell):
    def __init__(self,
                 units: int,
                 activation: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._activation = tf.keras.activations.get(activation)
        self.units = units
        self._state_dim = units

        self._forgetter = DenseLayer(self._state_dim, activation='sigmoid')
        self._updater_gate = DenseLayer(self._state_dim, activation='sigmoid')
        self._updater = DenseLayer(self._state_dim, activation='tanh')
        self._output_gate = DenseLayer(self._state_dim, activation='sigmoid')

    def call(self, inputs: tf.Tensor, states: Tuple[tf.Tensor, tf.Tensor])\
            -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        bottom_state, top_state = states

        logging.debug(f'bottom_state: {bottom_state}  top_state: {top_state}  inputs: {inputs}')

        concat_input = tf.concat([bottom_state, inputs], axis=1)

        logging.debug(f'concat_input: {concat_input}')

        forget_vector = self._forgetter(concat_input)

        top_state = tf.math.multiply(top_state, forget_vector)

        update_gate_vector = self._updater_gate(concat_input)

        pre_update_vector = self._updater(concat_input)

        update_vector = tf.math.multiply(update_gate_vector, pre_update_vector)

        new_top_state = tf.math.add(top_state, update_vector)

        new_bottom_state_pregate = tf.math.tanh(new_top_state)
        new_bottom_state_gate = self._output_gate(concat_input)
        output = tf.math.multiply(new_bottom_state_pregate, new_bottom_state_gate)

        return output, (output, new_top_state)

    def compute_output_shape(self, batch_input_shape: tf.Tensor) -> tf.TensorShape:
        return tf.TensorShape([batch_input_shape[: -1]] + [self.units])

    def get_config(self) -> dict:
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": tf.keras.activations.serialize(self._activation)}

    @property
    def output_size(self) -> int:
        return self.units

    @property
    def state_size(self) -> Tuple[int, int]:
        return self._state_dim, self._state_dim
