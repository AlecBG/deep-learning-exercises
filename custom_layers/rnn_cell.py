from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from custom_layers.dense_layer import DenseLayer

BATCH_SIZE = 32


class RNNCell(layers.AbstractRNNCell):

    def __init__(self,
                 units: int,
                 state_dim: int,
                 activation: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._activation = tf.keras.activations.get(activation)
        self._state_dim = state_dim

        self._stateless_output = DenseLayer(self.units, activation=None)
        self._stateful_output = DenseLayer(self.units, no_bias=True, activation=None)
        self._state_updater = DenseLayer(self._state_dim, activation=None)

    def call(self, inputs: tf.Tensor, states: Tuple[tf.Tensor]) -> Tuple[tf.Tensor, Tuple[tf.Tensor]]:
        states = states[0]
        stateful_output = self._stateful_output(states)

        non_stateful_output = self._stateless_output(input)

        state_update = self._state_updater(inputs)
        new_states = tf.math.add(states, state_update)
        new_states = tf.reshape(new_states, [BATCH_SIZE, self._state_dim])

        return self._activation(stateful_output + non_stateful_output), (tf.math.tanh(new_states))

    def compute_output_shape(self,
                             batch_input_shape: tf.Tensor):
        return tf.TensorShape([batch_input_shape[: -1]] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units, "state_dim": self._state_dim,
                "activation": tf.keras.activations.serialize(self._activation)}

    @property
    def output_size(self):
        return self.units

    @property
    def state_size(self) -> Tuple[int]:
        return self._state_dim,
