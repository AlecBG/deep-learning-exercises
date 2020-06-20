from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from custom_layers.lstm_cell import LSTMCell

BATCH_SIZE = 32


class BidirectionalLSTMCell(layers.AbstractRNNCell):
    def __init__(self,
                 units: int,
                 activation: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._left_lstm = LSTMCell(units, activation, **kwargs)
        self._right_lstm = LSTMCell(units, activation, **kwargs)

    def call(self, inputs: tf.Tensor, states: Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]])\
            -> Tuple[tf.Tensor, Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]]:
        left_states, right_states = states

        left_output, new_left_states = self._left_lstm.call(inputs, left_states)

        x = tf.reverse(inputs, 1)
        right_output, new_right_states = self._right_lstm.call(x, right_states)

        output = left_output + right_output

        return output, (new_left_states, new_right_states)

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
    def state_size(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (self._state_dim, self._state_dim), (self._state_dim, self._state_dim)
