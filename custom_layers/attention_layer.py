import logging
from abc import ABC

import tensorflow as tf

from custom_layers.dense_layer import DenseLayer

LARGE_CONSTANT = 10**9


class AbstractAttentionLayer(tf.keras.layers.Layer, ABC):
    def __init__(self,
                 units: int,
                 is_causal: bool,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units

        self._keys = DenseLayer(units)
        self._queries = DenseLayer(units)
        self._values = DenseLayer(units)
        self._metric = tf.Variable(
            initial_value=tf.keras.initializers.Identity()(shape=(units, units), dtype='float32'), trainable=True,
        )

        self._scaling_factor = tf.sqrt(float(units))
        self._is_causal = is_causal

    def compute(self, inputs_1, inputs_2):
        logging.debug(f'inputs_1: {inputs_1}')
        logging.debug(f'inputs_2: {inputs_2}')

        queries = self._queries(inputs_1)
        keys = self._keys(inputs_2)
        values = self._values(inputs_2)

        logging.debug(f'queries: {queries}  keys: {keys}  values: {values}')

        queries = tf.tensordot(queries, self._metric, axes=([2], [0]))
        scores = tf.einsum('biu,bju->bij', queries, keys) / self._scaling_factor

        logging.debug(f'queries: {queries}  scores: {scores}')

        if self._is_causal:
            causal_mask = tf.linalg.band_part(tf.ones_like(scores), -1, 0)
            scores = scores - LARGE_CONSTANT * (1.0 - causal_mask)

        # Shape of alpha is (batch_size, sequence_length, sequence_length)
        alpha = tf.nn.softmax(scores)
        output = tf.einsum('bij,bju->biu', alpha, values)

        logging.debug(f'alpha: {alpha}  output: {output}')
        return output


class SelfAttentionHead(AbstractAttentionLayer):

    def __init__(self,
                 units: int,
                 is_causal: bool = False,
                 **kwargs):
        super().__init__(units, is_causal, **kwargs)

    def call(self, inputs):
        return self.compute(inputs, inputs)


class CrossAttentionHead(AbstractAttentionLayer):

    def __init__(self,
                 units: int,
                 is_causal: bool = False,
                 **kwargs):
        super().__init__(units, is_causal, **kwargs)

    def call(self, inputs):
        inputs_1, inputs_2 = inputs
        return self.compute(inputs_1, inputs_2)