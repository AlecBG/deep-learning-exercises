import logging

from tensorflow import keras

from custom_layers.dense_layer import DenseLayer
from custom_layers.attention_layer import SelfAttentionHead
from custom_layers.lstm_cell import LSTMCell
from custom_layers.word_embedding_layer import get_embedding_layer

logger = logging.getLogger('__main__').getChild(__name__)


class RNNModel(keras.Model):
    def __init__(self,
                 tokenizer: keras.preprocessing.text.Tokenizer,
                 n_rnns: int,
                 n_dense: int,
                 n_attention_heads: int,
                 attention_units: int,
                 rnn_units: int,
                 dense_units: int,
                 activation='relu',
                 **kwargs):
        super().__init__(**kwargs)
        self._embedding = get_embedding_layer(tokenizer)

        self._attention_layers = [SelfAttentionHead(attention_units, name=f'attention_head_{i}')
                                  for i in range(n_attention_heads)]
        rnn_cells = [LSTMCell(rnn_units, name=f'rnn_{i}')
                     for i in range(n_rnns)]
        self._rnn_layers = [keras.layers.RNN(rnn_cell, return_sequences=True) for rnn_cell in rnn_cells[:-1]]
        if n_rnns > 0:
            self._rnn_layers.append(keras.layers.RNN(rnn_cells[-1], return_sequences=False))

        self._dense_layers = [DenseLayer(dense_units,
                                         activation=activation,
                                         name=f'dense_{j}') for j in range(n_dense)]
        self._final_layer = DenseLayer(1, activation='sigmoid', name='final_layer')

    def call(self, inputs):
        x = self._embedding(inputs)

        for attention_layer in self._attention_layers:
            x = attention_layer(x)

        for rnn_layer in self._rnn_layers:
            x = rnn_layer(x)

        for dense_layer in self._dense_layers:
            x = dense_layer(x)
        return self._final_layer(x)
