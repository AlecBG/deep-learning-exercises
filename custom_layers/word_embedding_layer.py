# This is word embedding layer using Google's word2vec model.

import os

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

EMBEDDING_DIM = 50  # Can be 50, 100, 200 or 300
GLOVE_DIR = 'glove'


def get_embedding_layer(tokenizer: tf.keras.preprocessing.text.Tokenizer) -> layers.Layer:
    vocab_size = len(tokenizer.word_index)
    embedding_matrix = _load_glove(tokenizer)
    return tf.keras.layers.Embedding(vocab_size + 1,
                                     EMBEDDING_DIM,
                                     embeddings_initializer=embedding_matrix,
                                     input_length=100,
                                     trainable=True,
                                     mask_zero=False)


def _load_glove(tokenizer):
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, EMBEDDING_DIM))
    f = open(os.path.join(GLOVE_DIR, f'glove.6B.{EMBEDDING_DIM}d.txt'), 'r', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        if word not in tokenizer.word_index:
            continue
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_matrix[tokenizer.texts_to_sequences([word])[0][0], :] = coefs
    f.close()

    return tf.keras.initializers.Constant(embedding_matrix)
