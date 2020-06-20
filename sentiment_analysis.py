import logging
import os
import pickle

import tensorflow as tf

from data_pipelines.imdb_data_processing import get_train_valid_dataset, get_test_dataset
from custom_models.rnn_model import RNNModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

logging.info("Getting data")
train_ds, valid_ds = get_train_valid_dataset()
test_ds = get_test_dataset()

logging.info("Loading tokenizer")
with open('data_pipelines/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

logging.info("Initializing model.")
model = RNNModel(tokenizer,
                 n_rnns=1,
                 n_dense=2,
                 n_attention_heads=4,
                 attention_units=20,
                 rnn_units=128,
                 dense_units=50)

logging.info('building model')
model.build(tf.TensorShape([None, 100]))
model.summary()

# model.load_weights('imdb_simple.h5')


def get_run_log_dir():
    import time
    root_logdir = os.path.join(os.curdir, "my_logs")
    root_logdir = os.path.join(root_logdir, "sentiment_analysis")
    run_id = time.strftime("run_%Y_%m_%d-%H_%m_%S")
    return os.path.join(root_logdir, run_id)


run_log_dir = get_run_log_dir()

logging.info("Compiling model")
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('imdb_simple.h5', save_weights_only=True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_log_dir, update_freq='batch')


logging.info("Fitting model")
history = model.fit(train_ds,
                    epochs=5,
                    validation_data=valid_ds,
                    callbacks=[tensorboard_cb, checkpoint_cb])
model.evaluate(test_ds)
