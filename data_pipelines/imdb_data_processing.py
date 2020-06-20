import logging
from pathlib import Path
from typing import Tuple
import pickle

import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
OOV_TOKEN = '<unk>'
COMPRESSION = 'GZIP'

RAW_DATA_DIR = Path('C:/Users/alec.barns-graham/tensorflow_datasets/imdb_reviews/plain_text/0.1.0')
PROCESSED_DATA_DIR = Path('C:/Users/alec.barns-graham/Documents/deep-learning-exercises/datasets/imdb/')
TRAINING_PATH = str(PROCESSED_DATA_DIR / 'training_set/training_text.tfrecord')
TEST_PATH = str(PROCESSED_DATA_DIR / 'test_set/test_text.tfrecord')

LABEL = 'label'
TOKEN = 'token'
TEXT = 'text'
PROC_FEATURE_DESCRIPTION = {LABEL: tf.io.FixedLenFeature([1], tf.int64),
                            TOKEN: tf.io.FixedLenFeature([100], tf.int64)}
UNPROC_FEATURE_DESCRIPTION = {LABEL: tf.io.FixedLenFeature([], tf.int64),
                              TEXT: tf.io.VarLenFeature(tf.string)}

SEQUENCE_LENGTH = 100

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')


def get_class_names():
    with open(RAW_DATA_DIR / 'label.labels.txt', 'r') as file:
        file_contents = file.read()
        labels = file_contents.split('\n')
        return labels[: 2]


CLASS_NAMES = get_class_names()


def create_tokenizer_and_datasets() -> ():

    list_train_paths = list(RAW_DATA_DIR.glob('*train*'))
    list_test_paths = list(RAW_DATA_DIR.glob('*test*'))
    list_train_paths = [str(x) for x in list_train_paths]
    list_test_paths = [str(x) for x in list_test_paths]

    train_dataset = tf.data.TFRecordDataset(filenames=list_train_paths)
    test_dataset = tf.data.TFRecordDataset(filenames=list_test_paths)
    train_dataset = train_dataset.map(lambda x: tf.io.parse_single_example(x, UNPROC_FEATURE_DESCRIPTION))
    test_dataset = test_dataset.map(lambda x: tf.io.parse_single_example(x, UNPROC_FEATURE_DESCRIPTION))
    train_dataset = train_dataset.map(lambda x: _process_text(x))
    test_dataset = test_dataset.map(lambda x: _process_text(x))
    logging.info('Datasets opened.')

    tokenizer = _create_tokenizer(train_dataset, test_dataset)
    _write_tfrecords(train_dataset, test_dataset, tokenizer)

    logging.info('Writing the tokenizer')
    with open('tokenizer.pkl', 'wb') as file_name:
        pickle.dump(tokenizer, file_name)
    return tokenizer


def _process_text(example):
    text, label = example[TEXT].values, example[LABEL]
    # Process text
    text = tf.strings.regex_replace(text, b'<br />', b'')
    text = tf.strings.regex_replace(text, b'([^a-zA-Z0-9 ])', b'')
    text = tf.strings.regex_replace(text, b' [ ]+', b' ')
    return text, label


def _create_tokenizer(train_dataset, test_dataset):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=OOV_TOKEN)
    all_text = []
    for count, example in enumerate(train_dataset):
        count += 1
        if count % 2000 == 0:
            logging.info(f'{count} training examples read for tokenizer.')
        text, _ = example
        text = _decode_text(text)
        all_text.append(text)
    logging.info(f'Read all of the training set. {count + 1} examples in total.')
    with tf.io.TFRecordWriter(TEST_PATH, COMPRESSION) as writer:
        for count, example in enumerate(test_dataset):
            count += 1
            if count % 2000 == 0:
                logging.info(f'{count} test examples read for tokenizer.')
            text, _ = example
            text = _decode_text(text)
            all_text.append(text)
    logging.info(f'Read all of the test set. {count} examples in total.')
    tokenizer.fit_on_texts(all_text)
    return tokenizer


def _write_tfrecords(
        train_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        tokenizer: tf.keras.preprocessing.text.Tokenizer):
    options = tf.io.TFRecordOptions(compression_type=COMPRESSION)
    with tf.io.TFRecordWriter(TRAINING_PATH, options) as writer:
        for count, example in enumerate(train_dataset):
            count += 1
            if count % 500 == 0:
                logging.info(f'{count} training examples processed.')
            _tokenize_example(example, tokenizer, writer)
    logging.info(f'Training data processing finished. {count} examples in total.')
    with tf.io.TFRecordWriter(TEST_PATH, COMPRESSION) as writer:
        for count, example in enumerate(test_dataset):
            count += 1
            if count % 500 == 0:
                logging.info(f'{count} test examples processed.')
            _tokenize_example(example, tokenizer, writer)
    logging.info(f'Test data processing finished. {count} examples in total.')


def _decode_text(text):
    text = text.numpy()[0]
    text = text.decode('utf-8')
    text = text.lower()
    text = text.split()
    return text


def _tokenize_example(example: tf.SparseTensor, tokenizer: tf.keras.preprocessing.text.Tokenizer,
                      writer: tf.io.TFRecordWriter) -> ():
    text, label = example
    text = _decode_text(text)
    # Tokenise
    tokens = tokenizer.texts_to_sequences(text)

    # Pad sequences if needed
    tokens = [token[0] for token in tokens if token != []]
    tokens = tokens[0: SEQUENCE_LENGTH]
    tokens += [0] * (SEQUENCE_LENGTH - len(tokens))

    tokens = np.array(tokens)
    label = np.array([label.numpy()])

    assert tokens.shape == (100,)
    assert label.shape == (1,)

    writer.write(_serialize_example(tokens, label))


def _serialize_example(tokens: np.array, label: np.array) -> str:
    feature = {
        TOKEN: tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
        LABEL: tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_test_dataset():
    dataset = tf.data.TFRecordDataset(filenames=[TEST_PATH], compression_type=COMPRESSION)
    dataset = dataset.map(_parse_single_example)
    return _prepare(dataset, is_train=False)


def get_train_valid_dataset():
    dataset = tf.data.TFRecordDataset(filenames=[TRAINING_PATH], compression_type=COMPRESSION)
    dataset = dataset.map(_parse_single_example)

    train_list_ds = dataset.enumerate().filter(lambda count, x: count % 5 != 0)
    train_list_ds = train_list_ds.map(lambda count, x: x)

    valid_list_ds = dataset.enumerate().filter(lambda count, x: count % 5 == 0)
    valid_list_ds = valid_list_ds.map(lambda count, x: x)

    return _prepare(train_list_ds), _prepare(valid_list_ds)


def _parse_single_example(example):
    return tf.io.parse_single_example(serialized=example, features=PROC_FEATURE_DESCRIPTION)


def _prepare(ds, is_train=True, cache=False, shuffle_buffer_size=1000):
    """
    Cache if true (probably shouldn't, data is big), shuffle it, loop it,
    batch it and prefetch batches.
    """
    ds = ds.map(_split_into_token_label_tuple)
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if is_train:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def _split_into_token_label_tuple(example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return example[TOKEN], example[LABEL]


if __name__ == '__main__':
    create_tokenizer_and_datasets()
