import logging
import pickle

import tensorflow as tf

from custom_models.rnn_model import RNNModel

SEQUENCE_LENGTH = 100

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

logging.info("Loading tokenizer")
with open('data_pipelines/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

model = RNNModel(tokenizer,
                 n_rnns=1,
                 n_dense=2,
                 n_attention_heads=4,
                 attention_units=20,
                 rnn_units=128,
                 dense_units=50)

logging.info('building model')
model.build(tf.TensorShape([None, 100]))
model.load_weights('imdb_simple.h5')


def classify_review(text):
    tokens = tokenizer.texts_to_sequences(text.split(' '))
    tokens = tokens[0: SEQUENCE_LENGTH]
    tokens = [t[0] for t in tokens]
    tokens += [0] * (SEQUENCE_LENGTH - len(tokens))
    tokens = tf.constant([tokens])
    print(text)
    print(model.predict(tokens)[0][0])


if __name__ == '__main__':
    classify_review('this movie is shit')
    # 0.18047556

    classify_review('this movie is amazing')
    # 0.83419615

    classify_review('i love everything about this film the casting was great and the music was moving')
    # 0.9363523

    classify_review('i hate everything about this film the casting was shit and the music was garbage dont '
                    'get me started on the sexism ugh 0 out of 5 is not low enough for this piece of filth')
    # 3.1702504e-05

    classify_review('this film made me want to throw myself in a microwave and ask someone to throw me in thames '
                    'because it was so terrible it was so stinky')
    # 0.0004168154

    classify_review('this movie was the most amazing film i have ever seen in my life i would wholeheartedly recommend'
                    ' it to anyone from donald trump to my great aunt susan god rest her soul')
    # 0.95315135

    classify_review('this movie had a plot and characters they did things time passed i ate popcorn i like cheese'
                    ' dont you')
    # 0.5618066
