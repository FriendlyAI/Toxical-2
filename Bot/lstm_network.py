import datetime
import time
from random import randint

import numpy as np
import tensorflow as tf

from ids import MAX_SENTENCE_LENGTH, NUM_FILES, NUM_NEGATIVE, tokenize

# Hyperparameters

BATCH_SIZE = 24
LSTM_UNITS = 32
NUM_CLASSES = 2
ITERATIONS = 100000
NUM_DIMENSIONS = 50
LEARNING_RATE = 0.0005


def create(train_model=False):
    return Model(train_model)


class Model:
    def __init__(self, train_model):
        self.IDS = np.load('ids_matrix.npy')

        # Load word vectors
        # GloVe vectors from https://nlp.stanford.edu/projects/glove/
        # 50 Dimensional vectors

        self.word_list = np.load(
            'Word Vectors/GloVe_words.npy').tolist()  # Load as list; used for getting indices to access word vectors

        self.word_list = [word.decode('UTF-8') for word in self.word_list]  # Decode all words in UTF-8 format

        self.word_vectors = np.load('Word Vectors/GloVe_vectors.npy')  # Load as NumPy array

        tf.reset_default_graph()

        self.labels = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])
        self.input_data = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SENTENCE_LENGTH])

        tf.Variable(tf.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH, NUM_DIMENSIONS]), dtype=tf.float32)
        data = tf.nn.embedding_lookup(self.word_vectors, self.input_data)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
        value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([LSTM_UNITS, NUM_CLASSES]))
        bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        self.prediction = (tf.matmul(last, weight) + bias)

        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, labels=self.labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        self.session = tf.InteractiveSession()
        saver = tf.train.Saver()

        if train_model:
            self.train(loss, optimizer, saver)
        else:
            saver.restore(self.session, tf.train.latest_checkpoint('Models'))

    def to_matrix(self, sentence):
        sentence_matrix = np.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH], dtype='int32')
        tokenized = tokenize(sentence)
        for index, word in enumerate(tokenized):
            if index >= 100:
                break
            else:
                try:
                    sentence_matrix[0, index] = self.word_list.index(word)
                except ValueError:
                    sentence_matrix[0, index] = 399999  # Vector for unknown words
        return sentence_matrix

    def predict(self, sentence):
        sentence_matrix = self.to_matrix(sentence)
        return self.session.run(self.prediction, {self.input_data: sentence_matrix})[0]

    def train(self, loss, optimizer, saver):
        start = time.time()

        self.session.run(tf.global_variables_initializer())
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', self.accuracy)
        merged = tf.summary.merge_all()
        logdir = f'Tensorboard/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
        writer = tf.summary.FileWriter(logdir, graph=self.session.graph)

        for i in range(1, ITERATIONS + 1):
            print(f'Batch {i}')
            next_batch, next_batch_labels = self.get_training_batch()
            self.session.run(optimizer, {self.input_data: next_batch, self.labels: next_batch_labels})

            # Write summary to Tensorboard
            # Run tensorboard --logdir=Data/Tensorboard
            # View at localhost:6006
            if i % 50 == 0:
                summary = self.session.run(merged, {self.input_data: next_batch, self.labels: next_batch_labels})
                writer.add_summary(summary, i)
                writer.flush()

            # Save the network every 10,000 training iterations
            if i % 10000 == 0:
                save_path = saver.save(self.session, 'Models/pretrained_lstm.ckpt', global_step=i)
                print(f'Saved to {save_path}')
            # writer.close()

        print(f'Took {(time.time() - start) / 60} minutes')

    def test_model(self):
        iterations = 10
        average_accuracy = 0
        for i in range(iterations):
            next_batch, next_batch_labels = self.get_testing_batch()
            test_accuracy = self.session.run(self.accuracy,
                                             {self.input_data: next_batch, self.labels: next_batch_labels}) * 100

            print(f'Accuracy for batch {i + 1}: {test_accuracy}')
            average_accuracy += test_accuracy
        print(f'Average accuracy: {average_accuracy / 10}')

    def get_training_batch(self):
        labels = []
        arr = np.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH])
        for i in range(BATCH_SIZE):
            if i % 2 == 0:
                num = randint(1, NUM_NEGATIVE)  # Add negative file to train
                labels.append([1, 0])  # Negative label
            else:
                num = randint(NUM_NEGATIVE, NUM_FILES)  # Add non-negative file to train
                labels.append([0, 1])  # Non-negative label
            arr[i] = self.IDS[num - 1]
        return arr, labels

    def get_testing_batch(self):
        labels = []
        arr = np.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH])
        for i in range(BATCH_SIZE):
            num = randint(1, NUM_FILES)
            if num <= NUM_NEGATIVE:
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            arr[i] = self.IDS[num - 1]
        return arr, labels


if __name__ == '__main__':
    create(True)

# Took 55.92745286623637 minutes
