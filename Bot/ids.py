import os
import re
import time
from os import listdir
from os.path import isfile, join

import numpy as np

start = time.time()

MAX_SENTENCE_LENGTH = 100
NUM_FILES = 313217  # From metadata.py
NUM_NEGATIVE = 80712
NUM_NON_NEGATIVE = 232505

os.chdir('../Data')


def tokenize(sentence):
    """
    Remove non-words and split sentence by word
    :param sentence:
    :return: list[str]
    """
    special_chars = re.compile(r'\W+')  # Replace all characters except letters and numbers
    tokens = re.sub(special_chars, ' ', sentence)  # Substitute special characters with spaces
    return [token.lower() for token in tokens.split() if token]


# Load word vectors
def main():
    word_list = np.load('Word Vectors/GloVe_words.npy').tolist()  # Used for getting indices to access word vectors
    word_list = [word.decode('UTF-8') for word in word_list]  # Decode all words in UTF-8 format

    negative_files = ['Training Data/Negative/' + f for f in listdir('Training Data/Negative/')
                      if isfile(join('Training Data/Negative/', f))]
    non_negative_files = ['Training Data/Non-negative/' + f for f in listdir('Training Data/Non-negative/')
                          if isfile(join('Training Data/Non-negative/', f))]

    ids = np.zeros((NUM_FILES, MAX_SENTENCE_LENGTH), dtype='int32')

    file_count = 0

    for n in negative_files:
        # 80,712 files
        with open(n, 'r') as f:
            index = 0
            line = f.readline()
            cleaned = tokenize(line)
            for word in cleaned:
                try:
                    ids[file_count][index] = word_list.index(word)
                except ValueError:
                    ids[file_count][index] = 399999  # Vector for unknown words
                index += 1
                if index >= MAX_SENTENCE_LENGTH:
                    break
            file_count += 1
            print(f'{file_count}/{NUM_FILES}\t{round(file_count / NUM_FILES * 100, 4)}%')

    for nn in non_negative_files:
        # 232,505 files
        with open(nn, 'r') as f:
            index = 0
            line = f.readline()
            cleaned = tokenize(line)
            for word in cleaned:
                try:
                    ids[file_count][index] = word_list.index(word)
                except ValueError:
                    ids[file_count][index] = 399999  # Vector for unknown words
                index += 1
                if index >= MAX_SENTENCE_LENGTH:
                    break
            file_count += 1
            print(f'{file_count}/{NUM_FILES}\t{round(file_count / NUM_FILES * 100, 4)}%')

    np.save('ids_matrix', ids)

    print(f'Took {(time.time() - start) / 60} minutes')


if __name__ == '__main__':
    main()

# Took 74.54950703382492 minutes
