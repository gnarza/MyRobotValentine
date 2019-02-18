# Natalie Garza
# MyRobotValentine
# 2019

# Generate text from already trained network

from __future__ import print_function
import numpy as np
import re
import sys
from keras.models import load_model
from train import sample

if __name__ == "__main__":
    # all unique words found in cleanQuotes.txt
    vocab = sys.argv[1]
    # Path of the trained network
    net = sys.argv[2]
    # Seed used to generate text
    seed = sys.argv[3]

    # sequence length default is 10
    # Value of diversity default is .6
    # Quantity of words to generate default is 50
    if len(sys.argv) != 6:
        seqLen = 10
        diversity = .6
        quantity = 5
    else:
        seqLen = sys.argv[4]
        diversity = sys.argv[5]
        quantity = sys.argv[6]

    model = load_model(net)
    print('Model Summary: ')
    model.summary()

    vocabulary = open(vocab, "r").readlines()
    vocabulary = [re.sub(r'(\S+)\s+', r'\1', w) for w in vocabulary]
    vocabulary = sorted(set(vocabulary))

    wordToIndex = dict((c, i) for i, c in enumerate(vocabulary))
    indexToWord = dict((i, c) for i, c in enumerate(vocabulary))

    seed = " ".join((((seed+" ")*seqLen)+seed).split(" ")[-seqLen:])

    sentence = seed.split(" ")
    print("----- Generating text")
    print('----- Diversity:' + str(diversity))
    print('----- Generating with seed:\n"' + seed)

    print(seed)
    for i in range(quantity):
        xPred = np.zeros((1, seqLen, len(vocabulary)))
        for t, word in enumerate(sentence):
            xPred[0,t,wordToIndex[word]] = 1

            preds = model.predict(xPred, verbose=0)[0]
            nextIndex = sample(preds, diversity)
            nextWord = indexToWord[nextIndex]

            sentence = sentence[1:]
            sentence.append(nextWord)

            print(" "+nextWord, end="")
        print("\n")
