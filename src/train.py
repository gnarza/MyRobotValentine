# MyRobotValentine
# Natalie Garza
# 2019

# Train recurrent neural network (RNN) using long short-term memory (LSTM) units

import numpy as np
import sys
import io
import os
import codecs

SEQUENCE_LEN = 3
PERCENT_TEST = 10
STEP = 1

if __name__ == "__main__":
    quotes = sys.argv[1]

    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')

    # Making list of all words and lines in cleanQuotes.txt
    # ('Quotes length in characters: ', 82099)
    # ('Quotes length in words: ', 15975)
    # ('Quotes length in lines: ', 944)

    with io.open(quotes, encoding='utf-8') as f:
        # text is used to just count words and ignore new lines
        text = f.read().replace('\n', ' ')

    with io.open(quotes, encoding='utf-8') as g:
        # textNewLine used to form sentences only within each quote.
        textNewLine = g.read().replace('\n', ' \n ')
        # textNewLine = f.read().lower()

    # print('Quotes length in characters: ', len(text))

    textInWords = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
    textInLines = [w.strip() for w in textNewLine.split('\n') if w.strip() != '']
    # print('Quotes length in words: ', len(textInWords))
    # print('Quotes length in lines: ', len(textInLines))
    # print(textInLines[943])

    # Creating a set of textInWords to get unique words and then sorting set
    # then creating dictionaries of word to index and index to word
    # ('Unique words in quotes: ', 2638)

    words = sorted(set(textInWords))
    # print('Unique words in quotes: ', len(words))
    wordToIndex = dict((c,i) for i, c in enumerate(words))
    indexToWord = dict((c,i) for i, c in enumerate(words))

    # Cut up textInWords into sequence where SEQUENCE_LEN is to be put into
    # sentences[] and the word immediately following the sequence is to be put
    # into nextWord[]. Making sure to keep the corresponding indices in check
    # Ex. sentences[0] = ['i', 'love', 'your']
    #     nextWord[0]  = 'smile'
    # Also want to keep in mind the fact that i don't want to intermingle two
    # quotes in the sentence and nextWord. Once the nextWord is found on the
    # next line the loop should stop and start making sentences on the next
    # quote, this is to keep the net from generating random quotes

    sentences = []
    nextWord = []
    for currLine in textInLines:
        currLine = currLine.split(' ')
        for i in range(0, len(currLine)-SEQUENCE_LEN):
            sentences.append(currLine[i: i+ SEQUENCE_LEN])
            nextWord.append(currLine[i+SEQUENCE_LEN])

    # print(sentences[0])
    # print(nextWord[0])

    # Shuffle and split training and testing set
    tmpSentences = []
    tmpNextWord = []
    for i in np.random.permutation(len(sentences)):
        tmpSentences.append(sentences[i])
        tmpNextWord.append(nextWord[i])


    cutIndex = int(len(sentences) * (1.-(PERCENT_TEST/100.)))
    xTrain, xTest = tmpSentences[:cutIndex], tmpSentences[cutIndex:]
    yTrain, yTest = tmpNextWord[:cutIndex], tmpNextWord[cutIndex:]
    print("Length of training set: ", len(xTrain))
    print("Length of testing set: ", len(xTest))
