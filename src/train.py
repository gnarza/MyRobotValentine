# MyRobotValentine
# Natalie Garza
# 2019

# Train recurrent neural network (RNN) using long short-term memory (LSTM) units.
# Save checkpoints of model trained.
# Create file containing all outputs of each epoch.
# Run from root directory using :
# python src/train.py data/cleanQuotes.txt fileForOutput.txt
# Note that you can name that last file whatever you want.

from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
import numpy as np
import sys
import io
import os

SEQUENCE_LEN = 10
PERCENT_TEST = 10
DROPOUT = 0
BATCH_SIZE = 32
STEP = 1

# Used to feed model in chunks as opposed to all at once.
def generator(sentenceList, nextWordList, batchsize):
    index = 0
    while True:
        # np.zeros returns a new array of given shape and type, filled with zeros
        x = np.zeros((batchsize, SEQUENCE_LEN, len(words)), dtype=np.bool)
        y = np.zeros((batchsize, len(words)), dtype=np.bool)
        for i in range(batchsize):
            for t, w in enumerate(sentenceList[index % len(sentenceList)]):
                x[i, t, wordToIndex[w]] = 1
            y[i, wordToIndex[nextWordList[index % len(sentenceList)]]] = 1
            index += 1
        yield x, y

# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    expPreds = np.exp(preds)
    preds = expPreds / np.sum(expPreds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def onEpochEnd(epoch, logs):
    examplesFile.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Picking a random sentence to use as the seed for the network
    seedIndex = np.random.randint(len(senTrain+senTest))
    seed = (senTrain+senTest)[seedIndex]

    # Diversity (Temperature in LSTM talk) controls the randmoness of the
    # predictions made. Smaller values make up the random, and larger values
    # appear to be more confident in their predictions.
    for diversity in [.3, .4, .5, .6, .7]:
        sentence = seed
        examplesFile.write('----- Diversity: ' + str(diversity) + '\n')
        examplesFile.write('----- Generating with : ' + ' '.join(sentence) + '\n')
        # examplesFile.write(' '.join(sentence))

        for i in range(20):
            # This is the numpy array required as the input data for predict()
            xPred = np.zeros((1, SEQUENCE_LEN, len(words)))
            for t, word in enumerate(sentence):
                # Take a word from the seed sentence and set the corresponding
                # value of the word in array to 1
                xPred[0, t, wordToIndex[word]] = 1.

                preds = model.predict(xPred, verbose=0)[0]
                nextIndex = sample(preds, diversity)
                nextWord = indexToWord[nextIndex]

                sentence = sentence[1:]
                sentence.append(nextWord)

                examplesFile.write(" " + nextWord)
            examplesFile.write("\n")
        examplesFile.write("=" * 80 + "\n")
        examplesFile.flush()

if __name__ == "__main__":
    quotes = sys.argv[1]
    examples = sys.argv[2]

    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')

    # Making list of all words and lines in cleanQuotes.txt
    with io.open(quotes) as f:
        # text is used to just count words and ignore new lines
        # text = f.read().replace('\n', ' ').encode('utf-8')
                text = f.read().replace('\n', ' \n ').encode('utf-8')

    with io.open(quotes) as g:
        # textNewLine used to form sentences only within each quote.
        textNewLine = g.read().encode('utf-8')

    textInWords = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
    textInLines = [w.strip() for w in textNewLine.split('\n') if w.strip() != '']

    # Creating a set of textInWords to get unique words and then sorting set
    # then creating dictionaries of word to index and index to word
    textInWords = set(textInWords)
    textInWords.add('')
    words = sorted(textInWords)

    wordToIndex = dict((c,i) for i, c in enumerate(words))
    indexToWord = dict((i,c) for i, c in enumerate(words))

    # Cut up textInWords into sequence where SEQUENCE_LEN is to be put into
    # sentences[] and the word immediately following the sequence is to be put
    # into nextWord[]. Making sure to keep the corresponding indices in check
    # Ex. sentences[0] = ['i', 'love', 'your']
    #     nextWord[0]  = 'smile'
    # Also want to keep in mind the fact that i don't want to intermingle two
    # quotes in the sentence and nextWord. Once the nextWord is found on the
    # next line the loop should stop and start making sentences on the next
    # quote
    sentences = []
    nextWords = []
    for currLine in textInLines:
        currLine = currLine.split(' ')
        for i in range(0, len(currLine)-SEQUENCE_LEN):
            if currLine[i+SEQUENCE_LEN] != '':
                sentences.append(currLine[i: i + SEQUENCE_LEN])
                nextWords.append(currLine[i+SEQUENCE_LEN])

        # sentences.append(currLine[len(currLine) - SEQUENCE_LEN: len(currLine)])
        # nextWords.append('\n')

    # print(sentences[:50])
    # print(nextWords[:50])
    # Shuffle and split training and testing set
    tmpSentences = []
    tmpNextWord = []
    for i in np.random.permutation(len(sentences)):
        tmpSentences.append(sentences[i])
        tmpNextWord.append(nextWords[i])


    cutIndex = int(len(sentences) * (1.-(PERCENT_TEST/100.)))
    senTrain, senTest = tmpSentences[:cutIndex], tmpSentences[cutIndex:]
    nwTrain, nwTest = tmpNextWord[:cutIndex], tmpNextWord[cutIndex:]

    # Training the model
    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
    if DROPOUT > 0:
        model.add(Dropout(DROPOUT))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Saving trained models
    filePath = "./checkpoints/LSTM_LOVE-epoch{epoch:03d}-words%d-sequence%d-" \
                "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % \
                (len(words), SEQUENCE_LEN)

    checkpoint = ModelCheckpoint(filePath, monitor='val_acc', save_best_only=True)
    printCallback = LambdaCallback(on_epoch_end=onEpochEnd)
    earlyStopping = EarlyStopping(monitor='val_acc', patience=5)
    callbacksList = [checkpoint, printCallback, earlyStopping]

    examplesFile = open(examples, "w")
    model.fit_generator(generator(senTrain, nwTrain, BATCH_SIZE),
                        steps_per_epoch=int(len(senTrain)/BATCH_SIZE) + 1,
                        epochs=100,
                        callbacks=callbacksList,
                        validation_data=generator(senTest, nwTest, BATCH_SIZE),
                        validation_steps=int(len(senTest)/BATCH_SIZE) + 1)
