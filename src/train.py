# MyRobotValentine
# Natalie Garza
# 2019

# Train recurrent neural network (RNN) using long short-term memory (LSTM) units

import numpy as np
import sys
import io
import os
from keras.models import Sequential
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional

SEQUENCE_LEN = 3
PERCENT_TEST = 10
DROPOUT = 0
BATCH_SIZE = 32
STEP = 1

def generator(sentenceList, nextWordList, batchsize):
    print('hello')
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

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    expPreds = np.exp(preds)
    preds = expPreds . np.sum(expPreds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def onEpochEnd(epoch, logs):
    examplesFile.write('\n----- Generating text after Epoch: %d\n' % epoch)

    seedIndex = np.random.randint(len(senTrain, senTest))
    seed = (senTrain+senTest)[seedIndex]

    for diversity in [.3, .4, .5, .6, .7]:
        sentence = seed
        examplesFile.write('----- Diversity: ' + str(diversity) + '\n')
        examplesFile.write('----- Generating with :\n' + ' '.join(sentence) + '\n')
        examplesFile.write(' '.join(sentence))

        for i in range(50):
            xPred = np.zeros((1, SEQUENCE_LEN, len(words)))
            for t, word in enumerate(sentence):
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
    nextWords = []
    for currLine in textInLines:
        currLine = currLine.split(' ')
        for i in range(0, len(currLine)-SEQUENCE_LEN):
            sentences.append(currLine[i: i+ SEQUENCE_LEN])
            nextWords.append(currLine[i+SEQUENCE_LEN])

    # print(sentences[0])
    # print(nextWord[0])

    # Shuffle and split training and testing set
    tmpSentences = []
    tmpNextWord = []
    for i in np.random.permutation(len(sentences)):
        tmpSentences.append(sentences[i])
        tmpNextWord.append(nextWords[i])


    cutIndex = int(len(sentences) * (1.-(PERCENT_TEST/100.)))
    senTrain, senTest = tmpSentences[:cutIndex], tmpSentences[cutIndex:]
    nwTrain, nwTest = tmpNextWord[:cutIndex], tmpNextWord[cutIndex:]
    print("Length of sentences training set: ", len(senTrain))
    print("Length of nextWords training set: ", len(nwTrain))

    print("Length of sentences testing set: ", len(senTest))
    print("Length of nextWords testing set: ", len(nwTest))


    # Training the model
    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
    if DROPOUT > 0:
        model.add(Dropout(DROPOUT))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
