# :sparkling_heart: My Robot Valentine :sparkling_heart:
---
##### By Natalie Garza
##### 2019

##### :sparkling_heart: Why I made this
---
Why do machines get criticized for being emotionless?

As a short investigation and personal study on Natural Language Processing I\ decided to train a recurrent neural network in the language of love\ in light of Valentine's Day.

##### :sparkling_heart: What it is
---
Building upon Enrique A.'s Word-Level Long Short-Term Memory Text Generator\ Neural Network I wanted
to feed the network with my own data-set.

- checkpoints/: In this directory you will find the network data I trained on my computer.
- data/: Here is all the scripts I used to scrape up the data as well as the final data-set\ I used to train the network "cleanQuotes.txt".
- src/: Contains the main training program as well as the program used to load an already\ existing network from checkpoints/. Notice that "train.py" is coded specifically for "cleanQuotes.txt".

##### :sparkling_heart: How to use
---
On Linux: Make sure you have Python and Pip installed

Cloning Repository: Copy this repo to your computer and go to the directory.
```sh
$ git clone
cd MyRobotValentine
```

For Dependencies: Install Pipenv to take care of dependencies for you.
```sh
pip install --user pipenv
pipenv install
pipenv shell
```

Training Network: Provide the training data file, training output file, and\ the vocabulary file
Note: A new training output file and vocabulary file will be created if doesn't\ already exist.
Run this from the root of MyRobotValentine.
```sh
pipenv run python src/train.py data/cleanQuotes.txt trainOut.txt vocab.txt
```

Generating Text: Provide the vocab.txt, pre-trained model checkpoint, and a\ quote to use as a seed.
```sh
pipenv run python src/generate.py vocab.txt checkpoint/**copy & paste path of a checkpoint** "seed used to generate: write whatever you want"
```
Example:
```sh
pipenv run python src/generate.py vocab.txt checkpoint/checkpoints/LSTM_LOVE-epoch014-words2639-sequence7-loss2.2513-acc0.5253-val_loss8.1992-val_acc0.0813 "I love your"
```

#### :sparkling_heart: Where I found stuff
---
[Word-Level LSTM text generator.](https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb)\
[How to create a poet using text generation using Python.](https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/)\
[aiweirdness blog](http://aiweirdness.com/)

Used for scraping data:
- [Greeting Card Poet](www.greetingcardpoet.com)
- [Pick Up Lines Galore](www.pickuplinesgalore.com)
- [Wisdom Quotes](http://wisdomquotes.com/)
