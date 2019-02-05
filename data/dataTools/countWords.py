# MyRobotValentine
# Natalie Garza
# 2019

# Used to count the number of words in the data set as well as the unique words.
# Also small wordCloud examination.

# RESULTS
# ('TOTAL WORDS: ', 16015)
# ('UNIQUE WORDS: ', 2976)

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

totalWords = 0
uniqueWords = set()

stopwords = set(STOPWORDS)
wordsies = ' '

file = open("cleanQuotes.txt", "r")
for line in file:
    lineList = line.split(" ")
    totalWords += len(lineList)
    for word in lineList:
        uniqueWords.add(word)
        wordsies = wordsies + word + ' '

file.close()

wordcloud = WordCloud(width = 800, height = 800, background_color = 'white', stopwords = stopwords, min_font_size = 10).generate(wordsies)

plt.figure(figsize = (8,8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
print("TOTAL WORDS: ", totalWords)
print("UNIQUE WORDS: ", len(uniqueWords))
