# MyRobotValentine
# Natalie Garza
# 2019

# WordCloud examination.

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
wordsies = ' '

file = open("data/cleanQuotes.txt", "r")
for line in file:
    line.replace('\n', ' ')
    line.strip()
    lineList = line.split(' ')
    for word in lineList:
        wordsies = wordsies + word + ' '

file.close()

wordcloud = WordCloud(width = 800, height = 800, background_color = 'white', stopwords = stopwords, min_font_size = 10).generate(wordsies)

plt.figure(figsize = (8,8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
