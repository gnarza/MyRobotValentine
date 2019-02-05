# MyRobotValentine
# Natalie Garza
# 2019

# HTML scraper used to collect catchphrases, pick up lines, sayings, quotes,
# and popular love songs relating to Valentine's Day.
# Does not completely return fully clean data, still needed some manual
# cleaning after scraping. Look at saved text file to get the clean data.

from lxml import html
import requests

allQuotes = set()
cuttingBoard = []

# greetingcardpoet.com -----------------------------------------------------------------------------------------------------
sourcesOne = ['https://www.greetingcardpoet.com/love-quotes/','https://www.greetingcardpoet.com/romance-quotes-new-funny-for-him/','https://www.greetingcardpoet.com/long-distance-relationship-quotes-and-sayings/','https://www.greetingcardpoet.com/cute-couple-quotes-friendship-romance-and-love/']
quoteLen = [87,61,56,48]
# used to cut up scraped text if format isn't correct

j = 0
for s1 in sourcesOne:
    page = requests.get(s1)
    # tree contains the whole HTML file in a tree structure
    tree = html.fromstring(page.content)

    quotes = tree.xpath('/html/body/div[1]/div[1]/div/main/article/div//p[position()<%d]' % quoteLen[j])
    j += 1

    for q1 in quotes:
        q1 = q1.text_content().lower()
        cuttingBoard = q1.rsplit('.', 1)
        allQuotes.add(cuttingBoard[0])

# pickuplinesgalore.com --------------------------------------------------------------------------------------------------------
sourcesTwo = ['https://www.pickuplinesgalore.com/computer.html', 'https://www.pickuplinesgalore.com/math.html','https://www.pickuplinesgalore.com/physics.html','https://www.pickuplinesgalore.com/engineering.html', 'https://www.pickuplinesgalore.com/astronomy.html','https://www.pickuplinesgalore.com/robot.html']
for s2 in sourcesTwo:
    page = requests.get(s2)
    # tree contains the whole HTML file in a tree structure
    tree = html.fromstring(page.content)

    quotes = tree.xpath('/html/body/main/main/p')
    for q2 in quotes:
        q2 = q2.text_content().lower()
        cuttingBoard = q2.split('\n')
        for k in cuttingBoard:
            allQuotes.add(k.lstrip())

# wisdomquotes.com--------------------------------------------------------------------------------------------------------
page = requests.get('http://wisdomquotes.com/love-quotes/')
tree = html.fromstring(page.content)

quotes = tree.xpath('//*[@id="post-866"]/div[2]/blockquote')
for q3 in quotes:
    q3 = q3.text_content().lower()
    cuttingBoard = q3.rsplit('.', 1)
    allQuotes.add(cuttingBoard[0])

# Create text --------------------------------------------------------------------------------------------------------
file = open('quotes.txt', 'w')
for q in allQuotes:
    q = q + '\n'
    file.write(q.encode('ascii', 'ignore'))
file.close()

# TESTING --------------------------------------------------------------------------------------------------------
# print "ALLQUOTES: ",len(allQuotes), "\n"
#
# i = 0
# for q in allQuotes:
#     i += 1
#     print i, ': ', q, '\n'
