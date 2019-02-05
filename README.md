Step one was finding websites containing enough quotes to be able to collect with as little code as possible.

Using lxml, requests, and the inspect tool on Google the code was scraped. Note that the outcome was slightly cleaned code.
This means that a lot of the punctuation was still present and some unnecessary text was also scraped (ex. "Find more love quotes here").

Manually parsing through the document I was able to get rid of unwanted text such as author names, initials, and stuff like the example above.
After a couple of passes (Don't worry it took maybe 10 minutes or so) I was able to write a short program to get rid of the punctuation.

The remaining cleaned code was put into the cleanQuotes.txt. Once more manual pass was done to fix any issues the removal of punctuation may have caused.
Since the neural network will

Resources I used to help me make this project possible:
