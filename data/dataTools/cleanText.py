# MyRobotValentine
# Natalie Garza
# 2019

# Will parse through quotes.txt and remove any preceding and following whitespace,
# commas, periods, quotations, etc...
# Also note that I still had to make some manual edits to both quotes.txt and
# cleanQuotes.txt.

# New file for cleaned text removes all punctuation characters
# Put a 1 in the file name as a safe gauard to avoid ruining manually cleaned code.
newFile = open("cleanQuotes1.txt","w")
file = open("quotes.txt", "r")
for line in file:

    for i in ",.!?'\"\t:;=+-*/":
        line = line.replace(i, "")
    newFile.write(line)

file.close()
newFile.close()

# Used for small test
# e=0
# if(e<5):
#     print(line)
# e+=1
