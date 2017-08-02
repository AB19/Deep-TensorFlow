# create the bag of words model from the created lexicon 
# does not perform great whne compared to seqeuncial models
# does not consider the order of the words

# use nltk to create the bag of words model
import nltk
from nltk.tokenize import word_tokenize

# set the directory
import os
os.chdir (".")

# import the data-sets
# the data-sets are text files
import glob
file_list = sorted (glob.glob ("*.txt"))
lexicon = []
for file in file_list:
    with open (file, 'r') as f:
        contents = f.readlines ()
        for line in contents:
            words = word_tokenize (line.lower ())
            lexicon = lexicon + words
print ("Size of lexicon: " + str (len (lexicon))) 
# lemmatize the entire lexicon
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer ()
lexicon = [lemma.lemmatize (i) for i in lexicon]
# create a dictionary of the occurance of every word
from collections import Counter
word_counts = Counter (lexicon)
# create a list of words that make a difference
# eliminate words that are common (occur more than 1000)
clean_lexicon = []
for w in word_counts:
    if 5 < word_counts [w] < 1000:
        clean_lexicon.append (w)
print ("Size of the cleaned lexicon: " + str (len (clean_lexicon)))
