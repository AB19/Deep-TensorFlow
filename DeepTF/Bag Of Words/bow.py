# libraries
import pandas as pd
import os
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# set the directory
os.chdir (".")

# import the data-set
# header = 0 -> first row are headings
# delimiter = "\t" -> tab separated
# quoting = 3 -> ignore the " in the text
data = pd.read_csv ("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv ("testData.tsv", header = 0, delimiter = "\t", quoting = 3)
reviews = data.iloc [:, 2].values
sentiment = data.iloc [:, 1].values
test_reviews = test.iloc [:, 1].values

# cleaning the data, remove the html tags and keep only the text
reviews = [BeautifulSoup (review).text for review in reviews]
test_reviews = [BeautifulSoup (review).text for review in test_reviews]

# removal of stop-words 
# these are reviews, so things like !!! & :), :( do carry value
# donot eliminate them, but numbers can be taken out
reviews = [re.sub ("[^a-zA-Z]", " ", review) for review in reviews]
test_reviews = [re.sub ("[^a-zA-Z]", " ", review) for review in test_reviews]

# convert all the words to lower case
reviews = [review.lower () for review in reviews]
test_reviews = [review.lower () for review in test_reviews]

# tokenize the words
reviews = [word_tokenize (review) for review in reviews]
test_reviews = [word_tokenize (review) for review in test_reviews]

# removal of stop-words
stop_words = stopwords.words ("english")
cleaned_reviews = []
cleaned_test_reviews = []

for review in reviews:
    cleaned_review = [word for word in review if word not in stop_words]
    cleaned_review = " ".join (cleaned_review)
    cleaned_reviews.append (cleaned_review)

for review in test_reviews:
    cleaned_test_review = [word for word in review if word not in stop_words]
    cleaned_test_review = " ".join (cleaned_test_review)
    cleaned_test_reviews.append (cleaned_test_review)
    
# now, we can use the CountVectorizer method in sklearn
# it does two things, first -> learn the vocabulary/ bag of words
# second -> creates feature vectors for the traning set
# we can limit the size of vocabulary - to avoid sparse huge sparse matrices
vectorizer = CountVectorizer (analyzer = "word", preprocessor = None,
                              tokenizer = None, stop_words = None)
features = vectorizer.fit_transform (cleaned_reviews)

# use a classifier to train for the sentiment scores 
# random forest classifier
forest = RandomForestClassifier (n_estimators = 100) 
# fit the classifier to the features & sentiments
forest = forest.fit (features, sentiment)

# transform the model to the test data-set
test_features = vectorizer.transform (cleaned_test_reviews)
result = forest.predict (test_features)

test_reviews = [" ".join (review) for review in test_reviews]
output = pd.DataFrame ( data = {"reviews": test_reviews, "sentiment": result} )
output.to_csv ("bow_model.csv", index = False, quoting = 3 )

# Analysis on the percentage of yes and no
pos, neg = 0, 0
for value in result:
    if value == 1:
        pos += 1
    else:
        neg += 1
print ("Percentage of Positive reviews: " + str ( (pos) / len (result)))
print ("Percentage of Negative reiews: " + str ( (neg) / len (result)))