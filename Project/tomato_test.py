from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
import sklearn
from sklearn.datasets import load_files
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

# read in dataset
reviews = pd.read_csv("./input/IMDB_Dataset.csv")

movie_reviews = pd.read_csv("movie_reviews/frozen2.csv")
movie_reviews.drop_duplicates(subset="review", keep="first", inplace=True)

# take away all columns except ci and rc
# reviews = reviews[["review_type", "review_content"]]
reviews = reviews[["sentiment", "review"]]


# drop all empty reviews
reviews = reviews.dropna()

# read in the reviews and critic (rotten/fresh)
all_ci = reviews.sentiment
all_rc = reviews.review

# shuffle data
all_ci, all_rc = shuffle(all_ci, all_rc, random_state=67)

# test data
test_ci = all_ci[100000:199999]
test_rc = all_rc[100000:199999]

# tomatometer model
tomatoModel = pickle.load(open("tomatoModel_SGD.sav", "rb"))

predicted = tomatoModel.predict(movie_reviews.review)
print(len(movie_reviews))
print(len(predicted))
# print("confusion matrix: ", metrics.confusion_matrix(movie_reviews, predicted))

print("tomatometer: ", np.mean(predicted == "positive"))

# GRID SEARCH
