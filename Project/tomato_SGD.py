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
import pickle

# read in dataset
# reviews = pd.read_csv("./input/rotten_tomatoes_critic_reviews.csv")
reviews = pd.read_csv("./input/IMDB_Dataset.csv")


# take away all columns except ci and rc
# reviews = reviews[["review_type", "review_content"]]
reviews = reviews[["sentiment", "review"]]


# drop all empty reviews
reviews = reviews.dropna()

# read in the reviews and critic (rotten/fresh)
all_ci = reviews.sentiment
all_rc = reviews.review

# review from a specific movie
review_ci = all_ci[143:280]
review_rc = all_rc[143:280]

# shuffle data
all_ci, all_rc = shuffle(all_ci, all_rc, random_state=67)

# PIPELINE
tomato_clf = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        (
            "clf",
            SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=1e-7,
                random_state=42,
                max_iter=500,
                tol=None,
            ),
        ),
    ]
)

tomato_clf.fit(all_rc, all_ci)

# Save model and write to sav-file
filename = "tomatoModel_SGD.sav"
pickle.dump(tomato_clf, open(filename, "wb"))
