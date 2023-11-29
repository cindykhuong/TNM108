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

# take out shuffled training set with 100000
train_ci = all_ci[:199999]
train_rc = all_rc[:199999]

test_ci = all_ci[100000:199999]
test_rc = all_rc[100000:199999]

# PIPELINE
tomato_clf = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", LogisticRegression(solver="liblinear", multi_class="auto")),
    ]
)
# ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-5, random_state=42
# ,max_iter=8, tol=None)),
# ])
# ('clf', MultinomialNB()),
# ])

tomato_clf.fit(all_rc, all_ci)
# print("hej")

filename = "tomatoModel_LR.sav"
pickle.dump(tomato_clf, open(filename, "wb"))


# CROSS VALIDATION
# cv = 10
# scores = cross_val_score(tomato_clf, all_rc, all_ci, cv=cv)
# print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# parameters = {
# 'vect__ngram_range': [(1, 1), (1, 2)],
# 'tfidf__use_idf': (True, False),
# 'clf__alpha': (1e-2, 1e-5),
# }

# GRID SEARCH

# gs_clf = GridSearchCV(tomato_clf, parameters, cv=5, iid=False, n_jobs=-1)
# gs_clf = gs_clf.fit(all_rc, all_ci)
# for param_name in sorted(parameters.keys()):
# print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# print(pred[0])

# NEXT STEP: validering, test-data, räkna ut score, koppla till andra filen för att få filmnamn och jämföra betyget
# skapa validering
