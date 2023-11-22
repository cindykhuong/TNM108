text = """Neo-Nazism consists of post-World War II militant social or political
movements seeking to revive and implement the ideology of Nazism. Neo-Nazis seek to
 employ their ideology to promote hatred and attack minorities, or in some cases
to create a fascist political state. It is a global phenomenon, with organized
representation in many countries and international networks. It borrows elements
from Nazi doctrine, including ultranationalism, racism, xenophobia, ableism, hom
ophobia, anti-Romanyism, antisemitism, anti-communism and initiating the Fourth
Reich. Holocaust denial is a common feature, as is the incorporation of Nazi sym
bols and admiration of Adolf Hitler. In some European and Latin American countri
es, laws prohibit the expression of pro-Nazi, racist, anti-Semitic, or homophobi
c views. Many Nazi-related symbols are banned in European countries (especially
Germany) in an effort to curtail neo-Nazism. The term neo-Nazism describes any p
ost-World War II militant, social or political movements seeking to revive the i
deology of Nazism in whole or in part. The term neo-Nazism can also refer to the
ideology of these movements, which may borrow elements from Nazi doctrine, inclu
ding ultranationalism, anti-communism, racism, ableism, xenophobia, homophobia,
anti-Romanyism, antisemitism, up to initiating the Fourth Reich. Holocaust denia
l is a common feature, as is the incorporation of Nazi symbols and admiration of
Adolf Hitler. Neo-Nazism is considered a particular form of far-right politics and
right-wing extremism."""

pingu = """"Penguins are flightless birds with flippers instead of wings. Their 
bodies are adapted for swimming and diving in the water, with some species able to
 reach speeds up to 15 miles per hour. Their body shape is fusiform (tapered at both ends) 
 and streamlined, allowing them to be expert swimmers. They have a large head, short neck, 
 and elongated body. Their tails are short, stiff, and wedge-shaped.

Their legs and webbed feet are set far back on the body, which gives penguins their 
upright posture on land. When snow conditions are right, they will slide on their 
bellies."""

from summa.summarizer import summarize

# Define length of the summary as a proportion of the text
print(summarize(pingu, ratio=0.2))
print(summarize(pingu, words=50))

from summa import keywords

print("Keywords:\n", keywords.keywords(pingu))
# to print the top 3 keywords
print("Top 3 Keywords:\n", keywords.keywords(pingu, words=3))

# ----------------------------------------------------------------------------
# # Parameter Tuning Using Grid Search
# from sklearn.datasets import fetch_20newsgroups

# categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
# twenty_train = fetch_20newsgroups(
#     subset="train", categories=categories, shuffle=True, random_state=42
# )
# twenty_test = fetch_20newsgroups(
#     subset="test", categories=categories, shuffle=True, random_state=42
# )

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline

# text_clf = Pipeline(
#     [
#         ("vect", CountVectorizer()),
#         ("tfidf", TfidfTransformer()),
#         ("clf", MultinomialNB()),
#     ]
# )

# text_clf.fit(twenty_train.data, twenty_train.target)

# import numpy as np

# docs_test = twenty_test.data
# predicted = text_clf.predict(docs_test)
# print("multinomialBC accuracy ", np.mean(predicted == twenty_test.target))

# # training SVM classifier
# from sklearn.linear_model import SGDClassifier

# text_clf = Pipeline(
#     [
#         ("vect", CountVectorizer()),
#         ("tfidf", TfidfTransformer()),
#         (
#             "clf",
#             SGDClassifier(
#                 loss="hinge",
#                 penalty="l2",
#                 alpha=1e-3,
#                 random_state=42,
#                 max_iter=5,
#                 tol=None,
#             ),
#         ),
#     ]
# )
# text_clf.fit(twenty_train.data, twenty_train.target)
# predicted = text_clf.predict(docs_test)
# print("SVM accuracy ", np.mean(predicted == twenty_test.target))

# from sklearn import metrics

# print(
#     metrics.classification_report(
#         twenty_test.target, predicted, target_names=twenty_test.target_names
#     )
# )

# print(metrics.confusion_matrix(twenty_test.target, predicted))

# from sklearn.model_selection import GridSearchCV

# parameters = {
#     "vect__ngram_range": [(1, 1), (1, 2)],
#     "tfidf__use_idf": (True, False),
#     "clf__alpha": (1e-2, 1e-3),
# }

# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

# gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

# print(twenty_train.target_names[gs_clf.predict(["God is love"])[0]])

# print(gs_clf.best_score_)

# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
