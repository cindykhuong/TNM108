# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.utils import shuffle
import pickle

# Load your dataset
data = pd.read_csv("IMDB Dataset.csv")
# Select the relevant columns
data = data[['review', 'sentiment']]
# Drop any missing values
data = data.dropna()
# Shuffle the data
data = shuffle(data, random_state=42)

# Split the data into training and testing sets
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Create a pipeline with CountVectorizer, TfidfTransformer, and/or SGD, LR
sentiment_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-7, random_state=42, max_iter=500, tol=None)),
    # ('clf', LogisticRegression(solver='liblinear', random_state=42, multi_class='ovr')),
])
# Train the model
sentiment_pipeline.fit(train_data['review'], train_data['sentiment'])

# Save the trained model to a .sav file
# filename = 'sentimentModelLR.sav'
filename = 'sentimentModelSGD_LR.sav'
pickle.dump(sentiment_pipeline, open(filename, 'wb'))
