import pandas as pd
import pickle

# Load the trained sentiment analysis model from the .sav file
sentiment_model = pickle.load(open('sentimentModelSGD_LR.sav', 'rb'))

# Read the CSV file with reviews and sentiments
reviews_df = pd.read_csv('input/parasite.csv')

# Assuming your CSV file has a 'review' column
reviews = reviews_df['review']

# Predict sentiment for the reviews
predicted_sentiments = sentiment_model.predict(reviews)

# Add predicted sentiments to the DataFrame
reviews_df['predicted_sentiment'] = predicted_sentiments

# Display the DataFrame with predicted sentiments
# print(reviews_df)

# Calculate the percentage of positive reviews
positive_reviews = reviews_df[reviews_df['predicted_sentiment'] == 'positive']
positive_percentage = (len(positive_reviews) / len(reviews_df)) * 100


print("Review Sentiment Analysis Results:")
print("+----------------------+")
print(f"| Positive reviews    | {len(positive_reviews)}")
print(f"| Negative reviews    | {len(reviews_df) - len(positive_reviews)}")
print("+----------------------+")
print(f"| IMDB Score Prediction | {positive_percentage}%")
print("+----------------------+")

