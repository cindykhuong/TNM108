# Defining dataset
# Assigning features and label variables
# First Feature
weather = [
    "Sunny",
    "Sunny",
    "Overcast",
    "Rainy",
    "Rainy",
    "Rainy",
    "Overcast",
    "Sunny",
    "Sunny",
    "Rainy",
    "Sunny",
    "Overcast",
    "Overcast",
    "Rainy",
]
# Second Feature
temp = [
    "Hot",
    "Hot",
    "Hot",
    "Mild",
    "Cool",
    "Cool",
    "Cool",
    "Mild",
    "Cool",
    "Mild",
    "Mild",
    "Mild",
    "Hot",
    "Mild",
]
# Label or target variable
play = [
    "No",
    "No",
    "Yes",
    "Yes",
    "Yes",
    "No",
    "Yes",
    "No",
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "No",
]

# In order to encode this data, you could map each value to a number Overcast:0, Rainy:1, Sunny:2
# This process is known as label encoding.
# Import LabelEncoder
from sklearn import preprocessing

# creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers
weather_encoded = le.fit_transform(weather)
print(weather_encoded)

# Similarly, you can encode temperature and play
# converting string labels into numbers
temp_encoded = le.fit_transform(temp)
label = le.fit_transform(play)

# Combining Features
# You can combine multiple columns (or features) into a single set of data using the zip function

# combinig weather and temp into single listof tuples
features = list(zip(weather_encoded, temp_encoded))

# Build the KNN classifier model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training sets
model.fit(features, label)
# Predict Output
predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild
print("Prediction: ", predicted)

# KNN with Multiple Labels
# Here you will learn about KNN with multiple labels where each label corresponds to a different class.

# Import scikit-learn dataset library
from sklearn import datasets

# Load dataset
wine = datasets.load_wine()

# print the names of the features
print(wine.feature_names)
# print the label species(class_0, class_1, class_2)
print(wine.target_names)

# print the wine data (top 5 records)
print(wine.data[0:5])
# print the wine labels (0:Class_0, 1:Class_1, 2:Class_3)
print(wine.target)

# You can check the shape of the dataset.
# print data(feature)shape
print(wine.data.shape)
# print target(or label)shape
print(wine.target.shape)

# Splitting Data
# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3
)

# Let's build KNN classifier model for k=5.
from sklearn.neighbors import KNeighborsClassifier

# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
# Train the model using the training sets
knn.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = knn.predict(X_test)

# Let's estimate how accurately the classifier or model can predict the type of cultivars.
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy for k=5: ", metrics.accuracy_score(y_test, y_pred))

# Let's build KNN classifier model for k=7.
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Let's again estimate how accurately the classifier can predict the type of cultivars.
from sklearn import metrics

print("Accuracy for k=7: ", metrics.accuracy_score(y_test, y_pred))
