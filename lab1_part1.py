# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)


# print("***** Train_Set *****")
# print(train.head())
# print("\n")
# print("***** Test_Set *****")
# print(test.head())

# #  statistics of both the train and test DataFrames using pandas' describe()method.
# print("***** Train_Set *****")
# print(train.describe())

# # list the feature names for you:
# print(train.columns.values)

# # So we need to handle the missing values present in the data. Let's first see where the values missing are:
# # For the train set
# train.isna().head()
# # For the test set
# test.isna().head()

# # Let's get the total number of missing values in both datasets.
# print("*****In the train set*****")
# print(train.isna().sum())
# print("\n")
# print("*****In the test set*****")
# print(test.isna().sum())

# Mean Imputation - Fixing missing data
# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(), inplace=True)

# Let's see if you have any missing values in the training dataset.
print(train.isna().sum())

# Let's see if you have any missing values in the test set.
print(test.isna().sum())

# Ticket is a mix of numeric and alphanumeric data types. Cabin is
# alphanumeric. Let see some sample values.
train["Ticket"].head()
train["Cabin"].head()

# Survival count with respect to Pclass:
train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(
    by="Survived", ascending=False
)
# Survival count with respect to Sex:
train[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(
    by="Survived", ascending=False
)
# Survival count with respect to SibSp:
train[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(
    by="Survived", ascending=False
)

# Let's first plot the graph of "Age vs. Survived":
g = sns.FacetGrid(train, col="Survived")
g.map(plt.hist, "Age", bins=20)
plt.show()

# See how Pclass and Survived features are related to each other with a graph:
grid = sns.FacetGrid(train, col="Survived", row="Pclass", aspect=1.6)
grid.map(plt.hist, "Age", alpha=0.5, bins=20)
grid.add_legend()
plt.show()

# data types of different features
train.info()

# Dropping part
train = train.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)
test = test.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)

# 'Sex' feature to a numerical one (only 'Sex'
# is remaining now which is a non-numeric feature).
# Using a technique called Label Encoding.
labelEncoder = LabelEncoder()
labelEncoder.fit(train["Sex"])
labelEncoder.fit(test["Sex"])
train["Sex"] = labelEncoder.transform(train["Sex"])
test["Sex"] = labelEncoder.transform(test["Sex"])

# Let's investigate if you have non-numeric data left.
train.info()

# Train your K-Means model
X = np.array(train.drop(["Survived"], 1).astype(float))
y = np.array(train["Survived"])

# # review all the features you are going to feed to the algorithm
# train.info()

# Let's now build the K-Means model. You want cluster the passenger records into 2: Survived
# or Not survived.
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
KMeans(
    algorithm="lloyd",
    copy_x=True,
    init="k-means++",
    max_iter=300,
    n_clusters=2,
    n_init=10,
    random_state=None,
    tol=0.0001,
    verbose=0,
)

# see how well the model is doing by looking at the percentage of passenger
# records that were clustered correctly
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct / len(X))

# tweak the values of these parameters and see if there is a change in the result.
kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm="lloyd")
kmeans.fit(X)
KMeans(
    algorithm="lloyd",
    copy_x=True,
    init="k-means++",
    max_iter=600,
    n_clusters=2,
    n_init=10,
    random_state=None,
    tol=0.0001,
    verbose=0,
)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct / len(X))

# scale the values of the features to a same range
# [0-1] as the range interval across all the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
KMeans(
    algorithm="lloyd",
    copy_x=True,
    init="k-means++",
    max_iter=600,
    n_clusters=2,
    n_init=10,
    random_state=None,
    tol=0.0001,
    verbose=0,
)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct / len(X))
