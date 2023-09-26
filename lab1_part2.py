# Dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# SHOPPING DATASET
customer_data = pd.read_csv("shopping_data.csv")
# check the number of records and attributes,
customer_data.shape

# Will return (200, 5) which means that the dataset contains 200 records and 5 attributes
# to see the data structure
customer_data.head()

# filter the first three columns from our dataset:
data = customer_data.iloc[:, 3:5].values

# -----------------------------------------------------------------
# Need to know the number of clusters that we want our data to be split to
# create the clusters
linked = linkage(data, "ward")
# linked = linkage(data, "single")


# scattered dots
labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(data[:, 0], data[:, 1], label="True Position")

for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(-3, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
    )
plt.title("Graph 1")
plt.show()

# number of clusters?
print(len(data))


# Create the dendogram
labelList = range(1, len(data) + 1)
plt.figure(figsize=(10, 7))
dendrogram(
    linked,
    orientation="top",
    labels=labelList,
    distance_sort="descending",
    show_leaf_counts=True,
)
plt.show()

# plot the cluster
cluster = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
cluster.fit_predict(data)

# Plot cluster
plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap="rainbow")
plt.show()

# x income and y spending
