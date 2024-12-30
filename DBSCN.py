import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
df = pd.read_csv('Clustering.csv')

# Drop the 'Gender' column as it is not needed for clustering
df = df.drop("Gender", axis=1)

# Convert the DataFrame to a NumPy array and handle NaN values
x = df.values[:, 1:]
x = np.nan_to_num(x)

# Standardize the dataset using StandardScaler
X = StandardScaler().fit_transform(x)

# Set parameters for DBSCAN
epsilon = 0.5  # Maximum distance between two samples for them to be considered as in the same neighborhood
minimum_samples = 5  # Minimum number of samples in a neighborhood for a point to be considered a core point

# Fit the DBSCAN clustering algorithm to the standardized data
db = DBSCAN(eps=epsilon, min_samples=minimum_samples).fit(X)

# Extract cluster labels assigned by DBSCAN
labels = db.labels_

# Create a mask for core samples
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Calculate the number of clusters (excluding noise)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Extract unique labels
unique_labels = set(labels)

# Generate colors for plotting clusters
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

# Plot each cluster and the noise
for k, col in zip(unique_labels, colors):
    if k == -1:  # Black color for noise
        col = "k"
    
    # Get the points that belong to the current cluster
    class_member_mask = (labels == k)
    
    # Plot core samples
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], label=f'Cluster {k}' if k != -1 else 'Noise')
    
    # Plot outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=5, c=[col])

# Show the plot with a legend
plt.title(f'DBSCAN Clustering (Estimated number of clusters: {n_clusters})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
