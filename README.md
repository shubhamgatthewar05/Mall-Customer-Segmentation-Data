
# Customer Segmentation Using K-Means Clustering

## Overview

This project performs customer segmentation using K-Means clustering on a dataset of mall customers. The dataset includes information about customers' annual income and spending score. The goal is to identify distinct customer segments to better understand their behavior and preferences.

## Dataset

The dataset used is `Mall_Customers.csv`, which contains the following columns:
- `CustomerID`: Unique identifier for each customer
- `Gender`: Gender of the customer
- `Age`: Age of the customer
- `Annual Income (k$)`: Annual income in thousands of dollars
- `Spending Score (1-100)`: Spending score assigned by the mall

## Steps

1. **Load the Dataset**:
   Load the dataset using Pandas.

2. **Extract Features**:
   Extract relevant features for clustering: Annual Income and Spending Score.

3. **Determine Optimal Number of Clusters**:
   Use the Elbow Method to determine the optimal number of clusters.

4. **Cluster Data**:
   Apply K-Means clustering with the determined number of clusters.

5. **Evaluate Clusters**:
   Compute the Silhouette Score and plot the silhouette scores for each cluster.

6. **Analyze Clusters**:
   Analyze the average values of features for each cluster.

## Dependencies

Ensure you have the following libraries installed:

- `pandas`
- `matplotlib`
- `scikit-learn`
- `numpy`

You can install the required libraries using pip:

```bash
pip install pandas matplotlib scikit-learn numpy
```

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import matplotlib.cm as cm

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Extract the relevant features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Determine the optimal number of clusters using the elbow method (optional)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow method results (optional)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Assuming the elbow method suggests 5 clusters
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Add the cluster labels to the original DataFrame
df['Cluster'] = y_kmeans

# Compute the Silhouette score
silhouette_avg = silhouette_score(X, df['Cluster'])
sample_silhouette_values = silhouette_samples(X, df['Cluster'])

# Create a subplot with 1 row and 1 column
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(18, 7)

# The silhouette coefficient range is [-0.1, 1]
ax1.set_xlim([-0.1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters
ax1.set_ylim([0, len(X) + (kmeans.n_clusters + 1) * 10])

y_lower = 10
for i in range(kmeans.n_clusters):
    # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[df['Cluster'] == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / kmeans.n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the y-axis labels / ticks
ax1.set_xticks([i/10.0 for i in range(-1, 11)])

plt.show()

# Analyze each cluster
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)
```

## Results

The Elbow Method suggests an optimal number of clusters, and K-Means clustering is applied with that number. The silhouette plot helps evaluate the quality of the clusters, and cluster summaries provide insights into the characteristics of each cluster.



Feel free to modify any sections as needed to better fit your project!
