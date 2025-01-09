# K-Means Clustering

## Overview
K-Means Clustering is an unsupervised machine learning algorithm that partitions data into K distinct, non-overlapping clusters based on feature similarity. The algorithm assigns data points to clusters by minimizing the within-cluster variance (inertia) through an iterative process.

---

## Objective
The goal of K-Means is to group similar data points together while keeping different groups as separate as possible. Each cluster is defined by its centroid (center point), and points are assigned to the cluster with the nearest centroid.

---

## Key Concepts

1. **Centroids**:
   - Center points of clusters
   - Initially placed randomly
   - Updated iteratively based on mean of points

2. **Distance Metrics**:
   - Euclidean distance (most common)
   - Manhattan distance
   - Cosine similarity

3. **Algorithm Steps**:
   - Initialization
   - Assignment
   - Update
   - Convergence

4. **Important Parameters**:
   - Number of clusters (k)
   - Maximum iterations
   - Initialization method
   - Tolerance level

---

## Advantages and Disadvantages

### Advantages
1. Simple to understand and implement
2. Scales well to large datasets
3. Guarantees convergence
4. Easily adapts to new examples
5. Generalizes to clusters of different shapes

### Disadvantages
1. Requires pre-specified number of clusters
2. Sensitive to initial centroid positions
3. Not suitable for non-convex clusters
4. Sensitive to outliers
5. Can converge to local optima

---

## Steps to Implement K-Means

1. **Data Preprocessing**:
   - Feature scaling
   - Handle missing values
   - Dimensionality reduction (if needed)

2. **Model Configuration**:
   - Choose number of clusters (k)
   - Set initialization method
   - Define convergence criteria

3. **Training Process**:
   - Initialize centroids
   - Assign points to nearest centroid
   - Update centroid positions
   - Repeat until convergence

4. **Evaluation**:
   - Inertia (within-cluster sum of squares)
   - Silhouette score
   - Calinski-Harabasz index
   - Davies-Bouldin index

---

## Python Implementation
Here's a basic example using scikit-learn:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Get cluster assignments
labels = kmeans.labels_

# Get cluster centers
centers = kmeans.cluster_centers_

# Evaluate model
print("Inertia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Predict new points
new_points = np.array([[2, 2], [5, 3]])
predictions = kmeans.predict(scaler.transform(new_points))