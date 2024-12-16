# K-Nearest Neighbors (KNN)

## Overview
K-Nearest Neighbors (KNN) is a simple, versatile, and intuitive machine learning algorithm used for both classification and regression tasks. It makes predictions based on the similarity (distance) between the input data point and all training instances, using the k closest training examples to make a prediction.

---

## Objective
The goal of KNN is to classify new data points based on the majority class of their k nearest neighbors or predict values based on the average of the k nearest neighbors' values. The choice of k is crucial as it directly impacts the model's performance and susceptibility to noise.

---

## Key Concepts
1. **Distance Metric**: Commonly used distance measures include:
   - Euclidean Distance (most common)
   - Manhattan Distance
   - Minkowski Distance
   - Hamming Distance (for categorical variables)

2. **K Value Selection**:
   - Small k: More sensitive to noise
   - Large k: Smoother decision boundaries but may oversmooth
   - k should be odd (for classification) to avoid ties
   - Common practice: k = √n, where n is the number of samples

3. **Feature Scaling**:
   - Essential preprocessing step
   - Ensures all features contribute equally to distance calculations
   - Common methods: StandardScaler, MinMaxScaler

---

## Advantages and Disadvantages

### Advantages
1. Simple to understand and implement
2. No training phase (lazy learner)
3. Naturally handles multi-class problems
4. No assumptions about data distribution
5. Can be used for both regression and classification

### Disadvantages
1. Computationally expensive for large datasets
2. Sensitive to irrelevant features
3. Requires feature scaling
4. Memory-intensive (stores all training data)
5. Susceptible to imbalanced data

---

## Steps to Implement KNN

1. **Data Preprocessing**:
   - Handle missing values
   - Feature scaling
   - Feature selection/dimensionality reduction
   
2. **Model Configuration**:
   - Choose k value
   - Select distance metric
   - Define voting method (for classification)

3. **Training**:
   - Store training data (no actual training phase)

4. **Prediction Process**:
   - Calculate distances to all training instances
   - Select k nearest neighbors
   - For classification: majority voting
   - For regression: average of neighbor values

5. **Evaluation**:
   - Classification metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - Regression metrics:
     - MSE
     - RMSE
     - MAE

---

## Python Implementation
Here's a basic example using scikit-learn:

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))