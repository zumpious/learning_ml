# Support Vector Machines (SVM)

## Overview
Support Vector Machines (SVM) is a powerful supervised learning algorithm used for both classification and regression tasks. It works by finding the optimal hyperplane that maximally separates different classes in the feature space, making it particularly effective for complex, high-dimensional problems.

---

## Objective
The main goal of SVM is to find a hyperplane that best divides the dataset into classes while maximizing the margin (the distance between the hyperplane and the nearest data points from each class). These nearest points are called support vectors, hence the algorithm's name.

---

## Key Concepts

1. **Hyperplane**: 
   - A decision boundary that separates different classes
   - For 2D space: a line
   - For 3D space: a plane
   - For higher dimensions: a hyperplane

2. **Support Vectors**:
   - Data points nearest to the hyperplane
   - Most critical elements in defining the hyperplane
   - Only these points affect the position of the hyperplane

3. **Kernel Functions**:
   - Linear
   - Polynomial
   - Radial Basis Function (RBF)
   - Sigmoid

4. **Margin**:
   - Distance between hyperplane and nearest data points
   - Can be soft (allowing some misclassification)
   - Can be hard (requiring perfect separation)

---

## Advantages and Disadvantages

### Advantages
1. Effective in high-dimensional spaces
2. Memory efficient (uses only subset of training points)
3. Versatile through different kernel functions
4. Robust against overfitting

### Disadvantages
1. Not suitable for large datasets (computationally intensive)
2. Sensitive to feature scaling
3. Choice of kernel and parameters can be complex
4. Limited interpretability compared to simpler models

---

## Steps to Implement SVM

1. **Data Preprocessing**:
   - Feature scaling (crucial for SVM)
   - Handling missing values
   - Converting categorical variables

2. **Model Configuration**:
   - Choose kernel type
   - Set hyperparameters (C, gamma, etc.)
   - Define kernel parameters

3. **Training**:
   - Fit model to training data
   - Optimize parameters using cross-validation

4. **Evaluation**:
   - Classification metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - Regression metrics (for SVR):
     - MSE
     - RMSE
     - MAE

---

## Python Implementation
Here's a basic example using scikit-learn:

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Sample data
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm.predict(X_test_scaled)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))