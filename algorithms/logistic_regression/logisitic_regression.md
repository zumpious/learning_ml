# Logistic Regression

Logistic Regression is a statistical method for analyzing datasets in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

## Key Concepts

- **Binary Classification**: Logistic regression is used for binary classification problems where the output is a binary variable (0 or 1, true or false, yes or no).
- **Sigmoid Function**: The logistic function, also known as the sigmoid function, is used to map predicted values to probabilities.

## Applications

- **Medical Field**: Predicting the presence or absence of a disease.
- **Finance**: Credit scoring to determine the likelihood of a borrower defaulting on a loan.
- **Marketing**: Predicting whether a customer will buy a product or not.

## Advantages

- Simple to implement and interpret.
- Provides probabilities and insights into the importance of features.
- Works well with linearly separable data.

## Disadvantages

- Assumes linearity between the independent variables and the log-odds.
- Not suitable for complex relationships between variables.
- Sensitive to outliers.

## Implementation in Python

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Create logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

Logistic Regression is a powerful tool for binary classification problems, providing a probabilistic framework for understanding the relationship between the dependent and independent variables.