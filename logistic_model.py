import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()

#(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
#print(iris.data)

X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

probabilities = model.predict_proba(X_test)
print("Probabilities: ", probabilities)

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)
