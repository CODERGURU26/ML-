import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_iris()

X = data.data
y = data.target

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test , y_pred))

print("Feature Importances: ", model.feature_importances_)

print("Classes: ", model.classes_)

print("Weights: ", model.tree_.feature)

print("Intercept: ", model.tree_.threshold)

plt.figure(figsize=(10,6))
plot_tree(model, filled=True)
plt.show()