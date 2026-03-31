import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split

data = load_iris()

x=data.data
y=data.target

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(x_train , y_train)

y_pred  = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test , y_pred))    

print("Feature importances: ", model.feature_importances_)

