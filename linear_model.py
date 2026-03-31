import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error 

#(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
dataset = datasets.load_diabetes()
#print(dataset.data)

diabetes_X = dataset.data
#print(diabetes_X)

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_Y_train = dataset.target[:-20]
diabetes_Y_test = dataset.target[-20:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predicted = model.predict(diabetes_X_test)
print("Mean Squared Error: %.2f" % mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

#With One Feature
# diabetes_X_train = diabetes_X_train[:, np.newaxis, 2]
# plt.scatter(diabetes_X_test, diabetes_Y_test, color='black')
# plt.plot(diabetes_X_test, diabetes_Y_predicted, color='blue', linewidth=3)

# plt.show()

# Mean Squared Error: 2548.07
# Weights:  [938.23786125]
# Intercept:  152.91886182616113

#With All Features
#diabetes_X = dataset.data
# Mean Squared Error: 2004.52
# Weights:  [ 3.06094248e-01 -2.37635570e+02  5.10538048e+02  3.27729878e+02
#  -8.14111926e+02  4.92799595e+02  1.02841240e+02  1.84603496e+02
#   7.43509388e+02  7.60966464e+01]
# Intercept:  152.76429169049118