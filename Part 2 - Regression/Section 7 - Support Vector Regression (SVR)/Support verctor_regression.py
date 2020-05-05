# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:49:25 2020

@author: ASUS
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMporting the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(x)

print(y)
# Trainsforming the y into 2D array , becuse it needs to be in the same size
y = y.reshape(len(y), 1)
y

# apply the feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print(x)
print(y)

# Tainning the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Radial basis function
regressor.fit(x, y)

# Predicting the new result
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

# Visualizing the results of svr models

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('Support vector regression')
plt.xlabel('Position in the company')
plt.ylabel('Salary')
plt.show()

# Visulizing the SVR results(for higher resolution and smoohter curve)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x) , sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('SVR smother regression')
plt.xlabel('Position in the company')
plt.ylabel('Salary')
plt.show()
