# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:23:24 2020

@author: ASUS
"""
# Import Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMport dataset
dataset = pd.read_csv("Salary_data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Spliting the dataset into trainnig test and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Simple Linear Regression to the Trainnig set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)

# Predicting the test set result
y_pred = regressor.predict(x_test)

#Visulising the Test set results
plt.scatter(x_test, y_test, color = 'red')
# Y-cordinate
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
