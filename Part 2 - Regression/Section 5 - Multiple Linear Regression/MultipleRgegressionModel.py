# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:35:08 2020

@author: ASUS
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMporting the dataset
dataset = pd.read_csv('50_startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding the dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Avoiding the Dummy Variable Trap
x = x[:, 1:]

# Splitting the dataset into test set and trainning set
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Trainnig set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(x_test)  
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Building the optimal model using backward elimation
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int) , values = x, axis = 1)