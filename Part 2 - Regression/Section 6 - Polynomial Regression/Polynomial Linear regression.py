
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


#Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Trainning the dataset 
from sklearn.linear_model import LinearRegression
# lin_reg is for linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Trainnig the polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 7)
x_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualizing the Linear Regression model on the whole dataset
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff(Linear Regression model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression Results.
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polonomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visulizing the Polynomial Regression (for higher resolution and smoother curve)
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(x_grid , lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or bluff (polynomial model)')
plt.xlabel('Postional level')
plt.ylabel('Salary')
plt.show()

# Predicting the new result with Linear Regressor
lin_reg.predict([[6.5]])

#Predicting the new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))