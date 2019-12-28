#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 22:31:01 2019

@author: praveen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, -1]

# Fit linear regression to the dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X,y)

# Fit Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_reg = PolynomialFeatures(degree=4)
X_polynom = polynomial_reg.fit_transform(X)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_polynom, y)

# Visualizing linear regression
plt.scatter(X, y, c = 'green')
plt.plot(X, linear_reg.predict(X), c = 'red')
plt.title('Position vs Salary (Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

# Visualizing polynomial regression
X = np.asarray(X)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, c ='green')
plt.plot(X_grid, linear_reg2.predict(polynomial_reg.fit_transform(X_grid)), c = 'red')
plt.title('Position vs Salary (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

# Predictions 
y_pred = linear_reg.predict([[6.5]])

y_pred = linear_reg2.predict(polynomial_reg.fit_transform([[6.5]]))

