#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 20:36:51 2019

@author: praveen
"""

# Decision Tree Regression
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

y = y.reshape((10,1))

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X, sc_y = StandardScaler(), StandardScaler()
# X, y = sc_X.fit_transform(X), sc_y.fit_transform(y)

# Fit decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)


# Visualizing Decision Tree regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, c ='green')
plt.plot(X_grid, regressor.predict(X_grid), c = 'red')
plt.title('Position vs Salary (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

# Predictions 
y_pred = regressor.predict([[6.5]])

