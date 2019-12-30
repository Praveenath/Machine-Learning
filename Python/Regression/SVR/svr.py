#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:58:09 2019

@author: praveen
"""
# Support Vector Regression
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
from sklearn.preprocessing import StandardScaler
sc_X, sc_y = StandardScaler(), StandardScaler()
X, y = sc_X.fit_transform(X), sc_y.fit_transform(y)

# Fit linear regression to the dataset
from sklearn.svm import SVR
sv_regressor = SVR(kernel = 'rbf')
sv_regressor.fit(X,y)


# Visualizing polynomial regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, c ='green')
plt.plot(X_grid, sv_regressor.predict(X_grid), c = 'red')
plt.title('Position vs Salary (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

# Predictions 
y_pred = sc_y.inverse_transform(sv_regressor.predict(sc_X.transform([[6.5]])))