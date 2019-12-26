#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 12:33:44 2019

@author: praveen
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# Read Dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# Encoding the Independent Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap
X = X[:,1:] # Regression model automatically takes care of it. No need to do it manually

#Spliting the dataset into Train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fit training data with MLR model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction on test set
y_pred = regressor.predict(X_test)

# Build optimal model using backward elimination
import statsmodels.api as sm

# make const column for b0
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1)

X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS =  sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS =  sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS =  sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS =  sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


