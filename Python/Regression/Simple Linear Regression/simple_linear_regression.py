# Simple Linear Regression

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# Read Dataset
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


#Spliting the dataset into Train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fit the model with training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict test data
y_pred = regressor.predict(X_test)

# Visualizing Trained data
plt.scatter(X_train,y_train, c = 'red')
plt.plot(X_train, regressor.predict(X_train), c = 'blue')
plt.title('Salary vs Year of Experience(Training data)')
plt.xlabel('Salary')
plt.ylabel('Years of Experience')
plt.show()


# Visualizing Predicted data
plt.scatter(X_test,y_test, c = 'red')
plt.plot(X_train, regressor.predict(X_train), c = 'blue')
plt.title('Salary vs Year of Experience (Test Data)')
plt.xlabel('Salary')
plt.ylabel('Years of Experience')
plt.show()
