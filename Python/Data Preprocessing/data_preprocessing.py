import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# Read Dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


# Replace missing values with mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy="mean", axis=0) # Mean of the collumn
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
 
 
# Encoding categorical Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onhotencoder = OneHotEncoder(categorical_features=[0])
