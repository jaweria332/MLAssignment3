#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer

#Reading the given dataset
data = pd.read_csv('E:\\python\\assignment 2\\50_Startups.csv')

#Retrieving startupsof two states only
dataset = data[(data.State == 'Florida') | (data.State == 'California')]

x = dataset.iloc[:, 0:3].values
y = dataset.iloc[: , 4].values

#Dividing into training and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting linear regression to data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test set result
y_pred = regressor.predict(x_test)

#To chek accuracy of output using scikit learn Python library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#creating model
reg = LinearRegression()

#Fitting training data
reg = reg.fit(x, y)

#Y prediction
Y_pred = reg.predict (x)

#calculating RMSE and R2 Score
mse = mean_squared_error(y, Y_pred)
rmse = np.sqrt(mse)
r_score = reg.score(x, y)
Accuracy = r_score * 100
print("Overall Accuracy = " + str(Accuracy))