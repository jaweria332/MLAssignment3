#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the given dataset
dataset = pd.read_csv('E:\\python\\assignment 2\\global_co2.csv')
data = dataset[dataset.Year > 1969]
#Selecting Rows and columns
x = data.iloc[:, :1].values
y = data.iloc[:, 1].values

#Dividing dataset of first state into training and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#Fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Fitting Polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

#Visualizing the Polynomial Regression Result
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x, lin_reg.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Production of Global CO2 for the next few years')
plt.xlabel('Year')
plt.ylabel('Global CO2 Production')
plt.show()

#To chek accuracy of output using scikit learn Python library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#cannot use Rank 1 matrix in scikit learn
x = x.reshape((len(x), 1))              

#creating model
reg = LinearRegression()

#Fitting training data
reg = reg.fit(x, y)

#Y prediction
Y_pred = reg.predict (x)

#calculating RMSE and R2 Score
mse = mean_squared_error(y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(x, y)
Accuracy = r2_score * 100
print("Accuracy = " + str(Accuracy))