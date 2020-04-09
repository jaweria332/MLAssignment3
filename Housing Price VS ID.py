#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the dataset
data = pd.read_csv('E:\\python\\assignment 2\\housing price.csv')

#Seting values to x and y
x = data.iloc[:, 0:1].values
y = data.iloc[:, 1].values

#Diving data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the result
y_pred = regressor.predict([[6.5]])

#Visualizing scattered dataset
plt.figure(figsize = (10,7))
plt.scatter(x_train, y_train, color = 'red')
plt.title('Predicting Price using ID')
plt.xlabel('ID')
plt.ylabel('House Price')
plt.show()

#Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures (degree = 4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

#Visualizing polynomial Regression
plt.figure(figsize = (10,7))
plt.scatter(x_train, y_train, color ='red')
plt.plot(x,lin_reg.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Predicting Price using ID')
plt.xlabel('ID')
plt.ylabel('House Price')
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
r_score = reg.score(x, y)
Accuracy = r_score * 100
print("Accuracy = " + str(Accuracy))