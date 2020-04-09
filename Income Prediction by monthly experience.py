#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the given dataset
data = pd.read_csv('E:\\python\\assignment 2\\monthlyexp vs incom.csv')

x = data.iloc[:, 0:1].values
y = data.iloc[:, 1].values

#Dividing dataset of first state into training and test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Fitting Polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

# Visualising the Scattered dataset
plt.scatter(x_train, y_train, color = 'red')
plt.title('Months Experience VS Income (Training set)')
plt.xlabel('Months Experience')
plt.ylabel('Income')
plt.show()


#Visualizing the Polynomial Regression Result
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x, lin_reg.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Months Experience VS Income (Training set)')
plt.xlabel('Months Experience')
plt.ylabel('Income')
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