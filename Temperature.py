#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the given dataset
data = pd.read_csv('E:\\python\\assignment 2\\annual_temp.csv')

x = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

#Separating temperatures of both states 
data1 = data[data.Source == 'GCAG']
data2 = data[data.Source == 'GISTEMP']

#Selecting Rows and columns
x1 = data1.iloc[:, 1:2].values
y1 = data1.iloc[:, 2].values

x2 = data2.iloc[:, 1:2].values
y2 = data2.iloc[:, 2].values

#Dividing dataset of first state into training and test 
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2, random_state = 0)

#Dividing dataset of second state into training and test 
from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size = 0.2, random_state = 0)

#Fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x1_train, y1_train)

#Fitting Polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
x1_poly = poly_reg.fit_transform(x1)
poly_reg.fit(x1_poly, y1)
lin_reg = LinearRegression()
lin_reg.fit(x1_poly, y1)

#Visualizing the Polynomial Regression Result
plt.scatter(x1_train, y1_train, color = 'red')
plt.plot(x1, lin_reg.predict(poly_reg.fit_transform(x1)), color = 'blue')
plt.title('Temperature of first state')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()

#To chek accuracy of output using scikit learn Python library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#cannot use Rank 1 matrix in scikit learn
x1 = x1.reshape((len(x1), 1))              

#creating model
reg = LinearRegression()

#Fitting training data
reg = reg.fit(x1, y1)

#Y prediction
Y1_pred = reg.predict (x1)

#calculating RMSE and R2 Score
mse = mean_squared_error(y1, Y1_pred)
rmse = np.sqrt(mse)
r1_score = reg.score(x1, y1)
Accuracy = r1_score * 100
print("Accuracy = " + str(Accuracy))

#Fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()
regressor2.fit(x2_train, y2_train)

#Fitting Polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg_2 = PolynomialFeatures(degree = 6)
x2_poly = poly_reg_2.fit_transform(x2)
poly_reg_2.fit(x2_poly, y2)
lin_reg2 = LinearRegression()
lin_reg2.fit(x2_poly, y2)

#Visualizing the Polynomial Regression Result
plt.scatter(x2_train, y2_train, color = 'orange')
plt.plot(x2, lin_reg2.predict(poly_reg_2.fit_transform(x2)), color = 'purple')
plt.title('Temperature of Second State')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()

#To chek accuracy of output using scikit learn Python library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#cannot use Rank 1 matrix in scikit learn
x2 = x2.reshape((len(x2), 1))              

#creating model
reg2 = LinearRegression()

#Fitting training data
reg2 = reg2.fit(x2, y2)

#Y prediction
Y2_pred = reg.predict (x2)

#calculating RMSE and R2 Score
mse = mean_squared_error(y2, Y2_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(x2, y2)
Accuracy = r2_score * 100
print("Accuracy = " + str(Accuracy))

#Visualizing result in single graph for comparison
plt.scatter(x1_train, y1_train, color = 'red')
plt.plot(x1, lin_reg.predict(poly_reg.fit_transform(x1)), color = 'blue')
plt.scatter(x2_train, y2_train, color = 'orange')
plt.plot(x2, lin_reg2.predict(poly_reg_2.fit_transform(x2)), color = 'purple')
plt.title('Temperature of two states')
plt.xlabel('Year')
plt.ylabel('Temperature')
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
print("Overall Accuracy = " + str(Accuracy))