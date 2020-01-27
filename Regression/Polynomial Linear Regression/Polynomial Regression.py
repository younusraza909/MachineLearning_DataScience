# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:31:00 2020

@author: Raza
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Loading Data Set
data=pd.read_csv("Poly_dataSet.csv")

#Splitting Data 
X=data.iloc[:,0:1].values
Y=data.iloc[:,1].values

#Regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)

#Visualization
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title(" Polynomial Regression Salary Vs Experience")
plt.xlabel("Positon Of Employee")
plt.ylabel("Salary")
plt.show()


#Plotting the graph Test Data
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg_2.predict(X_poly),color="blue")
plt.title("Polynomial Salary Vs Experience")
plt.xlabel("Positon Of Employee")
plt.ylabel("Salary")
plt.show()

#Prediction
lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))