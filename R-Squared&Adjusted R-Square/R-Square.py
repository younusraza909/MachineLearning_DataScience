# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:05:59 2020

@author: Raza
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv("Salary_DataSet.csv")

#Spliting Data Into X And Y

X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values

#Spliting data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0) 

#REGRESSION
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

y_predict=reg.predict(X_test)

#Plotting the graph Train Data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="blue")
plt.title("Salary Vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


#Plotting the graph Test Data
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,reg.predict(X_test),color="blue")
plt.title("Salary Vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#For R-Square
import statsmodels.api as sm
X1=sm.add_constant(X)
reg=sm.OLS(Y,X1).fit()
reg.summary()