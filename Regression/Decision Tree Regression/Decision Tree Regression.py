# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:47:44 2020

@author: Raza
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Loading Data Set
data=pd.read_csv("DTR.csv")

#Splitting Data 
X=data.iloc[:,0:1].values
Y=data.iloc[:,1].values

#Regression
from sklearn.tree import DecisionTreeRegressor
DTR_reg=DecisionTreeRegressor()
DTR_reg.fit(X,Y)

#Visualization
plt.scatter(X,Y,color="red")
plt.plot(X,DTR_reg.predict(X),color="blue")
plt.title(" DTR MODEL Position Vs Salary")
plt.xlabel("Positon Of Employee")
plt.ylabel("Salary")
plt.show()

#Changing a littile bit to more prominently visualize data
x_grid=np.arange(min(X),max(X),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(X,Y,color="red")
plt.plot(x_grid,DTR_reg.predict(x_grid),color="blue")
plt.title(" DTR MODEL Position Vs Salary")
plt.xlabel("Positon Of Employee")
plt.ylabel("Salary")
plt.show()



#Prediction
DTR_reg.predict([[6.5]])

