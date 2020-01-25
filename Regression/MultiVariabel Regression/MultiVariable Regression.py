# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:51:16 2020

@author: Raza
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("M_Regression.csv")
X=data.iloc[:,:-1].values
Y=data.iloc[:,3].values

#splitting data in Train Test
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0) 

#MultiVariable Regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
y_predict=reg.predict(X_test)