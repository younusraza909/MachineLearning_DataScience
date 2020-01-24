# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("DataSet.csv")
X=data.iloc[:,:-1].values
Y=data.iloc[:,3].values

#For Handling Missing Values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#For dummy Variable  For X
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelEncoder_x=LabelEncoder()
X[:,0]=labelEncoder_x.fit_transform(X[:,0])
onehot_x=OneHotEncoder(categorical_features=[0])
X=onehot_x.fit_transform(X).toarray()

#train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Standarized_X
from sklearn.preprocessing import StandardScaler
stc_x=StandardScaler()
X_train=stc_x.fit_transform(X_train)
X_test=stc_x.fit_transform(X_test)


