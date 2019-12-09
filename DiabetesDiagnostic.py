#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 00:44:28 2019
@author: andre
"""

import pandas as pd
import numpy as np

#use block below if you want user input all in one line
"""
intake = input("Please enter your number of pregnancies, glucose levels, diastolic blood pressure (mm Hg), triceps skin fold thickness (mm), 2-Hour serum insulin (mu U/ml), Body Mass Index, diabetes pedigree function, and age (years) \n" )
indata = np.array(intake.split(","))
indata = indata.astype(np.float64)
indata = indata.reshape(1,-1)
"""

#use block below if you want user to promt user with multiple questions

indata=np.zeros((8,))
intake = input("Please enter your number of pregnancies \n")
indata[0] = intake
intake = input("Please enter your blood-glucose levels \n")
indata[1]=intake
intake = input("Please enter your diastolic blood pressure (mm Hg) \n")
indata[2]=intake
intake = input("Please enter your triceps skin fold thickness (mm) \n")
indata[3]=intake
intake = input("Please enter your 2-Hour serum insulin (mu U/ml) \n")
indata[4]=intake
intake = input("Please enter your Body Mass Index, \n")
indata[5]=intake
intake = input("Please enter your diabetes pedigree function \n")
indata[6]=intake
intake = input("Please enter your age (years) \n")
indata[7]=intake
indata = indata.astype(np.float64)
indata = indata.reshape(1,-1)

#begin training AI
data = pd.read_csv('/Users/andre/Documents/diabetes_data.csv')
X = data.drop(columns = ['diabetes'])
y = data['diabetes'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.4,random_state=42)
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(X_train, y_train)
#end training AI
y_pred = lg.predict(X_test)
print("\n L O G I S T I C")
print(classification_report(y_test,y_pred))
classes = {0:"It is unlikely you have diabetes", 1:"It is likely you have diabetes"}
diagnosis=lg.predict(indata)
print(classes[diagnosis[0]])
#End of program


#Helpful testing example
"""
newin = [[6,148,72,35,0,33.6,0.627,50],
         [1,85,66,29,0,26.6,0.35100000000000003,31],
         [8,183,64,0,0,23.3,0.672,32],
        [1,89,66,23,94,28.1,0.16699999999999998,21],
         [0,137,40,35,168,43.1,2.2880000000000003,33]]
"""


#ALTERNATE, BUT LESS PRECISE AI MODELS BELOW

#Decision Tree Model(74% precision)
"""
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc.fit(X_train,y_train)
dtc.score(X_test, y_test)
y_pred = dtc.predict(X_test)
print("\n D E C I S I O N  T R E E")
print(classification_report(y_test,y_pred))
"""


#KNN Model(71% precision)
"""
from sklearn.metrics import confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
#end training AI
#print(knn.score(X_test,y_test))
result = knn.predict(indata)
print(classes[result[0]])

#Diagnostic Tools
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
print(data.dtypes)
"""
