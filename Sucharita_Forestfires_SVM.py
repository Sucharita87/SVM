# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:44:55 2020

@author: SUCHARITA
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

forest = pd.read_csv("F:\\ExcelR\\Assignment\\SVM\\forestfires.csv")
forest.shape
forest.info()
forest['size_category'].unique()
forest.head()
categorical = [var for var in forest.columns if forest[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)
forest[categorical].isnull().sum() # shows no null values

string_values= ["month", "day", "size_category"] 
# string values label encoded
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_values:
   forest[i] = number.fit_transform(forest[i])

forest.info()
forest['size_category'].value_counts() # 1:378 and 0 :139

train,test = train_test_split(forest, test_size = 0.3)
x_train = train.iloc[:,0:30]
y_train = train["size_category"] # series format

x_test = test.iloc[:,0:30]
y_test = test["size_category"] # series format

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

#kernel = linear
# create model
model_linear= SVC(kernel = "linear")
model_linear.fit(x_train, y_train)

#test model 
pred_test_linear = model_linear.predict(x_test)
np.mean(pred_test_linear == y_test)  # 98.7%

# kernel = poly
# create model
model_poly= SVC(kernel = "poly")
model_poly.fit(x_train, y_train)

#test model 
pred_test_poly = model_poly.predict(x_test)
np.mean(pred_test_poly == y_test)  # 75.6%

# kernel = rbf
# create model
model_rbf= SVC(kernel = "rbf")
model_rbf.fit(x_train, y_train)

#test model 
pred_test_rbf = model_rbf.predict(x_test)
np.mean(pred_test_rbf == y_test)  # 74.35%

# kernel = sigmoid
# create model
model_sigmoid= SVC(kernel = "sigmoid")
model_sigmoid.fit(x_train, y_train)

#test model 
pred_test_sigmoid = model_sigmoid.predict(x_test)
np.mean(pred_test_sigmoid == y_test)  # 73.71%

















