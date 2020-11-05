# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:08:42 2020

@author: SUCHARITA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler as sc
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE  # "pip install imblearn" through anaconda prompt
s_train = pd.read_csv("F:\\ExcelR\\Assignment\SVM\\SalaryData_Train(1).csv")
s_test = pd.read_csv("F:\\ExcelR\\Assignment\SVM\\SalaryData_Test(1).csv")

# data understanding
s_train.info() # 5 int, 9 obj
s_train.shape #(30161, 14)
s_train["Salary"].unique() #[' <=50K', ' >50K']
s_train.describe() 
s_train.isnull().sum() # no null values
s_train.skew()
s_train.kurtosis()
plt.hist(s_train['Salary']) # approx 7500 ppl earn >50k, while rest earn <50k, imbalanced data set
s_train.Salary.value_counts().plot(kind="pie") # ratio of 1:3 for o/p variable

# identify categorical variables and label encode them

categorical = [var for var in s_train.columns if s_train[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)
string_values = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native', 'Salary']
s_train[categorical].isnull().sum() # shows no null values

# label encoding of all string values
number= preprocessing.LabelEncoder()
for i in string_values:
   s_train[i] = number.fit_transform(s_train[i])
   s_test[i] = number.fit_transform(s_test[i])
   
s_train["Salary"].unique()   
s_train.Salary.value_counts() # 0 :22653, 1:7508
s_test["Salary"].unique()   
s_test.Salary.value_counts() # 0 :11360, 1:3700

x_train = s_train.iloc[:,0:13]
y_train = s_train["Salary"] # series format

x_test = s_test.iloc[:,0:13]
y_test = s_test["Salary"]


# imbalanced dataset management
  
sm = SMOTE(random_state = 2) 
x_train1, y_train1 = sm.fit_sample(x_train, y_train)  
print('After OverSampling, the shape of x_train: {}'.format(x_train1.shape)) #(45306, 13)
print('After OverSampling, the shape of y_train: {} \n'.format(y_train1.shape)) #(45306, 1)
y_train1.value_counts() # 1:22653     0:22653

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

#kernel = linear
# create model
model_linear= SVC()
model_linear.fit(x_train1, y_train1)

#test model 
pred_test_linear = model_linear.predict(x_test)
np.mean(pred_test_linear == y_test)  # 79.28%

# kernel = poly
# create model
model_poly= SVC(kernel = "poly")
model_poly.fit(x_train1, y_train1)

#test model 
pred_test_poly = model_poly.predict(x_test)
np.mean(pred_test_poly == y_test)  # 79.45%

# kernel = rbf
# create model
model_rbf= SVC(kernel = "rbf")
model_rbf.fit(x_train1, y_train1)

#test model 
pred_test_rbf = model_rbf.predict(x_test)
np.mean(pred_test_rbf == y_test)  # 79.28%
# kernel = sigmoid
# create model
model_sigmoid= SVC(kernel = "sigmoid")
model_sigmoid.fit(x_train1, y_train1)

#test model 
pred_test_sigmoid = model_sigmoid.predict(x_test)
np.mean(pred_test_sigmoid == y_test)  # 76.87%

# checking accuracy of teh model with highest predicted test and real test avlue match, here its "poly kernel"


print(confusion_matrix(y_test, pred_test_poly))
#[[11330     30]
# [ 3064   636]]
print(classification_report(y_test, pred_test_poly))   
#                  precision   recall  f1-score   support

#           0       0.79      1.00      0.88     11360
#           1       0.95      0.17      0.29      3700

#    accuracy                           0.79     15060
#   macro avg       0.87      0.58      0.59     15060
#weighted avg       0.83      0.79      0.74     15060
   
   