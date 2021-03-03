# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:24:48 2019

@author: SARKAR
"""

import pandas as pd
import random
col_names = ['cell_id','glatlon','lat','lon','glat','glan','date','time']
# load dataset
pima = pd.read_csv('C:\\Users\SARKAR\Desktop\Dataset_tower_final.csv', header=None, names=col_names)

#print(pima.head())
def c():
    #provide seed to get the randomness
    a=metrics.accuracy_score(y_test, y_pred)
    return a
    #return random.randint(62,78)
# Import train_test_split functi
from sklearn.model_selection import train_test_split
# Split dataset into features and labels
print("XYZ")
X=pima[['cell_id','time']]  # after removing unimportant features
y=pima['glatlon'] #target variable
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=10) # 70% training and 30% test
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=200)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",c())#
print("after random forests, predicted label is:")
print(clf.predict([[804,15.08]]))
#print(clf.predict([[839]]))

