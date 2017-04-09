# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:49:08 2017

@author: rucavado
"""

import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["font.size"] = 14
from sklearn.utils import check_random_state

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

labels = ["b", "r"]
X, y = make_blobs(n_samples=400, centers=42, random_state=42)
y = np.take(labels, (y < 10))

#QUESTION5 1
#Plot your dataset

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X, y)

plt.scatter(X[:,0], X[:, 1], c="b", lw=0, s=40)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")


#split your dataset into a training and testing set. Comment on how you decided to split your data.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)

# Based on  the total number of observations (n_samples = 400), a ratio of 70/30 gives enough ovservation in the dataset to predict and to test the accuracy of the classification. 
#To asses how accurate a train/test set is, I used the accuraty_score function, which gives as output of the number of times the  
                                             
from sklearn.metrics import accuracy_score
                                             
clf.fit(X_train, y_train) #fitting the model for the training set
pred_test = clf.predict(X_test) #predict the response
print(accuracy_score(y_test,pred_test)) #accuracy of test data

                                             
clf.fit(X_train, y_train) #fitting the model for the train set
pred_train = clf.predict(X_train) #predict the response
print(accuracy_score(y_train,pred_train)) #accuracy of train data

#now we do a model with train_size of 0.90     

X_train90, X_test90, y_train90, y_test90 = train_test_split(X, y, train_size=0.90)
clf.fit(X_train90, y_train90) #fitting the model for the training set
pred_test90 = clf.predict(X_test90) #predict the response
print(accuracy_score(y_test90,pred_test90)) #accuracy of test data

#now we do a model with train_size of 0.80     

X_train80, X_test80, y_train80, y_test80 = train_test_split(X, y, train_size=0.80)
clf.fit(X_train80, y_train80) #fitting the model for the training set
pred_test80 = clf.predict(X_test80) #predict the response
print(accuracy_score(y_test80,pred_test80)) #accuracy of test data
                                            
clf.fit(X_train80, y_train80) #fitting the model for the train set
pred_train80 = clf.predict(X_train80) #predict the response
print(accuracy_score(y_train80,pred_train80))

#now we do a model with train_size of 0.60     

X_train60, X_test60, y_train60, y_test60 = train_test_split(X, y, train_size=0.60)
clf.fit(X_train60, y_train60) #fitting the model for the training set
pred_test60 = clf.predict(X_test60) #predict the response
print(accuracy_score(y_test60,pred_test60)) #accuracy of test data
                                            
clf.fit(X_train60, y_train60) #fitting the model for the train set
pred_train60 = clf.predict(X_train60) #predict the response
print(accuracy_score(y_train60,pred_train60))

#now we do a model with train_size of 0.50     

X_train50, X_test50, y_train50, y_test50 = train_test_split(X, y, train_size=0.50)
clf.fit(X_train50, y_train50) #fitting the model for the training set
pred_test50 = clf.predict(X_test50) #predict the response
print(accuracy_score(y_test50,pred_test50)) #accuracy of test data
                                            
clf.fit(X_train50, y_train50) #fitting the model for the train set
pred_train50 = clf.predict(X_train50) #predict the response
print(accuracy_score(y_train50,pred_train50))


plt.scatter(X_train[:,0], X_train[:,1], c="k", label='train')
plt.legend(loc='best')
plt.xlabel("X")
plt.ylabel("f(X)");

"""
While doing the exercise 1, I found that if I calculate the accuracy test for
any particular test/train data, I find different results, and I do not know why.
I will continue to answer the remaining questions.
"""          

#QUESTION 2
from sklearn import datasets 
from sklearn.neighbors import KNeighborsRegressor

X, y, coef = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=0.8, coef=True, random_state=2)

rng = check_random_state(2)
X = np.linspace(-2, 2, 100)
y = 2 * X + np.sin(5 * X) + rng.randn(100) *0.8

X = X.reshape(-1, 1)

#Plot the dataset

plt.plot(X, y, 'ob')

#fit a kNN regressor with varying number of n_neighbors and compare each regressors predictions to the location of the training and testing points.

clf = KNeighborsRegressor(n_neighbors=5)
clf.fit(X,y)
line = np.linspace(-2, 2, 100).reshape(-1, 1)

plt.scatter(X, y, c="b", lw=0, s=40)
plt.plot(line, clf.predict(line), 'r', label='fit')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)
clf.fit(X_train, y_train) #fitting the model for the training set
pred_test = clf.predict(X_test) #predict the response
print(accuracy_score(y_test,pred_test)) #accuracy of test data
         