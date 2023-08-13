#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vovuncozer
"""

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


data = loadmat("digits.mat")
x = data["digits"]
y = data["labels"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lda = LDA(n_components=9)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)
plt.figure()
plot = plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.legend(handles=plot.legend_elements()[0], labels=[0,1,2,3,4,5,6,7,8,9])
plt.title("Scatter with 9-Dimensionality")

plt.figure()

for i in range(0,9):
    base = lda.scalings_[:,i].reshape(20, 20);
    plt. figure()
    plt.imshow(base)

train_errora = []
test_errora  = []

for dim in range (1,10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    lda     = LDA(n_components=dim)
    x_train = lda.fit_transform(x_train, y_train) 
    x_test  = lda.transform(x_test)
    
    model = GaussianNB()
    model.fit(x_train,y_train)
    
    y_train_model = model.predict(x_train)
    y_test_model  = model.predict(x_test)

    train_error   = 1 - accuracy_score(y_train, y_train_model)
    train_error_a = train_errora.append(train_error)
    
    test_error    = 1 - accuracy_score(y_test, y_test_model)
    test_error_a  = test_errora.append(test_error)

dim = [1,2,3,4,5,6,7,8,9] 
plt.figure()    
plt.plot(dim,train_errora)
plt.title("Classification error vs. the Dimension of Subspace for Training Data")
plt.figure()    
plt.plot(dim,test_errora)
plt.title("Classification error vs. the Dimension of Subspace for Test Data")
