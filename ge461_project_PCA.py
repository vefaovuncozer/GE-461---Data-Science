#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vovuncozer
"""
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


data = loadmat("digits.mat")
x = data["digits"]
y = data["labels"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

pca = PCA(n_components=400)
pca.fit(x_train)
sorted_eigen_values = -np.sort(-pca.explained_variance_)
plt.plot(sorted_eigen_values)
plt.title("Eigenvalues in Descending Order")

train_mean = pca.mean_
plt.figure()
plt.imshow(train_mean.reshape(20,20).T)
plt.title("Mean for the Training Data set")

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test  = sc.transform(x_test)

pca = PCA(0.95)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test  = pca.transform(x_test)
eigen_vectors = pca.components_
plt.figure()
plot = plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.legend(handles=plot.legend_elements()[0], labels=[0,1,2,3,4,5,6,7,8,9])
plt.title("Scatter with 137 Principal Components")

plt.figure()

plt.title("137 Principal Components")
for i in range(1,len(eigen_vectors)):
    plt.subplot(14,10,i)
    plt.imshow(eigen_vectors[i-1].reshape(20,20).T)
    plt.axis(False)

dim = np.arange(1, 201, 4).tolist()
train_error_t = []
test_error_t  = []

for i in dim:    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    pca = PCA(n_components=i)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test  = pca.transform(x_test)
    
    model = GaussianNB()
    model.fit(x_train, y_train)
    
    y_train_model = model.predict(x_train)
    y_test_model  = model.predict(x_test)
    
    train_error  = 1 - accuracy_score(y_train, y_train_model)
    train_error_t.append(train_error)
    test_error   = 1 - accuracy_score(y_test, y_test_model)
    test_error_t.append(test_error)
    
plt.figure()    
plt.plot(dim,train_error_t)
plt.title("Classification error vs. the Number of Components for Training Data")
plt.figure()    
plt.plot(dim,test_error_t)
plt.title("Classification error vs. the Number of Components for Test Data")

