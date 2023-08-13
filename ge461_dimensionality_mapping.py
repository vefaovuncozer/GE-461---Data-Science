#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vovuncozer
"""

from scipy.io import loadmat
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sammon import sammon

data = loadmat("digits.mat")
x = data["digits"]
y = data["labels"]

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=250)
tsne_results = tsne.fit_transform(x)
plt.figure()
plot = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],c=y, cmap='Spectral')
plt.legend(handles=plot.legend_elements()[0], labels=[0,1,2,3,4,5,6,7,8,9])
plt.title("t-sne Method for Mapping the Data Set 250 Iterations")

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=2000)
tsne_results = tsne.fit_transform(x)
plt.figure()
plot = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap='Spectral')
plt.legend(handles=plot.legend_elements()[0], labels=[0,1,2,3,4,5,6,7,8,9])
plt.title("t-sne Method for Mapping the Data Set 2000 Iterations")

[ys,E] = sammon(x, 2)
plot = plt.scatter(ys[:, 0], ys[:, 1], c=y, cmap='Spectral')
plt.title("Sammon's Mapping for the Data Set")
plt.legend(handles=plot.legend_elements()[0], labels=[0,1,2,3,4,5,6,7,8,9])
plt.show()

