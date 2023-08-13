"""
@author: vovuncozer

PCA AND DOGS
"""

import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt


data_sample = []
for filename in glob.glob('/Users/vovuncozer/Desktop/afhq_dog/*.jpg'):
    img = Image.open(filename).resize((64,64), Image.BILINEAR)
    array = np.array(img).reshape(4096,3)
    sample = array.astype(np.float32)
    data_sample.append(sample)

data_sample = np.array(data_sample)

channel1 = data_sample[:,:,0]
channel2 = data_sample[:,:,1]
channel3 = data_sample[:,:,2]


# Mean centering the data 
channel1_meaned = channel1 - np.mean(channel1 , axis = 0)
channel2_meaned = channel2 - np.mean(channel2 , axis = 0)
channel3_meaned = channel3 - np.mean(channel3 , axis = 0)

# calculating the covariance matrix of the mean-centered data.
cov_mat1 = np.cov(channel1_meaned, rowvar = False)
cov_mat2 = np.cov(channel2_meaned, rowvar = False)
cov_mat3 = np.cov(channel3_meaned, rowvar = False)

#Calculating Eigenvalues and Eigenvectors of the covariance matrix
eigen_values1, eigen_vectors1 = np.linalg.eigh(cov_mat1)
eigen_values2, eigen_vectors2 = np.linalg.eigh(cov_mat2)
eigen_values3, eigen_vectors3 = np.linalg.eigh(cov_mat3)

#sort the eigenvalues and eigenvectors  in descending order
sorted_index1 = np.argsort(eigen_values1)[::-1]
sorted_index2 = np.argsort(eigen_values1)[::-1]
sorted_index3 = np.argsort(eigen_values3)[::-1]

sorted_eigenvalue1 = eigen_values1[sorted_index1]
sorted_eigenvalue2 = eigen_values2[sorted_index2]
sorted_eigenvalue3 = eigen_values3[sorted_index3]

sorted_eigenvectors1 = eigen_vectors1[:,sorted_index1]
sorted_eigenvectors2 = eigen_vectors2[:,sorted_index2]
sorted_eigenvectors3 = eigen_vectors3[:,sorted_index3]

# select the first n eigenvectors, n is desired dimension
# of our final reduced data.
eig_vals_total1 = sum(eigen_values1)
eig_vals_total2 = sum(eigen_values2)
eig_vals_total3 = sum(eigen_values3)
explained_variance1 = [(i / eig_vals_total1)*100 for i in sorted_eigenvalue1]
explained_variance2 = [(i / eig_vals_total2)*100 for i in sorted_eigenvalue2]
explained_variance3 = [(i / eig_vals_total3)*100 for i in sorted_eigenvalue3]

cum_explained_variance1 = np.cumsum(explained_variance1)
cum_explained_variance2 = np.cumsum(explained_variance2)
cum_explained_variance3 = np.cumsum(explained_variance3)

n_components = 10 #you can select any number of components.

eigenvector_subset1 = sorted_eigenvectors1[:,0:n_components]
eigenvector_subset2 = sorted_eigenvectors2[:,0:n_components]
eigenvector_subset3 = sorted_eigenvectors3[:,0:n_components]

ten_eigenvectors1 = []
for i in range (10):
    x = eigenvector_subset1[:,i].reshape(64,64)
    x_max = np.amax(x)
    x_min = np.amin(x)
    x=(x-x_min)/(x_max-x_min)
    ten_eigenvectors1.append(x)

ten_eigenvectors2 = []
for i in range (10):
    x = eigenvector_subset2[:,i].reshape(64,64)
    x_max = np.amax(x)
    x_min = np.amin(x)
    x=(x-x_min)/(x_max-x_min)
    ten_eigenvectors2.append(x)
    
ten_eigenvectors3 = []
for i in range (10):
    x = eigenvector_subset3[:,i].reshape(64,64)
    x_max = np.amax(x)
    x_min = np.amin(x)
    x=(x-x_min)/(x_max-x_min)
    ten_eigenvectors3.append(x)

eigen_3channel = []
for i in range (10):
    eigen_3channel.append(np.stack((ten_eigenvectors1[i],ten_eigenvectors2[i],ten_eigenvectors3[i]),axis=-1))
    
eigen_3channel = np.array(eigen_3channel)

for i in range(1,11):
    plt.subplot(2, 5, i)
    plt.imshow(eigen_3channel[i-1])
    plt.axis(False)
    

sample_image = Image.open("/Users/vovuncozer/Desktop/afhq_dog/flickr_dog_000002.jpg").resize((64,64), Image.BILINEAR)
sample_imagex = np.array(sample_image).reshape(1,4096,3)
sample_imagex = sample_imagex.astype(np.float32)
sample_image1 = sample_imagex[:,:,0]
sample_image2 = sample_imagex[:,:,1]
sample_image3 = sample_imagex[:,:,2]

k_val = [1, 50, 250, 500, 1000, 4096]

for k in k_val:
    eigenvectors_c1 = sorted_eigenvectors1[:,0:k]
    eigenvectors_c2 = sorted_eigenvectors2[:,0:k]
    eigenvectors_c3 = sorted_eigenvectors3[:,0:k]
    
    Z1 = np.dot(sample_image1,eigenvectors_c1)
    X_PCA_c1 = np.dot(Z1,eigenvectors_c1.T) 

    Z2 = np.dot(sample_image2,eigenvectors_c2)
    X_PCA_c2 = np.dot(Z2,eigenvectors_c2.T)

    Z3 = np.dot(sample_image3,eigenvectors_c3)
    X_PCA_c3 = np.dot(Z3,eigenvectors_c3.T) 

    X_hat = np.stack((X_PCA_c1,X_PCA_c2,X_PCA_c3),axis=-1)
    X_hat = X_hat.astype(np.uint8)
    X_hat = X_hat.reshape(64,64,3)
    
    plt.figure()
    plt.title("PCA Reconstruction with " + str(k) + "- eigenvectors")
    plt.axis("off")
    plt.imshow(X_hat)
    plt.show()


