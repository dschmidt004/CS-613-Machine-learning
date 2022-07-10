#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import matplotlib.cm as cm

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8),subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image, cmap=cm.gray)
    ax.set_title(people.target_names[target])
    
print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))

# count how often each target appears
counts = np.bincount(people.target)
# print counts next to target names
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()
        
mask = np.zeros(people.target.shape, dtype=np.bool_)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
# build a KNeighborsClassifier using one neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.10f}".format(knn.score(X_test, y_test)))


# In[4]:


X_train = X_people
mean = np.mean(X_train ,axis=0)
std = np.std(X_train, axis=0,ddof=1)
SX_train = (X_train-mean)/std

n = np.shape(SX_train)[0]
a = np.shape(SX_train)[1]
k = 100
x=SX_train
y = (x.T@x)/(n-1)
values, vectors= np.linalg.eig(y)
eigvec = vectors[:,np.flip(np.argsort(values))]
eig = values[np.flip(np.argsort(values))]
eig = eig[:100]
neweig=eigvec[:,:100]
X = (neweig.T@x.T).T


# In[7]:


from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial import distance
def clusters(X,n):
    temp = np.random.RandomState(0)
    per = temp.permutation(np.shape(X)[0])[:n]
    circle = X[per]
    while 1:
        title = pairwise_distances_argmin(X, circle)
#         title = np.zeros((np.size(X[:,0])))
#         for i in range(np.size(X[:,0])):
#             temp = np.zeros((100,1))
#             for j in range(np.size(circle[:,0])):
#                 temp[j] = distance.euclidean(X[i,:],circle[j,:])
#             title[i] = np.argmin(temp)
        
        new_circle = np.array([X[title == i].mean(0) for i in range(n)])
        if np.all(circle == new_circle):
            break
        circle = new_circle
    return circle,title


# In[8]:


centers, labels = clusters(X, 100)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=5);
print(labels)


# In[111]:


title = np.zeros((np.size(X[:,0])))
for i in range(np.size(X[:,0])):
    temp = np.zeros((100,1))
    for j in range(np.size(circle[:,0])):
        temp[j] = distance.euclidean(X[i,:],circle[j,:])
    title[i] = np.argmin(temp)
print(title)


# In[106]:


circle=np.ones((a,0))
temp1=np.ones((a,0))
np.random.seed(0)
print(a)
for i in range(k):
    circle = np.c_[circle,X[rd.randint(0,n-1)]]
for j in range(k):
    sum1 = X-circle[:,j]
    temp2=np.square(np.sum(sum1,axis=0))
    temp1=np.c_[temp1,temp2]
min1 = np.argmin(temp1,axis=1)


# In[ ]:




