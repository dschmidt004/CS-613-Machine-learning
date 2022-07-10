#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
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


# In[2]:


mean = np.mean(X_train ,axis=0)
std = np.std(X_train, axis=0,ddof=1)
SX_train = (X_train-mean)/std
SX_test = (X_test-mean)/std
n = np.shape(SX_train)[0]
x = SX_train
z = (x.T@x)/n
values, vectors= np.linalg.eig(z)


pca1=(x@vectors)[:,0]
pca2 =(x@vectors)[:,1]
pmax = np.argmax(pca1)
pmax1 = np.argmax(pca2)
pmin = np.argmin(pca1)
pmin1 = np.argmin(pca2)
print(people.target_names[y_train[pmax]])
print(people.target_names[y_train[pmax1]])
print(people.target_names[y_train[pmin]])
print(people.target_names[y_train[pmin1]])
plt.scatter(pca1,pca2)


# In[29]:


print(X_people[1425].shape)


# In[25]:


#x1 = (x * std)+mean
#print(np.shape(vectors[:,0]@x1[:0,:]))
image = np.reshape(vectors[:,0],(87,65))
#people.images[0] = image
plt.imshow(image,interpolation='nearest',cmap=cm.gray)
plt.show()


# In[55]:


x1 = (x * std)+mean
#z = (vectors[:,:100].T@x1.T).T

Z = np.array(X_train@vectors[:,:1])
print(Z.shape)
final = Z@vectors[:,:1].T

image = np.reshape(final[0,:],(87,65))
plt.imshow(image,interpolation='nearest',cmap=cm.gray)
plt.show()


# In[39]:


i = 0
best = values[np.flip(np.argsort(values))]
total = np.sum(best)
sum1 = 0.0
while(sum1<.95):
    sum1 = np.sum(values[:i])/total
    i += 1
print(i)


# In[41]:


x1 = (x * std)+mean

Z = X_train@vectors[:,:i]
final = Z@vectors[:,:i].T

image = np.reshape(final[0,:],(87,65))
plt.imshow(image,interpolation='nearest',cmap=cm.gray)
plt.show()


# In[ ]:




