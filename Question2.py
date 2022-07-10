#!/usr/bin/env python
# coding: utf-8

# In[102]:


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
from scipy.spatial import distance

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
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
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
# build a KNeighborsClassifier using one neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.10f}".format(knn.score(X_test, y_test)))


# In[103]:


#1
def distances(X_train,y_train,X_test,y_test,d = 1):
    temp = np.zeros(np.shape(y_train))

    y_ans = np.zeros(np.shape(y_test))
    count = 0
    for i in range(np.shape(X_test)[0]):
        dis = temp
        for j in range(np.shape(X_train)[0]):
            dis[j] = distance.euclidean(X_test[i,:d], X_train[j,:d])

        y_ans[i] = y_train[np.argmin(dis)]
        if(y_ans[i]==y_test[i]):
            count += 1
    x = count/np.shape(y_ans)[0]
    return x*100


# In[107]:



print(np.shape(X_train))
#print(distances(X_train,y_train,X_test,y_test,100))

mean = np.mean(X_train ,axis=0)
std = np.std(X_train, axis=0,ddof=1)

SX_train = (X_train-mean)/std
SX_test = (X_test-mean)/std

n = np.size(SX_train[:,0])
x=SX_train
x1=SX_test
#x = np.cov(SX_train)
print(np.shape(x))
y = (x.T@x)/(n-1)
y1 = (x1.T@x1)/(n-1)
print(np.shape(y))
values, vectors= np.linalg.eig(y)

values1, vectors1= np.linalg.eig(y1)
# print(np.shape(values))
# print(np.shape(vectors))
print(values)
print(vectors)
X_train1 = x@vectors[:,:100]
X_test2 = x1@vectors1[:,:100]
print(distances(X_train1,y_train,X_test2,y_test,100))
#print(np.shape(z))



# In[110]:


print(np.shape(vectors[0,:100]))


# In[ ]:




