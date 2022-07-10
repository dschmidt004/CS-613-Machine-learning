#!/usr/bin/env python
# coding: utf-8

# In[5]:



class ID3_tree:
    def fit(self,Mat,Y,Xcolumns,Ycolumn):
        new_mat = Mat.copy()
        self.tree = self.decision_tree(new_mat,new_mat,Xcolumns,Ycolumn)
        
    def predict (self,X):
        
        chance = []
        for i in range(np.size(X[:,3])):
            chance.append(self.make_prediction(X[i,:],self.tree,1.0))
        return chance


    def entropy(self,a):
        e_list = []
        y, x = np.unique(a, return_counts=True)
        
        for i in range(len(y)):
            prob = x[i]/np.sum(x)
            e_list.append(np.log2(prob)*-prob)
        total = np.sum(e_list)
        return total

    def IG(self,examples,column,test):
        entropy = self.entropy(example[test])
        x,y = np.unique(example[test])
        eset = []
        for i in range(len(x)):
            probset = y[i]/np.sum(y)
            subetro = self.entropy(data.where(data[column]==x[i]).dropna()[column])
            eset.append(subetro*probset)
        total = eset
        total = sum(total)
        total = total - entropy  
        return total

    def decision_tree(self,X, Y, names, test,parent=None):
        unique = np.unique(X[:,test])
        
        print(1)
        
        if (len(names) == 0):
            return parent
        elif (len(unique) <= 1):
            return unique[0]

        elif (len(X) == 0):
            print("No examples")
            index = np.argmax(np.unique(Y[test], return_counts=True)[1])
            return np.unique(X1[test])[index]






        else:
            index = np.argmax(np.unique(X[test], return_counts=True)[1])
            parent = unique[index]
            IGlist = []
            
            
            
            for i in names:
                IGlist.append(self.IG(X, i, test))
            index = np.argmax(IGlist)
            next = names[index]

         #########
            tree = {child: {}}



            names.pop(index)

            parent.values = np.unique(X[child])
            for j in parent.values:


                sub_data = X.where(X[child] == j).dropna()

                # call the algorithm recursively
                subtree = self.decision_tree(sub_data, X1, names, test, parent)

                tree[child][j] = sub_tree
                print(1)
            return tree

    def make_prediction(self, sample, tree, default=1):
        # map sample data to tree
        for i in list(sample):
          # check if feature exists in tree
            if i in list(tree.keys):
                try:
                    result = tree[i][sample[i]]
                except:
                    return default

                result = tree[i][sample[i]]

                # if more attributes exist within result, recursively find best result
                if isinstance(result, dict):
                    return self.make_prediction(sample, result)
                else:
                    return result


# In[11]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from pprint import pprint
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from scipy.special import logsumexp
data = pd.read_csv('spambase.data')

#1. Reads in the data, ignoring the first row (header) and first column (index).
y = data.to_numpy()
print(np.shape(y))
np.random.seed(0)
#2. Randomizes the data
np.random.shuffle(y)

size = np.size(y[:,1])
#3. Selects the first 2/3 (round up) of the data for training and the remaining for testing
x = math.ceil(size*(2/3))
Yan = np.array(y[0:x,[57]]) #2/3 train data
yanTest = np.array(y[x:,[57]]) #1/3 test data

mat = np.array(y[:,:57])


one = np.ones((size,1))
mat = np.concatenate((one, mat), axis=1)
mat1= mat[:,1:]

#4. Standardizes the data (except for the last column of course) base on the training data
mean2 = np.mean(mat1 ,axis=0)
std2 = np.std(mat1, axis=0,ddof=1)

mat1 = (mat1-mean2)/std2
mat[:,1:]=mat1

mat2= np.array(mat[0:x,:])
mat_test = np.array(mat[x:,:])

tX = np.concatenate((mat2, Yan), axis=1)
tspam = tX[tX[:,58] == 1]
#print(np.shape(tspam))
tnotspam = tX[tX[:,58] == 0]
#print(np.shape(tnotspam))

testX = np.concatenate((mat_test, yanTest), axis=1)
testspam = testX[testX[:,58] == 1]
#print(np.shape(testspam))
testnotspam = testX[testX[:,58] == 0]
#print(np.shape(testnotspam))
columns = np.ones((1,59))
for i in range(59):
    columns[0,i] = int(i)

#print(columns)
print(np.shape(tspam))
print(np.shape(columns))
x = ID3_tree()
x.entropy(testX)

x.fit(tspam,tspam[:,58],columns[:58],int(columns[0,58]))



#testspam = np.concatenate((columns, testspam), axis=0)
#y_pred = x.predict(testspam)


# In[ ]:




