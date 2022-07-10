#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from collections import Counter
from scipy import stats
from scipy.stats import norm
from pprint import pprint
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.special import logsumexp
#read in data
y = pd.read_csv('spambase.data')
y = np.array(y)
#shuffle
np.random.seed(0)
np.random.shuffle(y)
#split data
size = np.size(y[:,1])
x = math.ceil(size*(2/3))
train = y[0:x,:57]
test = y[x:,:57]
train1 = y[0:x]
test1 = y[x:]
yt = train1[:,57]
ytest = test1[:,57]
#mean and std
mean = np.mean(train, axis=0)
std = np.std(train, axis=0, ddof=1)
meantrain = (train-mean)/std
meantest = (test-mean)/std
#find median
mediantrain = np.median(meantrain,axis=0)
mediantest = np.median(meantest,axis=0)

train = train.astype(int)
test = test.astype(int)
#find the accuracy of predicted 
def accuracy(y, spam):
    tspam,tnotspam,fspam ,fnotspam = 0,0,0,0
    total = len(y)
    #for every y value we test it verses our prediction
    for n in range(len(y)):
        if(y[n] == 0):
            #we count these values
            if(spam[n] == 0):
                tnotspam = tnotspam+ 1
            else:
                fspam = fspam + 1
        else:
            if (spam[n] == 1):
                tspam = tspam + 1
            else:
                fnotspam = fnotspam +  1
    #we calculate the precision,recall, fMeasure and accuracy
    prec = tspam/(tspam + fspam)
    print("Precision ")
    print(prec)
    rec = tspam/(tspam + fnotspam)
    print("Recall")
    print(rec)
    fMe =  2*rec * prec/(prec + rec)
    print("fMeasure")
    print(fMe)
    accuracy = ((tspam+tnotspam)/total) 
    print("this is the accuracy percentage ")
    print(accuracy*100)
#Node class for each tree arm  
class Node:
    def __init__(self,left=None, right=None,index=None,TheNode=None):
        self.left  = left
        self.right  = right
        self.index  = index
        self.TheNode  = TheNode
#tree class  
class tree:
    #initial the tree
    def __init__(self, Mdepth=40, num=None):
        self.root = None
        self.Mdepth = Mdepth
        self.num = num
    #function for moving through tree 
    def move(self, xtest, node):
        #checks to see if its a empty node
        if(not node.TheNode == None):
            return node.TheNode
        #else it branches right if node.index > .5
        elif (xtest[node.index] > 0.5):
            return self.move(xtest, node.right)
        #else branch left
        else:
            return self.move(xtest, node.left)
    #create a new fit for a tree
    def fit(self, xtest, ytest):
        self.num = len(xtest[0])
        self.root = self.arm(ytest, xtest)
    #calculates a prediction from roots
    def calculate(self, xtrain):
        arr = []

        for i in xtrain:
            arr.append(self.move(i,self.root))
        arr = np.array(arr)
        return arr
    #calculates entropy 
    def entropy(self,temp):
        #counts the number of values in temp
        countsofvar = np.bincount(temp)
        counts = countsofvar / len(temp)
        total = 0
        for i in counts:
            #calculates log value
            #if i = 0 we add a small value
            if(i == 0):
                i = np.finfo(float).eps
            total += i*np.log2(i)
        return -1*total
    def ig(self, spam, indexs):
        #put all index values less than .5 on left index
        leftv = np.argwhere(indexs <= 0.5).flatten()
        #put all > 0.5 on right
        rightv = np.argwhere(indexs > 0.5).flatten()
        #if either are = 0 then entropy = 0
        if (len(leftv) == 0 or len(rightv) == 0):
            total = 0
        #if they are the same entrop = 1
        elif (len(leftv) == len(rightv)):
            total = 1
        else:
            #else calculate entropy
            LEN = len(leftv)
            REN =  len(rightv)
            total = len(spam)
            LE = self.entropy(spam[leftv])
            RE = self.entropy(spam[rightv])
            branchE =  (LE *LEN / total) + (REN *RE / total) 
            tE = self.entropy(spam)
            total = tE - branchE
        return total
    #create tree arms
    def arm(self, value,data,d=0):
        
        index = None
        Spam_N = np.shape(data)[0]
        Spam_att = np.shape(data)[1]
        spamN = len(np.unique(value))
        #if values 1 or less return node mode
        if(spamN <= 1):
            x = stats.mode(value)[0]
            x = Node(TheNode = x)
            return x
        #if attributes are gone return node mode
        elif(1 >= Spam_N ):
            x = stats.mode(value)[0]
            x = Node(TheNode = x)
            return x
         # if you have reached a max depth return node
         # this is a safety precaution to prevent us from diving too deep on the data
         
        elif ( self.Mdepth <=d):
            x = stats.mode(value)[0]
            x = Node(TheNode = x)
            return x
        #else we make a branch
        else:
            maxg = 0
            #find the next max value
            for i in range(np.shape(data)[1]):
                IG = self.ig(value, data[:,i])
                if(maxg == 0):
                    maxg = IG
                    index = i
                elif( maxg <= IG):
                    maxg = IG
                    index = i
            #create left and right branches
            rightv = np.argwhere(data[:, index] > 0.5).flatten()
            right = self.arm( value[rightv],data[rightv, :],d+1)
            leftv = np.argwhere(data[:, index] <= 0.5).flatten()
            left = self.arm(value[leftv],data[leftv,:],d+1)
            #once we make these branches we return the node itteratively
            return Node(left,right,index)
        
#This binarizes the data from the median
for i in range(np.shape(meantrain)[0]):
    for j in range(np.shape(meantrain)[1]):
        if(meantrain[i,j]> mediantrain[j]):
            meantrain[i,j] =1
        else:
            meantrain[i,j] =0
for i in range(np.shape(meantest)[0]):
    for j in range(np.shape(meantest)[1]):
        if(meantest[i,j]> mediantest[j]):
            meantest[i,j] =1
        else:
            meantest[i,j] =0
#Once we binaries we want to make sure everything is labeled as int
meantrain = meantrain.astype(int)
meantest = meantest.astype(int)
yt = train1[:,57].astype(int)
ytest = test1[:,57].astype(int)
#create the tree with a max depth
x = tree(Mdepth=40)
#fit the data to the tree
x.fit(meantrain, yt)
#test are test values to y
y_final = x.calculate(meantest)
#get the accuracy and print
accuracy(ytest, y_final)


# In[ ]:




