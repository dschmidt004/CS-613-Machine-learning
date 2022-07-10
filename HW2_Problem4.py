
# coding: utf-8

# In[1]:


##
## Problem 4
##

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import norm
from pprint import pprint
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.special import logsumexp

data = pd.read_csv('spambase.data')
data = np.array(data)

np.random.seed(0)
np.random.shuffle(data)

# split into training and test sets
trainingIndex = -(-2 * len(data) // 3)
trainingData = data[0:trainingIndex]
testData = data[trainingIndex:]

# standardize data
mean = np.mean(trainingData, axis=0)
std = np.std(trainingData, axis=0, ddof=1)
sXTrainingData = (trainingData[:,0:57]-mean[0:57])/std[0:57]
sXTestData = (testData[:,0:57]-mean[0:57])/std[0:57]

# divide into spam vs non-spam groups
spamSamples = []
nonSpamSamples = []

for i in range(len(trainingData)):
    if trainingData[i][57] == 1:
        spamSamples.append(sXTrainingData[i])
    else:
        nonSpamSamples.append(sXTrainingData[i])

spamSamples = np.array(spamSamples)
nonSpamSamples = np.array(nonSpamSamples)

# get mean and variance
listOfMeansSpam = []
listOfMeansNonSpam = []
listOfVariancesSpam = []
listOfVariancesNonSpam = []

for j in range(57):
    listOfMeansSpam.append(np.mean(spamSamples[:,j]))
    listOfMeansNonSpam.append(np.mean(nonSpamSamples[:,j]))
    listOfVariancesSpam.append(np.var(spamSamples[:,j],ddof=1))
    listOfVariancesNonSpam.append(np.var(nonSpamSamples[:,j],ddof=1))

# create normal models
priorProbSpam = len(spamSamples)/(len(spamSamples)+len(nonSpamSamples))
priorProbNonSpam = len(nonSpamSamples)/(len(spamSamples)+len(nonSpamSamples))

spamProbabilities = []
nonSpamProbabilities = []
predictions = []

for k in range(len(testData)):
    spamProbList = [priorProbSpam]
    nonSpamProbList = [priorProbNonSpam]
    for m in range(len(sXTestData[k])):
        normalModelSpam =         norm.pdf(sXTestData[k][m],listOfMeansSpam[m],listOfVariancesSpam[m])
        normalModelNonSpam =         norm.pdf(sXTestData[k][m],listOfMeansNonSpam[m],listOfVariancesNonSpam[m])
        spamProbList.append(normalModelSpam + np.finfo(float).eps)
        nonSpamProbList.append(normalModelNonSpam + np.finfo(float).eps)
  
    if np.prod(spamProbList) > np.prod(nonSpamProbList):
        predictions.append(1)
    else:
        predictions.append(0)

# evaluate model
truePositives = 0
trueNegatives = 0
falsePositives = 0
falseNegatives = 0

for n in range(len(testData)):
    if testData[n][57] == 1:
        if predictions[n] == 1:
            truePositives += 1
        else:
            falseNegatives += 1
    else:
        if predictions[n] == 0:
            trueNegatives += 1
        else:
            falsePositives += 1

precision = truePositives/(truePositives + falsePositives)
recall = truePositives/(truePositives + falseNegatives)
fMeasure = (2 * precision * recall)/(precision + recall)
accuracy = ((truePositives+trueNegatives)/(truePositives+trueNegatives +            falsePositives+falseNegatives)) * 100

print(precision)
print(recall)
print(fMeasure)
print(accuracy)

