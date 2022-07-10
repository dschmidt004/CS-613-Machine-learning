
# coding: utf-8

# In[2]:


##
## Problem 3
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
Ymat = [trainingData[:,57]]
Ymat = np.array(Ymat)
Ymat = Ymat.T
Xmat = sXTrainingData

# initialize gradient descent parameters
eta = 0.01
N = len(sXTrainingData)
np.random.seed(0)
thetas = np.zeros((len(sXTrainingData[0]),1))
i = 0

for i in range(len(sXTrainingData[0])):
    thetas[i] = np.random.uniform(-1,1)

thetas = np.array(thetas)
errorChange = 1.0
minErrorChange = 2**(-23)
finalIteration = 10000
loss = 0

xPlot = []
yPlot = []
xTest = []
yTest = []

j = 0

while j < finalIteration and errorChange > minErrorChange:
    
    # update theta
    zMat = Xmat @ thetas
    sigmoid = 1/(1+np.exp(-zMat))
    gradient = Ymat - sigmoid
    update = (eta/N) * Xmat.T @ gradient
    thetas = thetas + update
    
    # calculate training set loss
    previousLoss = loss
    spamLoss = Ymat * np.log(sigmoid + np.finfo(float).eps)
    nonSpamLoss = (1-Ymat) * np.log(1 - sigmoid + np.finfo(float).eps)
    loss = -np.sum(spamLoss + nonSpamLoss)
    errorChange = np.abs(loss - previousLoss)
    
    yPlot.append(loss)
    
    xPlot.append(j)
    
    j += 1

plt.plot(xPlot,yPlot)

plt.show()
print(loss)

# make predictions
predictions = []
for k in range(len(sXTestData)):
    zMat = sXTestData[k] @ thetas
    sigmoid = 1/(1+np.exp(-zMat))
    spamLikelihood = sigmoid + np.finfo(float).eps
    nonSpamLikelihood = 1 - sigmoid + np.finfo(float).eps
    probabilitySpam = np.sum(np.log(spamLikelihood))
    probabilityNonSpam = np.sum(np.log(nonSpamLikelihood))
    if probabilitySpam > probabilityNonSpam:
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

print("Precision:\t",precision)
print("Recall:\t\t",recall)
print("fMeasure:\t",fMeasure)
print("Accuracy:\t",accuracy,"%")

