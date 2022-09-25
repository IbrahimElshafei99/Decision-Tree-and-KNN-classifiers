import pandas as pd
import numpy as np
import math
import itertools

#reading training data
training_data=pd.read_csv('trainingSet_csv.csv', sep=';', header=None)
print(len(training_data))
print(training_data.shape)
training_data.head()

#reading testing data
testing_data=pd.read_csv('TestSet_csv.csv', sep=';', header=None)
print(len(testing_data))
print(testing_data.shape)
testing_data.head()

#Xtrain, Ytrain
X_train=training_data.values[:, 0:8]
y_train=training_data.values[:, 8]

#Xtest, Ytest
X_test=testing_data.values[:, 0:8]
y_test=testing_data.values[:, 8]

#eculidenDistance function
def eculidenDistance(x , xi):
    d = 0.0
    for i in range(len(x)):
        d += pow(abs(x[i]-xi[i]),2)
    return math.sqrt(d)

def getKey(item):
    return item[1]

def knn(train,test,k):
    Xtrain = train[0]
    ytrain = train[1]
    Xtest = test[0]
    ytest = test[1]
    
    Count = 0
    for i in range(len(Xtest)):
        newDataSet = []
        for j in range(len(Xtrain)):
            newDataSet.append([j,eculidenDistance(Xtrain[j],Xtest[i]),ytrain[j]])
            
        newDataSet = sorted(newDataSet,key=getKey)
        dict = {}
        keyOfMaxItem = ''
        for item in itertools.islice(newDataSet , 0, k):
            key = item[2]
            keyOfMaxItem = key
            if key in dict:
                dict[key] = dict[key] + 1
            else:
                dict[key] = 1

        for key in dict:
            if dict[key] >= dict[keyOfMaxItem]:
                keyOfMaxItem = key
        if keyOfMaxItem == ytest[i]:
            Count += 1
        #print("Predicted class: ", keyOfMaxItem ,dict[keyOfMaxItem] ," - Actual Class: ", ytest[i])
    
    accuracy = (float(Count)/len(Xtest))*100
    
    print("Number of correctly classified instances : ",Count)
    print("Total number of instances : ",len(Xtest))
    print("Accuracy = ",accuracy)

for k in range(1,10):
    n_neighbors = k
    print("K = ",k)
    knn([X_train,y_train],[X_test,y_test],n_neighbors)
