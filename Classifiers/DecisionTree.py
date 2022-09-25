import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import metrics

col_names=['political_party', 'issue1', 'issue2', 'issue3', 'issue4', 'issue5', 'issue6', 'issue7', 'issue8', 'issue9', 'issue10', 'issue11', 'issue12', 'issue13', 'issue14', 'issue15', 'issue16']
vote_data = pd.read_csv('house-votes.csv', sep=';', header=None, names=col_names)

#reading data
print(len(vote_data))
print(vote_data.shape)
vote_data.head()

#encoding data to numric
le = preprocessing.LabelEncoder()
enc_vote_data = vote_data.apply(le.fit_transform)
enc_vote_data.head()
print(enc_vote_data.shape)
inputs = enc_vote_data.drop('political_party', axis='columns')
target = enc_vote_data['political_party']

#solving missing values

#ImputedModel=SimpleImputer(missing_values=2, strategy='most_frequent')
#ImputedX=ImputedModel.fit(X)
#X=ImputedX.transform(X)
#print(X[:10])

ImputedModel=SimpleImputer(missing_values=2, strategy='most_frequent')
inputs.issue1=ImputedModel.fit_transform(inputs['issue1'].values.reshape(-1,1))
inputs.issue2=ImputedModel.fit_transform(inputs['issue2'].values.reshape(-1,1))
inputs.issue3=ImputedModel.fit_transform(inputs['issue3'].values.reshape(-1,1))
inputs.issue4=ImputedModel.fit_transform(inputs['issue4'].values.reshape(-1,1))
inputs.issue5=ImputedModel.fit_transform(inputs['issue5'].values.reshape(-1,1))
inputs.issue6=ImputedModel.fit_transform(inputs['issue6'].values.reshape(-1,1))
inputs.issue7=ImputedModel.fit_transform(inputs['issue7'].values.reshape(-1,1))
inputs.issue8=ImputedModel.fit_transform(inputs['issue8'].values.reshape(-1,1))
inputs.issue9=ImputedModel.fit_transform(inputs['issue9'].values.reshape(-1,1))
inputs.issue10=ImputedModel.fit_transform(inputs['issue10'].values.reshape(-1,1))
inputs.issue11=ImputedModel.fit_transform(inputs['issue11'].values.reshape(-1,1))
inputs.issue12=ImputedModel.fit_transform(inputs['issue12'].values.reshape(-1,1))
inputs.issue13=ImputedModel.fit_transform(inputs['issue13'].values.reshape(-1,1))
inputs.issue14=ImputedModel.fit_transform(inputs['issue14'].values.reshape(-1,1))
inputs.issue15=ImputedModel.fit_transform(inputs['issue15'].values.reshape(-1,1))
inputs.issue16=ImputedModel.fit_transform(inputs['issue16'].values.reshape(-1,1))

#select X,Y from dataset
X=inputs
Y=target

#Using 25% training size
i=1
while i<4:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.75, random_state=0)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print('Experiment', i)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    sizeOfTree=clf.tree_
    print("Size: ", sizeOfTree.node_count)
    i+=1

#Training set sizes in the range (30-70%)
def bestTree1(x_data, y_data):
    best_accuracy=0.0000001
    for Size in [0.7,0.6,0.5,0.4,0.3]:
        total_acc=0
        min_acc=1000000
        max_acc=0.0000001
        for i in range(1,6):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=Size, random_state=0)
            clf = DecisionTreeClassifier()
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            score=metrics.accuracy_score(y_test, y_pred)
            
            print('Test size',Size*100,'% Experiment', i)
            print("Accuracy:",score)
            
            if score > max_acc:
                max_acc=score
            if score < min_acc:
                min_acc=score
            total_acc+=score
        avg_acc=total_acc/5
        print('Mean accuracy of ',Size*100,'% test tree : ', avg_acc)
        print('Maximum accuracy of ',Size*100,'% test tree : ', max_acc)
        print('Minimum accuracy of ',Size*100,'% test tree : ', min_acc)
        print('///////////////////////////////////////////////////////')
    if avg_acc > best_accuracy:
        best_accuracy=avg_acc
    print('Best accuracy = ', best_accuracy)

bestTree1(X,Y)

#Training data size 40% , 50% .... Until you reach 80%
def bestTree2(x_data, y_data):
    best_accuracy=0.0000001
    fig, (ax1, ax2) = plt.subplots(2)
    for Size in [0.6,0.5,0.4,0.3,0.2]:
        total_acc=0
        min_acc=1000000
        max_acc=0.0000001
        for i in range(1,4):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=Size, random_state=0)
            clf = DecisionTreeClassifier()
            clf = clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            score=metrics.accuracy_score(y_test, y_pred)
            
            print('Test size',Size*100,'% Experiment', i)
            print("Accuracy:",score)
            
            if score > max_acc:
                max_acc=score
            if score < min_acc:
                min_acc=score
            total_acc+=score
            #plot how accuracy varies with training set size
            ax1.plot(Size, score, marker="o")
            ax1.set_xlabel('Test size')
            ax1.set_ylabel('Accuracies')
            #plot how the number of nodes varies with training set size
            numOfNodes=clf.tree_
            ax2.plot(Size, numOfNodes.node_count, marker="o")
            ax2.set_xlabel('Test size')
            ax2.set_ylabel('Number of nodes')
            
        avg_acc=total_acc/3
        print('Mean accuracy of ',Size*100,'% test tree : ', avg_acc)
        print('Maximum accuracy of ',Size*100,'% test tree : ', max_acc)
        print('Minimum accuracy of ',Size*100,'% test tree : ', min_acc)
        print('///////////////////////////////////////////////////////')
    plt.show
    if avg_acc > best_accuracy:
        best_accuracy=avg_acc
    print('Best accuracy = ', best_accuracy)

 
bestTree2(X,Y) 
   
#Visualize the decision tree with best accuracy
fig = plt.figure(figsize=(25,20))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
clf=DecisionTreeClassifier()
clf=clf.fit(X_train, y_train)
graph=tree.plot_tree(clf, filled=True)
fig.savefig("decistion_tree.png")
