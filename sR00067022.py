#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on a sunny day

@author: Christopher O'Grady
@id: R00067022
@Cohort: SDH3-B
"""

from sklearn.model_selection import  StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


def Task1():
    # cross validate using 5 folds and 20% test size
    # use decision tree classifier

    df = pd.read_csv("humanDetails.csv", encoding="ISO-8859-1")
    #print(df['education'].value_counts())

    # cleaning data
    df = df[df["native-country"] != " ?"]
    df = df[df[" workclass"] != " ?"]
    df['age'] = df['age'].str.replace('90s', '90')
    df['age'] = df['age'].str.replace('80s', '80')
    df['age'] = df['age'].str.replace('70s', '70')
    df['age'] = df['age'].str.replace('60s', '60')
    df['age'] = df['age'].str.replace('50s', '50')
    df['age'] = df['age'].str.replace('40s', '40')
    df['age'] = df['age'].str.replace('20s', '20')

    # Income is either or
    df['Income'] = df['Income'].str.replace('<=50K', '0')
    df['Income'] = df['Income'].str.replace('>50K', '1')

    # assigning int values to workclass variables by dictionary
    all_workclass = np.unique(df[' workclass'].astype(str))
    dictW = {}
    c = 1
    for wc in all_workclass:
        dictW[wc] = c
        c = c+1
    df[' workclass'] = df[' workclass'].map(dictW)

    # todo -- to check dict values for work class -> print(dictW)


    # values are int
    df['age'] = df['age'].apply(pd.to_numeric, errors='coerce')
    df['Income'] = df['Income'].apply(pd.to_numeric, errors='coerce')

    # all unique countries
    all_countries = np.unique(df['native-country'].astype(str))
    # data values
    X = (df[[' workclass', 'age']])
    # our class value to find
    y = df[['Income']]

    # the following records average accuracy through all iterations of the tree,
    # allowing us to graph the training and test data
    kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    train_acc = []
    test_acc = []
    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(X, y)
    tree_train = []
    tree_test = []
    # testing all depths of the tree
    for i in range(tree_clf.get_depth()-1):
        tree_clf = tree.DecisionTreeClassifier(max_depth=i+1)
        for train, test in kfold.split(X, y):
            tree_clf.fit(X.iloc[train], y.iloc[train])
            train_acc.append(tree_clf.score(X.iloc[train], y.iloc[train]))
            test_acc.append(tree_clf.score(X.iloc[test], y.iloc[test]))
        tree_train.append(np.mean(train_acc))
        tree_test.append(np.mean(test_acc))
    plt.plot(tree_test, label='Test data')
    plt.plot(tree_train, label='Training data')
    plt.xlabel("depth")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    print("Diagram 1 shows a graph of accuracy")

    print("All other diagrams will display over fitting for each country that overfits")
    # executing the same operation for each country
    for country in all_countries:
        # dataframe will only have current country
        df2 = df[df["native-country"] == country]
        num = (df2['Income'].value_counts())
        # there must be more than one instance of Income being 1 and 0
        if num.size > 1:
            if num[0] > 1 and num[1] > 1:
                X = (df2[[' workclass', 'age']])
                y = df2[['Income']]
                train_acc = []
                test_acc = []
                for train, test in kfold.split(X, y):
                    tree_clf.fit(X.iloc[train], y.iloc[train])
                train_acc.append(tree_clf.score(X.iloc[train], y.iloc[train]))
                test_acc.append(tree_clf.score(X.iloc[test], y.iloc[test]))
                #print(round(np.mean(train_acc), 1), " -> ", end='')
                #print(round(np.mean(test_acc), 1))
                # uncomment the following lines to predict income base on work, age
                #print(country + " \t\t ", end='')
                print(tree_clf.predict([[2, 40]]))
                # here we graph the countries that over fit and label them
                if (np.mean(train_acc) - np.mean(test_acc)) > .2:
                    tree_train = []
                    tree_test = []
                    for i in range(tree_clf.get_depth()-1):
                        tree_train.append(np.mean(train_acc))
                        tree_test.append(np.mean(test_acc))
                    plt.plot(tree_test, label='Test data')
                    plt.plot(tree_train, label='Training data')
                    plt.title(country)
                    plt.xlabel("depth")
                    plt.ylabel("accuracy")
                    plt.legend()
                    plt.show()
                    print(country, " overfits")


# todo Task 1 findings: As per the training/test graph a depth of 3 or 4 produces the best accuracy.
#  In all graphs that overfit the line is straight across or for some reason does not exist.
#  This may be because data lacks quantity or all values of our data for a country are the same,
#  so they all earn over or all earn under 50k

def Task2():
    # Use hours-per-week, Occupation, Age and relationship to predict income.
    df = pd.read_csv("humanDetails.csv", encoding="ISO-8859-1")
    #print(df['occupation '].value_counts())

    #cleaning
    df['occupation '] = df['occupation '].str.replace('  Prof-specialty', ' Prof-specialty')
    most_frequent_occ = df['occupation '].value_counts().idxmax()
    df['occupation '] = df['occupation '].str.replace('?', most_frequent_occ)

    df['age'] = df['age'].str.replace('90s', '90')
    df['age'] = df['age'].str.replace('80s', '80')
    df['age'] = df['age'].str.replace('70s', '70')
    df['age'] = df['age'].str.replace('60s', '60')
    df['age'] = df['age'].str.replace('50s', '50')
    df['age'] = df['age'].str.replace('40s', '40')
    df['age'] = df['age'].str.replace('20s', '20')
    df['age'] = df['age'].apply(pd.to_numeric, errors='coerce')

    df = df[df["relationship"] != " Other-relative"]

    solo = df["hours-per-week"].value_counts() > 1
    df = df[df["hours-per-week"].isin(solo[solo].index)]
    df['hours-per-week'] = df['hours-per-week'].apply(pd.to_numeric, errors='coerce')

    # Income is either or
    df['Income'] = df['Income'].str.replace('<=50K', '0')
    df['Income'] = df['Income'].str.replace('>50K', '1')

    # assigning int values to occupation variables by dictionary
    all_occupation = np.unique(df['occupation '].astype(str))
    dictO = {}
    c = 1
    for oc in all_occupation:
        dictO[oc] = c
        c = c+1
    df['occupation '] = df['occupation '].map(dictO)

    # assigning int values to relationship variables by dictionary
    all_relationships = np.unique(df['relationship'].astype(str))
    dictR = {}
    c = 1
    for rc in all_relationships:
        dictR[rc] = c
        c = c+1
    df['relationship'] = df['relationship'].map(dictR)

    # data values
    X = (df[["hours-per-week", 'occupation ', 'age', "relationship"]])
    # our class value to find
    y = df[['Income']]

    models = [('KNN', KNeighborsClassifier(n_neighbors=3)), ('DTC', DecisionTreeClassifier(max_depth=3))]
    # evaluate each model in turn
    for name, model in models:
        kfold = StratifiedShuffleSplit(n_splits=5)
        tree_train = []
        tree_test = []
        for train, test in kfold.split(X, y):
            model.fit(X.iloc[train], y.iloc[train])

            tree_train.append(model.score(X.iloc[train], y.iloc[train]))
            tree_test.append(model.score(X.iloc[test], y.iloc[test]))
        print(name)
        print(np.mean(tree_train))
        print(np.mean(tree_test))
        plt.title(name)
        plt.plot(tree_test, label='Test data')
        plt.plot(tree_train, label='Training data')
        plt.xlabel("depth")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
        print("Diagram shows a graph of accuracy")
        # predict income by uncommenting [hours, occupation, age, relationship]
        #print(model.predict([[20, 1, 50, 1]]))

# todo task 2 findings: The model trained with this dataset is not over fitted


def Task3():
    # Use age, fnlwgt, education-num and hours-per-week
    df = pd.read_csv("humanDetails.csv", encoding="ISO-8859-1")
    #print(df['occupation '].value_counts())

    # cleaning
    df['age'] = df['age'].str.replace('90s', '90')
    df['age'] = df['age'].str.replace('80s', '80')
    df['age'] = df['age'].str.replace('70s', '70')
    df['age'] = df['age'].str.replace('60s', '60')
    df['age'] = df['age'].str.replace('50s', '50')
    df['age'] = df['age'].str.replace('40s', '40')
    df['age'] = df['age'].str.replace('20s', '20')
    df['age'] = df['age'].apply(pd.to_numeric, errors='coerce')

    solo = df["hours-per-week"].value_counts() > 1
    df = df[df["hours-per-week"].isin(solo[solo].index)]
    df['hours-per-week'] = df['hours-per-week'].apply(pd.to_numeric, errors='coerce')

    most_frequent_edu = df['education'].value_counts().idxmax()
    df['education'] = df['education'].str.replace('?', most_frequent_edu)

    # assigning int values to relationship variables by dictionary
    all_educations = np.unique(df['education'].astype(str))
    dictE = {}
    c = 1
    for ec in all_educations:
        dictE[ec] = c
        c = c+1
    df['education'] = df['education'].map(dictE)

    X = (df[["age", 'fnlwgt', 'hours-per-week']])

    scalingObj = preprocessing.MinMaxScaler()
    newFLT2 = scalingObj.fit_transform(X)
    costs = []

    pca = PCA(n_components=2)

    pc = pca.fit_transform(newFLT2)

    pDf = pd.DataFrame(data=pc, columns=['a1', 'a2'])

    # finding the elbow
    for i in range(15):
        kmeans = KMeans(n_clusters=i+1).fit(pDf)
        costs.append(kmeans.inertia_)
    plt.plot(costs)
    plt.xticks(range(1, len(costs)+1))
    plt.show()

    # n_clusters is our elbow, unsure how to construct graph/visualisation
    kalg = KMeans(n_clusters=5)
    kalg.fit(X)
    print(kalg.labels_)

# todo task 3 findings: unsure


Task1()

#Task2()

#Task3()
