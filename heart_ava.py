import pandas as pd
import math
import scipy as sp
import numpy as np
from random import shuffle
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from os import listdir
from os.path import isfile, join

def filterDataset(data):
    invalidIndexes = []
    for row in data.itertuples():
        if '?' in row:
            invalidIndexes.append(row[0])
    new_data = data.drop(data.index[invalidIndexes])
    cols = list(new_data.columns)
    new_data[cols] = new_data[cols].astype('float32')
    return new_data

def main():
    train = False
    print('Reading data....')
    data = pd.read_csv('processed.cleveland.data', names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num'], header=None)
    print('Removing invalid values from data....')
    data = filterDataset(data)
    ## split data into train and test
    cols = list(data.columns)
    cols.remove('num')
    target = data['num'].copy()
    input_data = data[cols].copy()
    dTrain, dTest, targetTrain, targetTest = train_test_split(input_data, target, test_size=0.20)
    train_index = dTrain.index
    scaler = preprocessing.StandardScaler().fit(dTrain)
    dTrain = scaler.transform(dTrain)
    cols = [i for i in range(dTrain.shape[1])]
    dTrain = pd.DataFrame(dTrain, index=train_index, columns=cols)
    classes = [0,1,2,3,4]
    votes = np.zeros((dTest.shape[0], len(classes)), dtype=np.int)
    clfs = []
    while len(classes) > 1:
        current_class = classes.pop(0)
        for c in classes:
            d = pd.concat([dTrain,targetTrain], axis=1, ignore_index=True)
            d = d.reset_index(drop=True)
            current_group = pd.concat([d[d[13]==current_class], d[d[13]==c]], axis=0, ignore_index=False)
            y = current_group[13].copy()
            cols = list(current_group.columns)
            cols.remove(13)
            x = current_group[cols].copy()

            print('Training model for classes ', current_class, 'and ', c, '...')
            clf = SVC().fit(x, y)
            clfs.append(clf)
            # clf.fit(dTrain, targetTrain)
            # pred = clf.predict(x)
            # for i in range(pred.shape[0]):
            #     votes[x.iloc[[i]].index[0], pred[i]] += 1
    for clf in clfs:
        dTest = scaler.transform(dTest)
        cols = [i for i in range(dTest.shape[1])]
        dTest = pd.DataFrame(dTest, columns=cols)
        clf_pred = clf.predict(dTest)
        for i in range(clf_pred.shape[0]):
            votes[np.int(dTest.iloc[[i]].index[0]), np.int(clf_pred[i])] += 1
    pred = []
    print(votes)
    for i in range(votes.shape[0]):
        pred.append(np.argmax(votes[i,:]))
    cm = confusion_matrix(targetTest, pred, labels=[0,1,2,3,4])
    score = np.sum(np.diagonal(cm))/np.sum(cm)
    print('score: ', score)
    np.savetxt("confusion_matrix_ava.csv", cm, delimiter=",")

if __name__ == "__main__":
    main()
