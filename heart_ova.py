import pandas as pd
import math
import scipy as sp
import numpy as np
from random import shuffle
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

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
    scaler = preprocessing.StandardScaler().fit(dTrain)
    dTrain = scaler.transform(dTrain)
    # if train:
    ovr = OneVsRestClassifier(LinearSVC(random_state=0), n_jobs=-1)
    print('Training model...')
    ovr.fit(dTrain, targetTrain)
    #     joblib.dump(ovr, 'oneVsAll.pkl')
    #     print('Model saved!')
    # else:
    #     ovr = joblib.load('oneVsAll.pkl')
    dTest = scaler.transform(dTest)
    pred = ovr.predict(dTest)
    print(ovr.score(dTest, targetTest))
    cm = confusion_matrix(targetTest, pred, labels=[0,1,2,3,4])
    np.savetxt("confusion_matrix_ova.csv", cm, delimiter=",")

if __name__ == "__main__":
    main()
