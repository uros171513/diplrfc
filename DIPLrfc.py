import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np

from sklearn import preprocessing
import datetime

if __name__ == '__main__':

    start_time=datetime.datetime.now()

    #ucitavanje podataka
    names = []
    for n in range(91):
        names.append(str(n+1))

    set = pandas.read_csv("YearPredictionMSD.csv", header=0, names=names)

    for x in names:
        med = np.nanmedian(set[x])
        set[x] = set[x].fillna(med)

    new_set=set

    # sredjivanje prve kolone - label encoding
    le = preprocessing.LabelEncoder()
    le.fit(new_set.values[:,0])
    transformed=le.transform(new_set.values[:,0])
    new_set.drop(['1'], axis=1, inplace=True)
    new_set.insert(0, '1', transformed)

    train_set=new_set[:463715]
    test_set=new_set[463715:]

    # testni podaci
    X_test = test_set.values[:, 1:]
    y_test = test_set.values[:, 0]
    # trening podaci
    X = train_set.values[:, 1:]
    y = train_set.values[:, 0]
    print("prosao podelu 1")

    # deljenje na skupove
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    # deljenje na skupove - kraj
    print("prosao podelu 2")

    # normalizacija
    mean = [np.mean(X_train[:, i:]) for i in range(len(names)-1)]
    std = [np.std(X_train[:, i:]) for i in range(len(names)-1)]
    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std
    X_test = (X_test - mean) / std
    # normalizacija - kraj

    #print(np.any(np.isnan(X_train)))
    #print(np.any(np.isnan(y_train)))
    #print(np.all(np.isfinite(X_train)))
    #print(np.all(np.isfinite(y_train)))

    print("prosao normalizaciju")

    clf = RandomForestClassifier(criterion='gini', max_depth=7, max_features='auto', n_estimators=700)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(f1_score(y_test, predicted, average='micro'))

    df = pandas.DataFrame({'REAL': y_test, 'PREDICTED': predicted})

    end_time=datetime.datetime.now()
