from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd


def train_model_by_kfold(df, model):
    kf = KFold(n_splits=10)
    X = df.drop(columns=['target'])
    y = df['target']
    scores = pd.DataFrame()
    for train_index, test_index in kf.split(df):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train,y_train)
        training_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        scores.append({'training_score': [training_score], 'test_score':[test_score]}, ignore_index=True)

    
    print('\n\nTest accuracy: %.3f +/- %.3f' % (scores['test_score'].mean(), scores['test_score'].std()))
    print('\n\nTraining accuracy: %.3f +/- %.3f' % (scores['training_score'].mean(), scores['training_score'].std()))
    return scores


def train_model_by_kfold_depth(df, model):
    kf = KFold(n_splits=10)
    X = df.drop(columns=['target'])
    y = df['target']
    scores = []
    for train_index, test_index in kf.split(df):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train,y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        print(score)
    
    print('\n\nCross-Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    return  np.mean(scores)