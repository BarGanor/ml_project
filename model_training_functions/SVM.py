from sklearn.svm import SVC
from model_training_functions.kfold_cross_validation import *
import numpy as np
import pandas as pd


def get_C_SVM(df):
    all_scores = pd.DataFrame()
    for c in np.arange(1, 2, 0.1):
        model = SVC(C=c, kernel='linear', random_state=42, probability=True)
        scores = train_model_by_kfold(df, model)
        scores['C'] = c
        all_scores = all_scores.append(scores, ignore_index=True)
    return  all_scores


def get_decision_function_shape_SVM(df):
    decision_function_shape= ['ovo', 'ovr']
    all_scores = pd.DataFrame()
    for decision in decision_function_shape:
        model =  SVC(C=1, kernel='linear',decision_function_shape=decision, random_state=42, probability=True)
        scores = train_model_by_kfold(df, model)
        scores['decision_function_shape'] = decision
        all_scores = all_scores.append(scores, ignore_index=True)
    return  all_scores

def get_result_coeff_and_intrec(df):
    x=df.drop(columns=['target'])
    y=df['target']
    SVM_best_model= SVC(C=1, kernel='linear',decision_function_shape='ovo', random_state=42, probability=True)
    SVM_best_model.fit(x,y)
    print('Coefficients: \n',SVM_best_model.coef_)
    print('Intercepts: \n', SVM_best_model.intercept_)
    return SVM_best_model