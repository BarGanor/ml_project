from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import KFold

def evaluation_model(df, model,model_name):
    kf = KFold(n_splits=10)
    X = df.drop(columns=['target'])
    y = df['target']
    scores = pd.DataFrame()
    scores.append({'model':model_name},ignore_index=True)
    scores_model = pd.DataFrame()
    for train_index, test_index in kf.split(df):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        # roc_auc scores
        training_score_roc_auc = roc_auc_score(y_true=y_train, y_score=model.predict_proba(X_train)[:, 1])
        test_score_roc_auc = roc_auc_score(y_true=y_test, y_score=model.predict_proba(X_test)[:, 1])
        #accuracy
        training_score_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_score_accuracy = accuracy_score(y_test, model.predict(X_test))
        #precision
        training_score_precision = precision_score(y_train, model.predict(X_train),average='binary')
        test_score_precision = precision_score(y_test,model.predict(X_test),average='binary')
        #recall
        training_score_recall = recall_score(y_train, model.predict(X_train),average='binary')
        test_score_recall = recall_score(y_test,model.predict(X_test),average='binary')
        #F1
        training_score_F1 = f1_score(y_train, model.predict(X_train),average='binary')
        test_score_F1 = f1_score(y_test, model.predict(X_test),average='binary')
        #scores apend
        scores_model = scores_model.append({'training score roc auc': training_score_roc_auc,
                                            'test score roc auc': test_score_roc_auc,
                                            'training score accuracy': training_score_accuracy,
                                            'test score accuracy': test_score_accuracy,
                                            'training score precision': training_score_precision,
                                            'test_score_precision': test_score_precision,
                                            'training_score_recall': training_score_recall,
                                            'test_score_recall': test_score_recall,
                                            'training_score_F1': training_score_F1,
                                            'test_score_F1': test_score_F1}, ignore_index=True)
    scores =pd.concat([scores, scores_model.mean()], axis=1)
    return scores