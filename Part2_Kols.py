# -----------------------------------------------------------------------------
# PART B
# -----------------------------------------------------------------------------

# Import Packages:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.tree import DecisionTreeClassifier ,plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Loading the Final Dataset From Part A:
XYtrain =pd.read_csv('C:/Users/kolsk/PycharmProjects/MachineLearningPartB/PartAFinal.csv')

# Selecting the Features from the FeatureSelection:
XYtrain_PartB = XYtrain[["view_count","comment_count", "dislikes", "likes", "title_char_count",'Airtime','category_ratio']]

# Converse the target variable to Classification mission
pd.options.mode.chained_assignment = None
for i in XYtrain_PartB.index:
    if XYtrain_PartB.loc[i, 'view_count'] <= 3417816.0:
        XYtrain_PartB.loc[i, 'view_count'] = '0'
    elif XYtrain_PartB.loc[i, 'view_count'] <= 6235006.66:
        XYtrain_PartB.loc[i, 'view_count'] = '1'
    else:
        XYtrain_PartB.loc[i, 'view_count'] = '2'

# Dividing The DataSets to X variables and y vector + making dummy in the x's:
Y_Train = XYtrain_PartB['view_count']
X_Train = XYtrain_PartB[['comment_count', 'dislikes', 'likes', 'title_char_count', 'Airtime', 'category_ratio']]
X_Train = pd.get_dummies(X_Train, columns=['category_ratio'])

# Make Normalized Dataset Based on MIN-MAX Normalize
# Make a copy of the X_TRAIN_DT and Y_TRAIN_DT to save normalized and un-normalized dataset:
scaler = MinMaxScaler()
X_Train_Normalize = scaler.fit_transform(X_Train)
X_Train_Normalize = pd.DataFrame(X_Train_Normalize)
Y_Train_Normalize = Y_Train

# -----------------------------------------------------------------------------
# Decision Tree
# -----------------------------------------------------------------------------
decision_tree = DecisionTreeClassifier(random_state=42)

parameter_options_DT = {"criterion": ['gini', 'entropy'],
                        "max_depth": range(1, 10),
                        "max_features": ['auto', 'sqrt', 'log2', None]}

grid_DT = GridSearchCV(decision_tree, param_grid=parameter_options_DT, cv=10,
                       return_train_score=True, refit=True)
grid_DT.fit(X_Train, Y_Train)

best_params_DT = grid_DT.best_params_
all_score_DT = pd.DataFrame({'param': grid_DT.cv_results_["params"],
                             'Val_Accuracy': grid_DT.cv_results_["mean_test_score"],
                             'Train_Accuracy': grid_DT.cv_results_["mean_train_score"]})
best_score_DT = all_score_DT[all_score_DT['param'] == best_params_DT]
best_DT_model = grid_DT.best_estimator_

# Post Pruning:
path = best_DT_model.cost_complexity_pruning_path(X_Train, Y_Train)
alphas = path.ccp_alphas

parameter_options_Prune = {"ccp_alpha": alphas}
gridPrune = GridSearchCV(best_DT_model, param_grid=parameter_options_Prune, cv=10,
                         return_train_score=True, refit=True)
gridPrune.fit(X_Train, Y_Train)

best_params_prune = gridPrune.best_params_
all_score_Prune = pd.DataFrame({'param': gridPrune.cv_results_["params"],
                                'Val_Accuracy': gridPrune.cv_results_["mean_test_score"],
                                'Train_Accuracy': gridPrune.cv_results_["mean_train_score"]})
best_score_Prune = all_score_Prune[all_score_Prune['param'] == best_params_prune]
best_DT_model_Prune = gridPrune.best_estimator_

# DT Draw:
plt.figure(figsize=(12, 10))
plot_tree(best_DT_model_Prune,max_depth=2, filled=True, class_names=True,feature_names=X_Train.columns[:])
plt.title("Decision Tree Model")
plt.show()



# Feature Importance:
importance = best_DT_model_Prune.feature_importances_
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))

plt.figure(figsize=(15, 7))
pd.Series(best_DT_model_Prune.feature_importances_).plot.bar(color='steelblue')
plt.bar(range(len(best_DT_model_Prune.feature_importances_)), best_DT_model_Prune.feature_importances_)
plt.xticks([0,1,2,3,4,5,6,7],X_Train.columns[:],rotation=30)
plt.tick_params(axis='both', which='minor', labelsize=4)
plt.ylabel('Feature_importance', fontsize=15)
plt.xlabel('Features', fontsize=15)
plt.title('DT Feature Importance')
plt.show()

# -----------------------------------------------------------------------------
# ANN(MLP)
# -----------------------------------------------------------------------------
# default
res = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
model = MLPClassifier(verbose=True)
for train_idx, val_idx in kfold.split(X_Train_Normalize):
    model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
    accuracy_training = accuracy_score(Y_Train_Normalize.iloc[train_idx],
                                       model.predict(X_Train_Normalize.iloc[train_idx]))
    accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx],
                                         model.predict(X_Train_Normalize.iloc[val_idx]))
    res = res.append({'AccuracyTraining': accuracy_training, 'AccuracyValidation': accuracy_validation},
                     ignore_index=True)

ANNAccTraining = res['AccuracyTraining'].mean()
ANNAccValidation = res['AccuracyValidation'].mean()
FinalResultsANN = pd.DataFrame()
FinalResultsANN = FinalResultsANN.append({'AccuracyTraining': ANNAccTraining ,
                                          'AccuracyValidation': ANNAccValidation}, ignore_index=True)

# Hyper parameter tuning

# Tuning MaxIter:
resMaxIter = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
for maxiter in [200, 300, 400, 600, 700, 800, 900, 1000]:
    resTemp = pd.DataFrame()
    model = MLPClassifier(random_state=1,
                          max_iter=maxiter,
                          learning_rate_init=0.01,
                          tol=0.00001,
                          n_iter_no_change=20,
                          verbose=True
                          )
    for train_idx, val_idx in kfold.split(X_Train_Normalize):
        model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
        accuracy_training = accuracy_score(Y_Train_Normalize.iloc[train_idx], model.predict(X_Train_Normalize.iloc[train_idx]))
        accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx], model.predict(X_Train_Normalize.iloc[val_idx]))
        resTemp = resTemp.append({'AccuracyTraining': accuracy_training, 'AccuracyValidation': accuracy_validation},
                                 ignore_index=True)
    ANNAccTraining = resTemp['AccuracyTraining'].mean()
    ANNAccValidation = resTemp['AccuracyValidation'].mean()
    resMaxIter = resMaxIter.append({'maxIterations': maxiter, 'AccuracyTraining': ANNAccTraining,
                                    'AccuracyValidation': ANNAccValidation}, ignore_index=True)

plt.figure(figsize=(7, 4))
plt.plot(resMaxIter['maxIterations'], resMaxIter['AccuracyTraining'], label='Train')
plt.plot(resMaxIter['maxIterations'], resMaxIter['AccuracyValidation'], label='Validation')
plt.legend()
plt.show()
plt.xlabel('maxIterations')
plt.ylabel('Accuracy')
plt.title('Tuning the number of maximum iterations')

# Tuning 1 layer Neurons over big range
resHidenTune = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
for size_ in range(1, 200, 10):
    resTemp = pd.DataFrame()
    model = MLPClassifier(random_state=1,
                          hidden_layer_sizes=(size_,),
                          max_iter=400,
                          learning_rate_init=0.01,
                          tol=0.00001,
                          n_iter_no_change=20,
                          verbose=True
                          )
    for train_idx, val_idx in kfold.split(X_Train_Normalize):
        model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
        accuracy_training = accuracy_score(Y_Train_Normalize.iloc[train_idx], model.predict(X_Train_Normalize.iloc[train_idx]))
        accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx], model.predict(X_Train_Normalize.iloc[val_idx]))
        resTemp = resTemp.append({'AccuracyTraining': accuracy_training, 'AccuracyValidation': accuracy_validation},
                         ignore_index=True)
    ANNAccTraining = resTemp['AccuracyTraining'].mean()
    ANNAccValidation = resTemp['AccuracyValidation'].mean()
    resHidenTune = resHidenTune.append({'HiddenLayersNeurons': size_, 'AccuracyTraining':ANNAccTraining , 'AccuracyValidation': ANNAccValidation},ignore_index=True)

plt.figure(figsize=(7, 4))
plt.plot(resHidenTune['HiddenLayersNeurons'], resHidenTune['AccuracyTraining'], label='Train')
plt.plot(resHidenTune['HiddenLayersNeurons'], resHidenTune['AccuracyValidation'], label='Validation')
plt.legend()
plt.show()
plt.xlabel('Neurons')
plt.ylabel('Accuracy')
plt.title('Tuning the number of neurons in 1 layer network')

# Tuning 1 layer Neurons over small range
resHidenTune_smallRange = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
for size_ in range(1, 11, 1):
    resTemp = pd.DataFrame()
    model = MLPClassifier(random_state=1,
                          hidden_layer_sizes=(size_,),
                          max_iter=400,
                          learning_rate_init=0.01,
                          tol=0.00001,
                          n_iter_no_change=20,
                          verbose=True
                          )
    for train_idx, val_idx in kfold.split(X_Train_Normalize):
        model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
        accuracy_training = accuracy_score(Y_Train_Normalize.iloc[train_idx], model.predict(X_Train_Normalize.iloc[train_idx]))
        accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx], model.predict(X_Train_Normalize.iloc[val_idx]))
        resTemp = resTemp.append({'AccuracyTraining': accuracy_training, 'AccuracyValidation': accuracy_validation},
                         ignore_index=True)
    ANNAccTraining = resTemp['AccuracyTraining'].mean()
    ANNAccValidation = resTemp['AccuracyValidation'].mean()
    resHidenTune_smallRange = resHidenTune_smallRange.append({'HiddenLayersNeurons': size_,
                                                              'AccuracyTraining':ANNAccTraining ,
                                                              'AccuracyValidation': ANNAccValidation},
                                                             ignore_index=True)

plt.figure(figsize=(7, 4))
plt.plot(resHidenTune_smallRange['HiddenLayersNeurons'],
         resHidenTune_smallRange['AccuracyTraining'], label='Train')
plt.plot(resHidenTune_smallRange['HiddenLayersNeurons'],
         resHidenTune_smallRange['AccuracyValidation'], label='Validation')
plt.legend()
plt.show()
plt.xlabel('Neurons')
plt.ylabel('Accuracy')
plt.title('Tuning the number of neurons in 1 layer network')

# Tuning Second Layer Neurons:
resScndHidenLayer = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
for scndlayerneruons in range(1, 11, 1):
    resTemp = pd.DataFrame()
    model = MLPClassifier(random_state=1,
                          hidden_layer_sizes=(11,scndlayerneruons),
                          max_iter=400,
                          learning_rate_init=0.01,
                          tol=0.00001,
                          n_iter_no_change=20,
                          verbose=True
                          )
    for train_idx, val_idx in kfold.split(X_Train_Normalize):
        model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
        accuracy_training = accuracy_score(Y_Train_Normalize.iloc[train_idx], model.predict(X_Train_Normalize.iloc[train_idx]))
        accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx], model.predict(X_Train_Normalize.iloc[val_idx]))
        resTemp = resTemp.append({'AccuracyTraining': accuracy_training, 'AccuracyValidation': accuracy_validation},
                         ignore_index=True)
    ANNAccTraining = resTemp['AccuracyTraining'].mean()
    ANNAccValidation = resTemp['AccuracyValidation'].mean()
    resScndHidenLayer = resScndHidenLayer.append({'SecondLayerNeurons': scndlayerneruons, 'AccuracyTraining':ANNAccTraining , 'AccuracyValidation': ANNAccValidation},ignore_index=True)

plt.figure(figsize=(7,4))
plt.plot(resScndHidenLayer['SecondLayerNeurons'] ,resScndHidenLayer['AccuracyTraining'],label='Train')
plt.plot(resScndHidenLayer['SecondLayerNeurons'] ,resScndHidenLayer['AccuracyValidation'],label='Validation')
plt.legend()
plt.xticks(np.arange(1,4,1))
plt.show()
plt.xlabel('SecondLayerNeurons')
plt.ylabel('Accuracy')
plt.title('Tuning the number of neurons in 2 layer network')

# Tuning Activation Function:
resActiv = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
for activation in ['identity','logistic','tanh','relu']:
    resTemp = pd.DataFrame()
    model = MLPClassifier(random_state=1,
                          hidden_layer_sizes=(11),
                          activation=activation,
                          max_iter=400,
                          learning_rate_init=0.01,
                          tol=0.00001,
                          n_iter_no_change=20,
                          verbose=True
                          )
    for train_idx, val_idx in kfold.split(X_Train_Normalize):
        model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
        accuracy_training = accuracy_score(Y_Train_Normalize.iloc[train_idx],
                                           model.predict(X_Train_Normalize.iloc[train_idx]))
        accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx],
                                             model.predict(X_Train_Normalize.iloc[val_idx]))
        resTemp = resTemp.append({'AccuracyTraining': accuracy_training,
                                  'AccuracyValidation': accuracy_validation},
                                    ignore_index=True)

    ANNAccTraining = resTemp['AccuracyTraining'].mean()
    ANNAccValidation = resTemp['AccuracyValidation'].mean()
    resActiv = resActiv.append({'ActivationFunction': activation, 'AccuracyTraining':ANNAccTraining ,
                                'AccuracyValidation': ANNAccValidation},ignore_index=True)

plt.figure(figsize=(7, 4))
plt.plot(resActiv['ActivationFunction'], resActiv['AccuracyTraining'], label='Train')
plt.plot(resActiv['ActivationFunction'], resActiv['AccuracyValidation'], label='Validation')
plt.legend()
plt.show()
plt.xlabel('ActivationFunction')
plt.ylabel('Accuracy')
plt.title('Tuning the activation function')

# Final Model ANN:
Finalres = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
model = MLPClassifier(random_state=1,
                      hidden_layer_sizes=(11),
                      activation='relu',
                      max_iter=400,
                      learning_rate_init=0.01,
                      tol=0.00001,
                      n_iter_no_change=20,
                      verbose=True
                      )
for train_idx, val_idx in kfold.split(X_Train_Normalize):
    model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
    accuracy_training = accuracy_score(Y_Train_Normalize.iloc[train_idx],model.predict(X_Train_Normalize.iloc[train_idx]))
    accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx],model.predict(X_Train_Normalize.iloc[val_idx]))
    Finalres= Finalres.append({'AccuracyTraining':accuracy_training ,'AccuracyValidation': accuracy_validation},ignore_index=True)

ANNAccTrainingAfterTune = Finalres['AccuracyTraining'].mean()
ANNAccValidationAfterTune = Finalres['AccuracyValidation'].mean()
FinalResultsANNAfterTune = pd.DataFrame()
FinalResultsANNAfterTune = FinalResultsANNAfterTune.append({'AccuracyTraining': ANNAccTrainingAfterTune , 'AccuracyValidation': ANNAccValidationAfterTune}, ignore_index=True)

# Confusion Matrix :
Final_Confusion_Matrix = np.zeros((3,3),dtype=int)

kfold = KFold(n_splits=10, shuffle=True, random_state=123)
model = MLPClassifier(random_state=1,
                      hidden_layer_sizes=(11),
                      activation='relu',
                      max_iter=400,
                      learning_rate_init=0.01,
                      tol=0.00001,
                      n_iter_no_change=20,
                      verbose=True
                      )
for train_idx, val_idx in kfold.split(X_Train_Normalize):
    model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
    confusion_temp =confusion_matrix(y_true=Y_Train_Normalize.iloc[val_idx] , y_pred=model.predict(X_Train_Normalize.iloc[val_idx]))
    for i in range(0,3,1):
        for j in range(0,3,1):
            Final_Confusion_Matrix[i,j] += confusion_temp[i,j]

Final_Confusion_Matrix = Final_Confusion_Matrix/10
Final_Confusion_Matrix = Final_Confusion_Matrix.round()
ax= plt.subplot()
sns.heatmap(Final_Confusion_Matrix, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix');


# -----------------------------------------------------------------------------
# SVM
# -----------------------------------------------------------------------------
SVM_model = SVC(kernel='linear', random_state=42)

param_grid = {'C': np.arange(1, 2, 0.1),
              'decision_function_shape': ['ovo', 'ovr']
              }

grid_search_SVM = GridSearchCV(estimator=SVM_model,
                               param_grid=param_grid,
                               refit=True,
                               cv=10,
                               return_train_score=True)

grid_search_SVM.fit(X_Train_Normalize, Y_Train_Normalize)

best_params_SVM = grid_search_SVM.best_params_
all_score_SVM = pd.DataFrame({'param': grid_search_SVM.cv_results_["params"],
                              'Val_Accuracy': grid_search_SVM.cv_results_["mean_test_score"],
                              'Train_Accuracy': grid_search_SVM.cv_results_["mean_train_score"]})
best_score_SVM = all_score_SVM[all_score_SVM['param'] == best_params_SVM]

best_model_SVM = grid_search_SVM.best_estimator_

print('Coefficients: \n', best_model_SVM.coef_)
print('Intercepts: \n', best_model_SVM.intercept_)
# -----------------------------------------------------------------------------
# K-Means
# -----------------------------------------------------------------------------
# graphs to understand which features are more informative
sns.pairplot(XYtrain_PartB, hue='view_count')
plt.show()

# 4 featurs
X_Train_Normalize_4Features = pd.DataFrame(X_Train_Normalize, columns=[0, 1, 2, 5, 6, 7])

# Categorical treatment
category_ratio_Kmeans_values = []
for i in X_Train_Normalize_4Features.index:
    if X_Train_Normalize_4Features.loc[i, 5] == 1:
        category_ratio_Kmeans_values.append(1)
    elif X_Train_Normalize_4Features.loc[i, 6] == 1:
        category_ratio_Kmeans_values.append(0)
    else:
        category_ratio_Kmeans_values.append(0.5)

# organize the data
X_Train_Normalize_4Features = pd.DataFrame(X_Train_Normalize_4Features, columns=[0, 1, 2])
X_Train_Normalize_4Features[3] = category_ratio_Kmeans_values

# Graphs of Kmeans measures
iner_list = []
dbi_list = []
sil_list = []
for n_clusters in range(2, 10, 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_Train_Normalize_4Features)
    assignment = kmeans.predict(X_Train_Normalize_4Features)

    iner = kmeans.inertia_
    sil = silhouette_score(X_Train_Normalize_4Features, assignment)
    dbi = davies_bouldin_score(X_Train_Normalize_4Features, assignment)

    dbi_list.append(dbi)
    sil_list.append(sil)
    iner_list.append(iner)

plt.plot(range(2, 10, 1), iner_list, marker='o')
plt.title("Inertia")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), dbi_list, marker='o')
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show()

# results
kmeans_4features = KMeans(n_clusters=3, random_state=42)
kmeans_4features.fit(X_Train_Normalize_4Features)
XYtrain_Kmeans_4features = pd.DataFrame(XYtrain_PartB, columns=['view_count', 'comment_count',
                                                                'dislikes', 'likes', 'category_ratio'])
XYtrain_Kmeans_4features['cluster_index'] = kmeans_4features.labels_

# -----------------------------------------------------------------------------
# Model Evaluation
# -----------------------------------------------------------------------------
# ANN
Evaluation_ANN_score = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
model = MLPClassifier(random_state=1,
                      hidden_layer_sizes=(11),
                      activation='relu',
                      max_iter=400,
                      learning_rate_init=0.01,
                      tol=0.00001,
                      n_iter_no_change=20,
                      verbose=True
                      )
for train_idx, val_idx in kfold.split(X_Train_Normalize):
    model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
    precision_validation = precision_score(Y_Train_Normalize.iloc[val_idx],
                                          model.predict(X_Train_Normalize.iloc[val_idx]),
                                          average='macro')
    recall_validation = recall_score(Y_Train_Normalize.iloc[val_idx],
                                     model.predict(X_Train_Normalize.iloc[val_idx]),
                                     average='macro')
    accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx],
                                         model.predict(X_Train_Normalize.iloc[val_idx]))
    Evaluation_ANN_score = Evaluation_ANN_score.append({'AccuracyValidation': accuracy_validation,
                                                        'PrecisionValidation': precision_validation,
                                                        'RecallValidation': recall_validation}, ignore_index=True)

Evaluation_ANN_mean_accuracy = Evaluation_ANN_score['AccuracyValidation'].mean()
Evaluation_ANN_mean_precision = Evaluation_ANN_score['PrecisionValidation'].mean()
Evaluation_ANN_mean_recall = Evaluation_ANN_score['RecallValidation'].mean()
Evaluation_ANN_measures = pd.DataFrame()
Evaluation_ANN_measures = Evaluation_ANN_measures.append({'Accuracy_ANN': Evaluation_ANN_mean_accuracy,
                                                          'Precision_ANN': Evaluation_ANN_mean_precision,
                                                          'Recall_ANN': Evaluation_ANN_mean_recall},
                                                         ignore_index=True)

# SVM
Evaluation_SVM_score = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
model = SVC(kernel='linear', random_state=42, C=1.9, decision_function_shape='ovo')
for train_idx, val_idx in kfold.split(X_Train_Normalize):
    model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
    precision_validation = precision_score(Y_Train_Normalize.iloc[val_idx],
                                           model.predict(X_Train_Normalize.iloc[val_idx]),
                                           average='macro')
    recall_validation = recall_score(Y_Train_Normalize.iloc[val_idx],
                                     model.predict(X_Train_Normalize.iloc[val_idx]),
                                     average='macro')
    accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx],
                                         model.predict(X_Train_Normalize.iloc[val_idx]))
    Evaluation_SVM_score = Evaluation_SVM_score.append({'AccuracyValidation': accuracy_validation,
                                                        'PrecisionValidation':precision_validation,
                                                        'RecallValidation':recall_validation}, ignore_index=True)

Evaluation_SVM_mean_accuracy = Evaluation_SVM_score['AccuracyValidation'].mean()
Evaluation_SVM_mean_precision = Evaluation_SVM_score['PrecisionValidation'].mean()
Evaluation_SVM_mean_recall = Evaluation_SVM_score['RecallValidation'].mean()
Evaluation_SVM_measures = pd.DataFrame()
Evaluation_SVM_measures = Evaluation_SVM_measures.append({'Accuracy_SVM': Evaluation_SVM_mean_accuracy,
                                                          'Precision_SVM': Evaluation_SVM_mean_precision,
                                                          'Recall_SVM': Evaluation_SVM_mean_recall},
                                                         ignore_index=True)

# Decision Tree
Evaluation_DT_score = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
model = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=7, max_features='log2',
                               ccp_alpha=0.004592169614071162)
for train_idx, val_idx in kfold.split(X_Train_Normalize):
    model.fit(X_Train_Normalize.iloc[train_idx], Y_Train_Normalize.iloc[train_idx])
    precision_validation = precision_score(Y_Train_Normalize.iloc[val_idx],
                                           model.predict(X_Train_Normalize.iloc[val_idx]),
                                           average='macro')
    recall_validation = recall_score(Y_Train_Normalize.iloc[val_idx],
                                     model.predict(X_Train_Normalize.iloc[val_idx]),
                                     average='macro')
    accuracy_validation = accuracy_score(Y_Train_Normalize.iloc[val_idx],
                                         model.predict(X_Train_Normalize.iloc[val_idx]))
    Evaluation_DT_score = Evaluation_DT_score.append({'AccuracyValidation': accuracy_validation,
                                                      'PrecisionValidation': precision_validation,
                                                      'RecallValidation': recall_validation},ignore_index=True)

Evaluation_DT_mean_accuracy = Evaluation_DT_score['AccuracyValidation'].mean()
Evaluation_DT_mean_precision = Evaluation_DT_score['PrecisionValidation'].mean()
Evaluation_DT_mean_recall = Evaluation_DT_score['RecallValidation'].mean()
Evaluation_DT_measures = pd.DataFrame()
Evaluation_DT_measures = Evaluation_DT_measures.append({'Accuracy_DT': Evaluation_DT_mean_accuracy,
                                                        'Precision_DT': Evaluation_DT_mean_precision,
                                                        'Recall_DT': Evaluation_DT_mean_recall}, ignore_index=True)

# -----------------------------------------------------------------------------
# Model Improvement
# -----------------------------------------------------------------------------
XY_Train_Improvement = XYtrain[['view_count', 'comment_count', 'dislikes', 'likes', 'title_char_count',
                                'Airtime', 'category_ratio','categoryId']]

# complete zeros values in numeric features
commentMedian = XY_Train_Improvement['comment_count'].median()
likesMedian = XY_Train_Improvement['likes'].median()
dislikesMedian = XY_Train_Improvement['dislikes'].median()

pd.options.mode.chained_assignment = None
for i in XY_Train_Improvement.index:
    if XY_Train_Improvement.loc[i, 'comment_count'] == 0:
        XY_Train_Improvement.loc[i, 'comment_count'] = commentMedian
    if XY_Train_Improvement.loc[i, 'likes'] == 0:
        XY_Train_Improvement.loc[i, 'likes'] = likesMedian
    if XY_Train_Improvement.loc[i, 'dislikes'] == 0:
        XY_Train_Improvement.loc[i, 'dislikes'] = dislikesMedian

XY_Train_Improvement['comment_count'] = np.log(XY_Train_Improvement['comment_count'])
XY_Train_Improvement['likes'] = np.log(XY_Train_Improvement['likes'])
XY_Train_Improvement['dislikes'] = np.log(XY_Train_Improvement['dislikes'])




# Converse the target variable to Classification mission
for i in XY_Train_Improvement.index:
    if XY_Train_Improvement.loc[i, 'view_count'] <= 3417816.0:
        XY_Train_Improvement.loc[i, 'view_count'] = '0'
    elif XY_Train_Improvement.loc[i, 'view_count'] <= 6235006.66:
        XY_Train_Improvement.loc[i, 'view_count'] = '1'
    else:
        XY_Train_Improvement.loc[i, 'view_count'] = '2'




# Dividing The DataSets to X variables and y vector + making dummy in the x's:
Y_Train_Improvement = XY_Train_Improvement['view_count']
X_Train_Improvement = XY_Train_Improvement[['comment_count', 'dislikes', 'likes', 'title_char_count', 'Airtime', 'category_ratio','categoryId']]
X_Train_Improvement = pd.get_dummies(X_Train_Improvement, columns=['category_ratio'])
X_Train_Improvement = pd.get_dummies(X_Train_Improvement, columns=['categoryId'])

# Make Normalized Dataset Based on MIN-MAX Normalize
# Make a copy of the X_TRAIN_DT and Y_TRAIN_DT to save normalized and un-normalized dataset:
scaler = StandardScaler()
X_Train_Improvement[['comment_count', 'dislikes', 'likes', 'title_char_count', 'Airtime']] = scaler.fit_transform(X_Train_Improvement[['comment_count', 'dislikes', 'likes', 'title_char_count', 'Airtime']])
X_Train_ImprovementNormalize =X_Train_Improvement
X_Train_ImprovementNormalize = pd.DataFrame(X_Train_ImprovementNormalize)
Y_Train_ImprovementNormalize = Y_Train_Improvement


# improvement Tuning 1 layer Neurons:
resHidenTune = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
for size_ in range(1, 201, 10):
    resTemp = pd.DataFrame()
    model = MLPClassifier(random_state=1,
                          hidden_layer_sizes=(size_,),
                          max_iter=400,
                          learning_rate_init=0.01,
                          tol=0.00001,
                          n_iter_no_change=20,
                          verbose=True
                          )
    for train_idx, val_idx in kfold.split(X_Train_ImprovementNormalize):
        model.fit(X_Train_ImprovementNormalize.iloc[train_idx], Y_Train_ImprovementNormalize.iloc[train_idx])
        accuracy_training = accuracy_score(Y_Train_ImprovementNormalize.iloc[train_idx],
                                           model.predict(X_Train_ImprovementNormalize.iloc[train_idx]))
        accuracy_validation = accuracy_score(Y_Train_ImprovementNormalize.iloc[val_idx],
                                             model.predict(X_Train_ImprovementNormalize.iloc[val_idx]))
        resTemp = resTemp.append({'AccuracyTraining': accuracy_training,
                                  'AccuracyValidation': accuracy_validation},
                                 ignore_index=True)
    ANNAccTraining = resTemp['AccuracyTraining'].mean()
    ANNAccValidation = resTemp['AccuracyValidation'].mean()
    resHidenTune = resHidenTune.append({'HiddenLayersNeurons': size_,
                                        'AccuracyTraining': ANNAccTraining,
                                        'AccuracyValidation': ANNAccValidation}, ignore_index=True)

plt.figure(figsize=(7, 4))
plt.plot(resHidenTune['HiddenLayersNeurons'], resHidenTune['AccuracyTraining'], label='Train')
plt.plot(resHidenTune['HiddenLayersNeurons'], resHidenTune['AccuracyValidation'], label='Validation')
plt.legend()
plt.show()
plt.xlabel('Neurons')
plt.ylabel('Accuracy')
plt.title('Tuning the number of neurons in 1 hidden layer network')

# improvement Tuning 2 layer Neurons:
resScndHidenLayer_Improvement = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
for scndlayerneruons in range(1, 81, 5):
    resTemp = pd.DataFrame()
    model = MLPClassifier(random_state=1,
                          hidden_layer_sizes=(51, scndlayerneruons),
                          max_iter=400,
                          learning_rate_init=0.01,
                          tol=0.00001,
                          n_iter_no_change=20,
                          verbose=True
                          )
    for train_idx, val_idx in kfold.split(X_Train_ImprovementNormalize):
        model.fit(X_Train_ImprovementNormalize.iloc[train_idx], Y_Train_ImprovementNormalize.iloc[train_idx])
        accuracy_training = accuracy_score(Y_Train_ImprovementNormalize.iloc[train_idx],
                                           model.predict(X_Train_ImprovementNormalize.iloc[train_idx]))
        accuracy_validation = accuracy_score(Y_Train_ImprovementNormalize.iloc[val_idx],
                                             model.predict(X_Train_ImprovementNormalize.iloc[val_idx]))
        resTemp = resTemp.append({'AccuracyTraining': accuracy_training,
                                  'AccuracyValidation': accuracy_validation},
                                 ignore_index=True)
    ANNAccTraining = resTemp['AccuracyTraining'].mean()
    ANNAccValidation = resTemp['AccuracyValidation'].mean()
    resScndHidenLayer_Improvement = resScndHidenLayer_Improvement.append({'HiddenLayersNeurons': scndlayerneruons,
                                        'AccuracyTraining': ANNAccTraining,
                                        'AccuracyValidation': ANNAccValidation}, ignore_index=True)

plt.figure(figsize=(7, 4))
plt.plot(resScndHidenLayer_Improvement['HiddenLayersNeurons'],
         resScndHidenLayer_Improvement['AccuracyTraining'], label='Train')
plt.plot(resScndHidenLayer_Improvement['HiddenLayersNeurons'],
         resScndHidenLayer_Improvement['AccuracyValidation'], label='Validation')
plt.legend()
plt.show()
plt.xlabel('Neurons')
plt.ylabel('Accuracy')
plt.title('Tuning the number of neurons in 2 hidden layer network')

# best improvement model after data changing
Finalres_improvement = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
model_improvement = MLPClassifier(random_state=1,
                                  hidden_layer_sizes=(51),
                                  max_iter=400,
                                  learning_rate_init=0.01,
                                  tol=0.00001,
                                  n_iter_no_change=20,
                                  verbose=True
                                  )
for train_idx, val_idx in kfold.split(X_Train_ImprovementNormalize):
    model_improvement.fit(X_Train_ImprovementNormalize.iloc[train_idx], Y_Train_ImprovementNormalize.iloc[train_idx])
    accuracy_training = accuracy_score(Y_Train_ImprovementNormalize.iloc[train_idx],
                                       model_improvement.predict(X_Train_ImprovementNormalize.iloc[train_idx]))
    accuracy_validation = accuracy_score(Y_Train_ImprovementNormalize.iloc[val_idx],
                                         model_improvement.predict(X_Train_ImprovementNormalize.iloc[val_idx]))
    Finalres_improvement = Finalres_improvement.append({'AccuracyTraining': accuracy_training,
                                                        'AccuracyValidation': accuracy_validation},
                                                       ignore_index=True)

ANNAccTrainingAfterTune_improvement = Finalres_improvement['AccuracyTraining'].mean()
ANNAccValidationAfterTune_improvement = Finalres_improvement['AccuracyValidation'].mean()
FinalResultsANNAfterTune_improvement = pd.DataFrame()
FinalResultsANNAfterTune_improvement = FinalResultsANNAfterTune_improvement.append({'AccuracyTraining': ANNAccTrainingAfterTune_improvement,
                                                                                    'AccuracyValidation': ANNAccValidationAfterTune_improvement},
                                                                                   ignore_index=True)

# best improvement model after model changing - dealing with muliply rows with the same title
#solver to sgd , tolerance down to 0.000001 , max iterations up to 1000 , learning rate - adaptive
Finalres_improvement2 = pd.DataFrame()
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
model_improvement2 = MLPClassifier(random_state=1,
                          hidden_layer_sizes=(51),
                          max_iter=1000,
                          solver = 'sgd',
                          learning_rate_init=0.01,
                          tol=0.000001,
                          n_iter_no_change=20,
                          verbose=True,
                          learning_rate='adaptive'
                          )
for train_idx, val_idx in kfold.split(X_Train_ImprovementNormalize):
    model_improvement2.fit(X_Train_ImprovementNormalize.iloc[train_idx], Y_Train_ImprovementNormalize.iloc[train_idx])
    accuracy_training = accuracy_score(Y_Train_ImprovementNormalize.iloc[train_idx],
                                       model_improvement2.predict(X_Train_ImprovementNormalize.iloc[train_idx]))
    accuracy_validation = accuracy_score(Y_Train_ImprovementNormalize.iloc[val_idx],
                                         model_improvement2.predict(X_Train_ImprovementNormalize.iloc[val_idx]))
    Finalres_improvement2 = Finalres_improvement2.append({'AccuracyTraining': accuracy_training,
                                                        'AccuracyValidation': accuracy_validation},
                                                       ignore_index=True)

ANNAccTrainingAfterTune_improvement2 = Finalres_improvement2['AccuracyTraining'].mean()
ANNAccValidationAfterTune_improvement2 = Finalres_improvement2['AccuracyValidation'].mean()
FinalResultsANNAfterTune_improvement2 = pd.DataFrame()
FinalResultsANNAfterTune_improvement2 = FinalResultsANNAfterTune_improvement2.append({'AccuracyTraining': ANNAccTrainingAfterTune_improvement2,
                                                                                    'AccuracyValidation': ANNAccValidationAfterTune_improvement2},
                                                                                   ignore_index=True)



# -----------------------------------------------------------------------------
# Model Final Predictions
# -----------------------------------------------------------------------------
#        Loading the Test Dataset and preforming same manipluation over it :
XYtest =pd.read_csv('C:/Users/kolsk/PycharmProjects/MachineLearningPartB/X_test.csv')

#####    Making the Features :
#TitleCharCount:
def charCount(origin, res):
    for i in XYtest.index:
        XYtest.loc[i, res] = len(XYtest.loc[i, origin])


XYtest['title_char_count'] = 0
charCount('title', 'title_char_count')

#AirTime:
XYtest["publishedAt"] = XYtest["publishedAt"].astype("datetime64")
XYtest["trending_date"] = XYtest["trending_date"].astype("datetime64")
XYtest['Airtime']= XYtest["trending_date"] - XYtest["publishedAt"]
XYtest['Airtime'] = XYtest['Airtime'].astype('timedelta64[m]')

#CategoryRatio:
def categoryCount(categoryId):
    categoryCount = (XYtest['categoryId'] == categoryId).sum()
    return categoryCount


XYtest['category_ratio'] = 0
numOfSamples = len(XYtest)


for i in XYtest.index:
    category_Count = categoryCount(XYtest.loc[i, 'categoryId'])
    category_ratio = category_Count / numOfSamples
    XYtest.loc[i, 'category_ratio'] = category_ratio


quan34 = np.quantile(XYtest['category_ratio'], 0.34)
quan66 = np.quantile(XYtest['category_ratio'], 0.66)


for i in XYtest.index:
    if XYtest.loc[i, 'category_ratio'] < quan34:
        XYtest.loc[i, 'category_ratio'] = 'Low'
    elif XYtest.loc[i, 'category_ratio'] > quan66:
        XYtest.loc[i, 'category_ratio'] = 'Medium'
    else:
        XYtest.loc[i, 'category_ratio'] = 'High'

#Selecting the features we need:
XYtest = XYtest[["comment_count", "dislikes", "likes", "title_char_count",'Airtime','category_ratio','categoryId']]

# complete zeros values in numeric features
commentMedian = XYtest['comment_count'].median()
likesMedian = XYtest['likes'].median()
dislikesMedian = XYtest['dislikes'].median()

pd.options.mode.chained_assignment = None
for i in XYtest.index:
    if XYtest.loc[i, 'comment_count'] == 0 | pd.isnull(XYtest.loc[i, 'comment_count']) :
        XYtest.loc[i, 'comment_count'] = commentMedian
    if XYtest.loc[i, 'likes'] == 0 | pd.isnull(XYtest.loc[i, 'likes']) :
        XYtest.loc[i, 'likes'] = likesMedian
    if XYtest.loc[i, 'dislikes'] == 0 | pd.isnull(XYtest.loc[i, 'dislikes']):
        XYtest.loc[i, 'dislikes'] = dislikesMedian

XYtest['comment_count'] = np.log(XYtest['comment_count'])
XYtest['likes'] = np.log(XYtest['likes'])
XYtest['dislikes'] = np.log(XYtest['dislikes'])



#Dividing the data set to y vector and x matrix with dummies
X_Test = XYtest[['comment_count', 'dislikes', 'likes', 'title_char_count', 'Airtime', 'category_ratio','categoryId']]
X_Test = pd.get_dummies(X_Test, columns=['category_ratio'])
X_Test = pd.get_dummies(X_Test, columns=['categoryId'])

#Dealing with missing category variable on the test set:
missing_cols = set(X_Train_ImprovementNormalize.columns ) - set( X_Test.columns )
for c in missing_cols:
    X_Test[c] = 0

X_Test = X_Test[X_Train_ImprovementNormalize.columns]


#Normalizing the X matrix
scaler = StandardScaler()
X_Test[['comment_count', 'dislikes', 'likes', 'title_char_count', 'Airtime']] = scaler.fit_transform(X_Test[['comment_count', 'dislikes', 'likes', 'title_char_count', 'Airtime']])
X_Test = pd.DataFrame(X_Test)


# Running The Model
Finalres_predictions = pd.DataFrame()
final_model = MLPClassifier(random_state=1,
                                  hidden_layer_sizes=(51),
                                  max_iter=400,
                                  learning_rate_init=0.01,
                                  tol=0.00001,
                                  n_iter_no_change=20,
                                  verbose=True
                                  )

final_model.fit(X_Train_ImprovementNormalize, Y_Train_ImprovementNormalize)
FinalPredictions = pd.DataFrame(final_model.predict((X_Test)))
FinalPredictions = FinalPredictions.rename(columns={0: 'y'})
FinalPredictions.to_csv('FinalPredictions.csv', index=False)



##GRID SERCH -OFRI

from sklearn.model_selection import GridSearchCV
X_Train = data.drop(columns=['target'])
Y_Train = data['target']
##DT
decision_tree = DecisionTreeClassifier(random_state=42)

parameter_options_DT = {"criterion": ['gini', 'entropy'],
                        "max_depth": range(1, 10),
                        "max_features": ['auto', 'sqrt', 'log2', None]}

grid_DT = GridSearchCV(decision_tree, param_grid=parameter_options_DT, cv=10,
                       return_train_score=True, refit=True)
grid_DT.fit(X_Train, Y_Train)

best_params_DT = grid_DT.best_params_
all_score_DT = pd.DataFrame({'param': grid_DT.cv_results_["params"],
                             'Val_Accuracy': grid_DT.cv_results_["mean_test_score"],
                             'Train_Accuracy': grid_DT.cv_results_["mean_train_score"]})
best_score_DT = all_score_DT[all_score_DT['param'] == best_params_DT]
best_DT_model = grid_DT.best_estimator_

##ANNN
from sklearn.model_selection import GridSearchCV
X_Train = data.drop(columns=['target'])
Y_Train = data['target']
ANN = MLPClassifier()

parameter_options_ANN = {"max_iter": range(100, 800, 100),
                         "hidden_layer_sizes": (100, 50, 10),
                         "activation": ['identity', 'logistic', 'tanh', 'relu'],
                         "solver":['sgd', 'adam', 'lbfgs'],
                         "n_iter_no_change":[5, 10, 15, 20]}

grid_ANN = GridSearchCV(ANN, param_grid=parameter_options_ANN, cv=10,
                        return_train_score=True, refit=True, scoring='roc_auc')
grid_ANN.fit(X_Train, Y_Train)

best_params_ANN = grid_ANN.best_params_
all_score_ANN = pd.DataFrame({'param': grid_DT.cv_results_["params"],
                              'Val_Accuracy': grid_ANN.cv_results_["mean_test_score"],
                              'Train_Accuracy': grid_ANN.cv_results_["mean_train_score"]})
best_score_ANN = all_score_ANN[all_score_DT['param'] == best_params_ANN]
best_ANN_model = grid_ANN.best_estimator_
print(best_ANN_model)
## SVM
SVM_model = SVC(kernel='linear', random_state=42)
param_grid = {'C': np.arange(1, 2, 0.1),
              'decision_function_shape': ['ovo', 'ovr']
              }

grid_search_SVM = GridSearchCV(estimator=SVM_model,
                               param_grid=param_grid,
                               refit=True,
                               cv=10,
                               return_train_score=True)

grid_search_SVM.fit(X_Train_Normalize, Y_Train_Normalize)

best_params_SVM = grid_search_SVM.best_params_
all_score_SVM = pd.DataFrame({'param': grid_search_SVM.cv_results_["params"],
                              'Val_Accuracy': grid_search_SVM.cv_results_["mean_test_score"],
                              'Train_Accuracy': grid_search_SVM.cv_results_["mean_train_score"]})
best_score_SVM = all_score_SVM[all_score_SVM['param'] == best_params_SVM]

best_model_SVM = grid_search_SVM.best_estimator_

print('Coefficients: \n', best_model_SVM.coef_)
print('Intercepts: \n', best_model_SVM.intercept_)

## PRINT
print(best_DT_model)
print(best_ANN_model)
print(best_model_SVM)