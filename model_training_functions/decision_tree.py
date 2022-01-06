import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
from model_training_functions.kfold_cross_validation import train_model_by_kfold
import matplotlib.pyplot as plt
import seaborn as sns


def max_depth(data, do_plot=False):
    depths = range(1, 10)
    all_scores = pd.DataFrame()
    
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth)
        scores = train_model_by_kfold(data, model)
        all_scores = all_scores.append({'max_depth': depth,
                          'scores': scores}, ignore_index=True)
    if do_plot:
        plot_max_depth(all_scores)

    return all_scores


def criterion(data, do_plot=False):
    criterions = ['gini', 'entropy']
    all_scores = pd.DataFrame()

    for criterion in criterions:
        model = DecisionTreeClassifier(max_depth=3, criterion=criterion)
        scores = train_model_by_kfold(data, model)
        all_scores = all_scores.append({'criterion':  criterion,
                          'scores': scores}, ignore_index=True)

    if do_plot:
        plot_criterion(all_scores)

    return all_scores

def min_samples_leaf(data, do_plot=False):
    min_samples = range(1, 200, 10)
    all_scores = pd.DataFrame()

    for min_sample in min_samples:
        model = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_leaf=min_sample)
        scores = train_model_by_kfold(data, model)
        all_scores = all_scores.append({'min_samples_leaf':  min_sample,
                          'scores': scores}, ignore_index=True)

    if do_plot:
        plot_min_sample_leaf(all_scores)

    return all_scores

def min_impurity_decrease(data, do_plot=False):
    min_impurities = [i / 1000 for i in range(10)]
    all_scores = pd.DataFrame()
    for min_impurity in min_impurities:
        model = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_leaf=81, min_impurity_decrease=min_impurity)
        scores = train_model_by_kfold(data, model)
        all_scores = all_scores.append({'min_impurity_decrease': min_impurity,
                                        'scores': scores}, ignore_index=True)

    if do_plot:
        plot_min_impurity(all_scores)

    return all_scores



def plot_max_depth(all_scores):
    plt.figure(figsize=(12, 5))
    ax1 = sns.lineplot(x=all_scores['max_depth'], y=all_scores['Validation Score'], label='Validation')
    ax2 = sns.lineplot(x=all_scores['max_depth'], y=all_scores['Training Score'], label='Training')
    ax1.set(xlabel='Max Depth', ylabel='ROC-AOC Score')
    ax2.set(xlabel='Max Depth', ylabel='ROC-AOC Score')
    plt.legend()
    plt.title('Max Depth Tuning')
    plt.show()


def plot_criterion(all_scores):
    plt.figure(figsize=(12, 5))
    temp = pd.melt(all_scores, id_vars='criterion')
    ax1 = sns.catplot(x='criterion', y='value',
                      hue='variable', data=temp, kind='bar', legend=False)
    ax1.set(xlabel='Criterion', ylabel='ROC-AOC Score')
    plt.title('Criterion Tuning')
    plt.ylim(temp['value'].min() - 0.01, temp['value'].max() + 0.01)
    plt.legend()
    plt.show()


def plot_min_sample_leaf(all_scores):
    plt.figure(figsize=(12, 5))
    ax1 = sns.lineplot(x=all_scores['min_samples_leaf'], y=all_scores['Validation Score'], label='Validation')
    ax2 = sns.lineplot(x=all_scores['min_samples_leaf'], y=all_scores['Training Score'], label='Training')
    ax1.set(xlabel='Min Sample Leaf', ylabel='ROC-AOC Score')
    ax2.set(xlabel='Min Sample Leaf', ylabel='ROC-AOC Score')
    plt.legend()
    plt.title('Min Sample Leaf')
    plt.show()


def plot_min_impurity(all_scores):
    plt.figure(figsize=(12, 5))
    ax1 = sns.lineplot(x=all_scores['min_impurity_decrease'], y=all_scores['Validation Score'], label='Validation')
    ax2 = sns.lineplot(x=all_scores['min_impurity_decrease'], y=all_scores['Training Score'], label='Training')
    ax1.set(xlabel='Min Impurity Decrease', ylabel='ROC-AOC Score')
    ax2.set(xlabel='Min Impurity Decrease', ylabel='ROC-AOC Score')
    plt.legend()
    plt.title('Min Impurity Decrease')
    plt.show()
    
def plot_Decision_Tree(df):
    X = df.drop(columns=['target'])
    Y = df['target']
    model = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_leaf=81)
    model.fit(X,Y)
    plt.figure(figsize=(60, 40))
    plot_tree(model, filled=True, class_names=True,feature_names=X.columns[:])
    plt.title("Decision Tree Model")
    plt.show()

def feature_importance(df):
    X = df.drop(columns=['target'])
    Y = df['target']
    model = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_leaf=81)
    model.fit(X,Y)
    importance = model.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    plot_Feature_Importance(df, model, importance,X)
   
    
def plot_feature_importance(df, model, importance,X):
    plt.figure(figsize=(15, 7))
    pd.Series(model.feature_importances_).plot.bar(color='steelblue')
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xticks([0,1,2,3,4,5],X.columns[:],rotation=30)
    plt.tick_params(axis='both', which='minor', labelsize=4)
    plt.ylabel('Feature_importance', fontsize=15)
    plt.xlabel('Features', fontsize=15)
    plt.title('DT Feature Importance')
    plt.show()
   