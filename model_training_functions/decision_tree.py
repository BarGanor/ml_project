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
        scores['max_depth'] = depth
        all_scores = all_scores.append(scores, ignore_index=True)

    if do_plot:
        plot_max_depth(all_scores)

        return all_scores


def criterion(data, do_plot=False):
    criterions = ['gini', 'entropy']
    all_scores = pd.DataFrame()

    for criterion in criterions:
        model = DecisionTreeClassifier(max_depth=5, criterion=criterion)
        scores = train_model_by_kfold(data, model)
        scores['criterion'] = criterion
        all_scores = all_scores.append(scores, ignore_index=True)

    if do_plot:
        plot_criterion(all_scores)

        return all_scores


def min_samples_leaf(data, do_plot=False):
    min_samples = range(1, 200, 10)
    all_scores = pd.DataFrame()

    for min_sample in min_samples:
        model = DecisionTreeClassifier(max_depth=5, criterion='entropy', min_samples_leaf=min_sample)
        scores = train_model_by_kfold(data, model)
        scores['min_samples_leaf'] = min_sample
        all_scores = all_scores.append(scores, ignore_index=True)

    if do_plot:
        plot_min_sample_leaf(all_scores)

        return all_scores


def min_impurity_decrease(data, do_plot=True):
    min_impurities = [i / 1000 for i in range(10)]
    all_scores = pd.DataFrame()
    for min_impurity in min_impurities:
        model = DecisionTreeClassifier(max_depth=5, criterion='entropy', min_samples_leaf=100, min_impurity_decrease=min_impurity)
        scores = train_model_by_kfold(data, model)
        scores['min_impurity_decrease'] = min_impurity
        all_scores = all_scores.append(scores, ignore_index=True)

    if do_plot:
        plot_min_impurity(all_scores)

        return all_scores
    if do_plot:
        plot_min_sample_leaf(all_scores)


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
