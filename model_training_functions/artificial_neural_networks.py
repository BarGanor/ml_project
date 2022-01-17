from model_training_functions.kfold_cross_validation import train_model_by_kfold
import itertools
from sklearn.neural_network import MLPClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def max_iterations(data, do_plot=False):
    iterations = range(100, 800, 100)
    all_scores = pd.DataFrame()
    for iteration in iterations:
        model = MLPClassifier(max_iter=iteration, verbose=False)
        scores = train_model_by_kfold(data, model)
        scores['max_iter'] = iteration
        all_scores = all_scores.append(scores, ignore_index=True)

    if do_plot:
        plot_max_iterations(all_scores)
    return all_scores


def hidden_layer_sizes(data, do_plot=False):
    all_scores = pd.DataFrame()
    sizes = [10, 50, 100]
    for L in range(0, len(sizes) + 1):
        for subset in itertools.permutations(sizes, L):
            model = MLPClassifier(max_iter=600, verbose=False, hidden_layer_sizes=subset)
            scores = train_model_by_kfold(data, model)
            scores['hidden_layer_size'] = subset
            all_scores = all_scores.append(scores, ignore_index=True)

    if do_plot:
        plot_hidden_layer_sizes(all_scores)
    return all_scores


def activations_function(data, do_plot=False):
    activation_functions = ['identity', 'logistic', 'tanh', 'relu']
    all_scores = pd.DataFrame()
    for activation in activation_functions:
        model = MLPClassifier(max_iter=600, hidden_layer_sizes=(100, 50, 10), activation=activation)
        scores = train_model_by_kfold(data, model)
        scores['activation'] = activation
        all_scores = all_scores.append(scores, ignore_index=True)

    if do_plot:
        plot_activation_functions(all_scores)
    return all_scores

def solvers(data, do_plot):
    solvers = ['sgd', 'adam', 'lbfgs']
    all_scores = pd.DataFrame()
    for solver in solvers:
        model = MLPClassifier(max_iter=600,hidden_layer_sizes=(100, 50, 10), activation='relu',solver=solver)
        scores = train_model_by_kfold(data, model)
        scores['solver'] = solver
        all_scores = all_scores.append(scores, ignore_index=True)
    
    if do_plot:
        plot_solvers(all_scores)

    return all_scores


def n_iter_no_change(data, do_plot):
    iter_no_change = [5, 10, 15, 20]
    all_scores = pd.DataFrame()
    for n_iter in iter_no_change:
        model = MLPClassifier(max_iter=600, hidden_layer_sizes=(100, 50, 10), activation='relu', n_iter_no_change=n_iter)
        scores = train_model_by_kfold(data, model)
        scores['n_iter_no_change'] = n_iter
        all_scores = all_scores.append(scores, ignore_index=True)

    if do_plot:
        plot_n_iter_no_change(all_scores)

    return all_scores


def plot_n_iter_no_change(all_scores):
    plt.figure(figsize=(12, 5))
    ax1 = sns.lineplot(x=all_scores['n_iter_no_change'], y=all_scores['test_score'], label='Validation')
    ax2 = sns.lineplot(x=all_scores['n_iter_no_change'], y=all_scores['training_score'], label='Training')
    ax1.set(xlabel='N Iterations No Change', ylabel='ROC-AOC Score')
    plt.legend()
    ax1.set_xticks(all_scores['n_iter_no_change'].values.astype('int64'))
    plt.title('Max Iterations Tuning')
    plt.show()


def plot_activation_functions(all_scores):
    plt.figure(figsize=(12, 5))
    temp = pd.melt(all_scores, id_vars='activation')
    ax1 = sns.catplot(x='activation', y='value',
                      hue='variable', data=temp, kind='bar', legend=False)
    ax1.set(xlabel='Activation Function', ylabel='ROC-AOC Score')
    plt.title('Activation Function Tuning')
    plt.ylim(temp['value'].min() - 0.01, temp['value'].max() + 0.01)
    plt.legend()
    plt.show()


def plot_max_iterations(all_scores):
    plt.figure(figsize=(12, 5))
    ax1 = sns.lineplot(x=all_scores['max_iter'], y=all_scores['test_score'], label='Validation')
    ax2 = sns.lineplot(x=all_scores['max_iter'], y=all_scores['training_score'], label='Training')
    ax1.set(xlabel='Max Iterations', ylabel='ROC-AOC Score')
    ax2.set(xlabel='Max Iterations', ylabel='ROC-AOC Score')
    plt.legend()
    plt.title('Max Iterations Tuning')
    plt.show()

def plot_solvers(all_scores):
    plt.figure(figsize=(12, 5))
    temp = pd.melt(all_scores, id_vars='solver')
    ax1 = sns.catplot(x='solver', y='value',
                      hue='variable', data=temp, kind='bar', legend=False)
    ax1.set(xlabel='solver Function', ylabel='ROC-AOC Score')
    plt.title('solver Function Tuning')
    plt.ylim(temp['value'].min() - 0.01, temp['value'].max() + 0.01)
    plt.legend()
    plt.show()
    
def plot_hidden_layer_sizes(all_scores):
    temp = pd.melt(all_scores, id_vars='hidden_layer_size')
    ax1 = sns.catplot(x='hidden_layer_size', y='value',
                      hue='variable', data=temp, kind='bar')
    ax1.set(xlabel='Hidden Layer Sizes', ylabel='ROC-AOC Score')
    ax1.fig.set_size_inches(20, 8)
    plt.ylim(temp['value'].min() - 0.01, temp['value'].max() + 0.01)
    plt.title('Hidden Layer Sizes Tuning')
    plt.show()
