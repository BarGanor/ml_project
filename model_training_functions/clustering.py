from model_training_functions.kfold_cross_validation import *
from sklearn_extra.cluster import KMedoids
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, roc_auc_score
from yellowbrick.cluster import KElbowVisualizer

def best_cluster(data, do_plot=False):
    x_train = data.drop(columns=['target'])
    scores = pd.DataFrame()
    for n_clusters in range(2, 10, 1):
        k_medoids = KMedoids(n_clusters=n_clusters, random_state=42)
        k_medoids.fit(x_train)
        pred = k_medoids.predict(x_train)
        inertia = k_medoids.inertia_
        sil_score = silhouette_score(x_train, pred)
        bouldin_score = davies_bouldin_score(x_train, pred)

        scores = scores.append({'Inertia': inertia, 'Sillhouette Score': sil_score, 'Bouldin Score': bouldin_score, 'n_clusters':n_clusters}, ignore_index=True)

    if do_plot:
        plot_inertia(scores)
        plot_sillhouette_bouldin(scores)

    return scores


def plot_inertia(all_scores):

    plt.figure(figsize=(12, 5))
    ax1 = sns.lineplot(x=all_scores['n_clusters'], y=all_scores['Inertia'])
    ax1.set(xlabel='Number of Clusters', ylabel='Inertia')
    plt.title('Elbow Method')
    plt.show()

def plot_sillhouette_bouldin(all_scores):
    plt.figure(figsize=(12, 5))
    plt.title('Sillhouette and Bouldin Scores')
    ax1 = sns.lineplot(y=all_scores['Sillhouette Score'], x=all_scores['n_clusters'], label='Sillhouette Score')
    ax2 = sns.lineplot(y=all_scores['Bouldin Score'], x=all_scores['n_clusters'], label='Bouldin Score')
    ax1.set(xlabel='Number of Clusters', ylabel='Score')
    plt.show()