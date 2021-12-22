from sklearn.decomposition import PCA
import pandas as pd
def get_pca_results(df):
    x = df.drop(columns=['target', 'enrollment'])

    pca = PCA(.95)
    pca.fit(x)
    pca_results = pca.transform(x)
    pca_results = pd.DataFrame(pca_results, columns=['PC-' + str(i) for i in range(1,pca_results.shape[1]+1)], index=df.index)

    print('Explained variance ratio of PCA: ' + str(pca.explained_variance_ratio_.sum()))
    plot_dict = []
    for i in range(len(pca.explained_variance_ratio_)):
        plot_dict.append({'PC-'+str(i+1) : pca.explained_variance_ratio_[i]})

    pd.DataFrame.from_records(plot_dict).plot(kind='bar', title='PCA Explained Ratio')
    return pd.concat([df.drop(columns=x.columns),pca_results], axis=1)