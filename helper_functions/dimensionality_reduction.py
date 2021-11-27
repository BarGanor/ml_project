from sklearn.decomposition import PCA
import pandas as pd

def get_pca_results(df):
    x = df.drop(columns=['target', 'enrollment'])

    pca = PCA(.95)
    pca.fit(x)
    pca_results = pca.transform(x)
    pca_results = pd.DataFrame(pca_results, columns=['PC-' + str(i) for i in range(1,pca_results.shape[1]+1)], index=df.index)
    print('Explained variance ratio of PCA: ' + str(pca.explained_variance_ratio_.sum()))
    return pd.concat([df.drop(columns=x.columns),pca_results], axis=1)