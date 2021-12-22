from sklearn.feature_selection import chi2
from scipy import stats
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

def get_categorical_vars_corr(df, col_list):
    chi2_result_df = pd.DataFrame()

    for col in col_list:
        chi2_val, p_val = chi2(np.array(df[col]).reshape(-1, 1), np.array(df['target']).reshape(-1, 1))
        col_result = pd.DataFrame({'statistic_score': chi2_val, 'p_value': p_val}, index=[col])

        chi2_result_df = pd.concat([chi2_result_df, col_result], axis=0)

    index_df = pd.DataFrame({'test type': ['chi_test'] * len(col_list), 'var name': chi2_result_df.index})
    index = pd.MultiIndex.from_frame(index_df)
    chi2_result_df.index = index
    return chi2_result_df


def get_numeric_vars_corr(df, col_list):
    pearson_result_df = pd.DataFrame()
    for col in col_list:
        pearson_val, p_val = stats.pearsonr(df[col], df['target'])
        col_result = pd.DataFrame({'statistic_score': pearson_val, 'p_value': p_val}, index=[col])
        pearson_result_df = pd.concat([pearson_result_df, col_result], axis=0)

    index_df = pd.DataFrame({'test type': ['pearson test'] * len(col_list), 'var name': pearson_result_df.index})
    index = pd.MultiIndex.from_frame(index_df)
    pearson_result_df.index = index
    return pearson_result_df


def get_correlation_df(df):

    categorical = ['Has relevent experience', 'Male', 'Female', 'company_type', 'Full time course', 'enrollment']
    numeric = ['training_hours', 'experience', 'company_size', 'education_level', 'last_new_job', 'qualification_score', 'relevant_experience_years']

    categorical_corr = get_categorical_vars_corr(df, categorical)
    numeric_corr = get_numeric_vars_corr(df, numeric)
    correlation_df = pd.concat([categorical_corr, numeric_corr], axis=0)
    display(correlation_df)
    return correlation_df


def drop_unimportant_vars(df, correlation_df):
    unimportant_vars = ['city', 'enrollee_id']
    high_p_value = correlation_df[correlation_df['p_value'] > 0.05].index.get_level_values(1)
    unimportant_vars.extend(high_p_value.tolist())

    print('Variables dropped (p-value < 0.05) : ' + ', '.join(unimportant_vars))
    return df.drop(columns=unimportant_vars)


def get_backtracking_results(df):
    model = LogisticRegression()
    rfe = RFE(model, n_features_to_select=1)
    x = df.drop(columns=['target'])
    y = df['target']

    rfe.fit(x, y)

    rfe_results = []
    for i in range(x.shape[1]):
        rfe_results.append(
            {
                'Feature_names': x.columns[i],
                'RFE_ranking': rfe.ranking_[i],
            }
        )
    rfe_results = pd.DataFrame(rfe_results)
    rfe_results = rfe_results.set_index('Feature_names', drop=True)
    display(rfe_results)
    return rfe_results


def drop_not_selected(df, chosen_cols):
    chosen_cols.append('target')
    return df[chosen_cols]


def get_slope(x,y):
    slope = (y[0] - y[-1])/ (x[0]-x[-1])
    return "%.3f" % slope


def get_qualitative_evaluation_charts(df):
    cols = df.drop(columns=['target']).columns
    fig, axes = plt.subplots(4, 3, figsize=(30, 20))
    axe = axes.ravel()
    for i in range(len(cols)):
        if df[cols[i]].dtypes == 'float64':
            p = sns.regplot(x='target', y=cols[i], data=df, ax=axe[i])
            x, y = p.get_lines()[0].get_data()
            slope = get_slope(x, y)
            p.set_title(cols[i] + '\n Slope of line: ' + str(slope))

        elif cols[i] != 'target':
            counts = (df.groupby(['target'])[cols[i]]
                      .value_counts(normalize=True)
                      .rename('percentage')
                      .mul(100)
                      .reset_index())
            sns.barplot(x=cols[i], y="percentage", hue="target", data=counts, ax=axe[i]).set_title(cols[i])


    fig.tight_layout(pad=3.0)