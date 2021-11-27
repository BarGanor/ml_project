import pandas as pd
import statsmodels.api as sm
from missing_values import *
from IPython.display import display


def get_coef_vals(df):
    col_vars = ['relevent_experience', 'enrolled_university', 'education_level', 'experience']
    logit_results = pd.DataFrame()
    for col in col_vars:
        x_values = df[col]
        if col == 'experience':
            logit_model = sm.Logit(df['target'], x_values)
        else:
            logit_model = sm.Logit(df['target'], pd.get_dummies(x_values))

        result = logit_model.fit(disp=0)
        col_params = pd.concat([result.params, result.pvalues], axis=1)
        col_params.columns = ['Coef', 'P-value']
        logit_results = pd.concat([logit_results, col_params], axis=0)
    display(logit_results)
    return logit_results


def get_relevant_exp_score(df, logit_results):
    if (df['relevent_experience'] == 'Has relevent experience'):
        return logit_results['Has relevent experience']
    else:
        return logit_results['No relevent experience']


def get_education_level_score(df, logit_results):
    if (df['education_level'] == 'Phd'):
        return logit_results['Phd']

    elif (df['education_level'] == 'Masters'):
        return logit_results['Masters']


    elif (df['education_level'] == 'Graduate'):
        return logit_results['Graduate']

    elif (df['education_level'] == 'High School'):
        return logit_results['High School']

    else:
        return logit_results['Primary School']


def get_experience_score(df, logit_results):
    return df['experience'] * logit_results['experience']


def get_enrolled_score(df, logit_results):
    if (df['enrolled_university'] == 'Full time course'):
        return logit_results['Full time course']

    elif (df['enrolled_university'] == 'Part time course'):
        return logit_results['Part time course']

    else:
        return logit_results['no_enrollment']


def get_qualification_index(df):
    logit_results = get_coef_vals(df)['Coef']
    df['qualification_score'] = df.apply(get_relevant_exp_score, axis=1, args=(logit_results,))
    df['qualification_score'] += df.apply(get_education_level_score, axis=1, args=(logit_results,))
    df['qualification_score'] += df.apply(get_experience_score, axis=1, args=(logit_results,))
    df['qualification_score'] += df.apply(get_enrolled_score, axis=1, args=(logit_results,))
    return df


def get_relevant_experience_years(df):
    if df["relevent_experience"] == 'No relevent experience':
        return 0
    else:
        return df["experience"]

def get_relevant_experience_feature(df):
    df['relevant_experience_years'] = df.apply(get_relevant_experience_years, axis=1)
    sns.kdeplot(df["relevant_experience_years"][df['target'] == 1], label='target=1', color='red')
    sns.kdeplot(df["relevant_experience_years"][df['target'] == 0], label='target=0')
    plt.legend()

    return df

