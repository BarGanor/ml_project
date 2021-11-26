import pandas as pd
import statsmodels.api as sm
from missing_values import *



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
    return logit_results


def get_relevant_exp_score(df):
    if (df['relevent_experience'] == 'Has relevent experience'):
        return logit_results['Has relevent experience']
    else:
        return logit_results['No relevent experience']


def get_education_level_score(df):
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


def get_experience_score(df):
    return df['experience'] * logit_results['experience']


def get_enrolled_score(df):
    if (df['enrolled_university'] == 'Full time course'):
        return logit_results['Full time course']

    elif (df['enrolled_university'] == 'Part time course'):
        return logit_results['Part time course']

    else:
        return logit_results['no_enrollment']


def get_qualification_index(df):
    df['qualification_score'] = df.apply(get_relevant_exp_score, axis=1)
    df['qualification_score'] += df.apply(get_education_level_score, axis=1)
    df['qualification_score'] += df.apply(get_experience_score, axis=1)
    df['qualification_score'] += df.apply(get_enrolled_score, axis=1)
    return df

data = pd.read_csv('/Users/barganor/Downloads/XY_train (1).csv')

processed_data = drop_nan_by_thresh(data, 12)

#%%

processed_data = replace_by_dict(processed_data, 'company_size')

#%%

processed_data = replace_by_dict(processed_data, 'last_new_job')

#%%

processed_data = replace_by_dict(processed_data, 'major_discipline')

#%%

processed_data = replace_by_dict(processed_data, 'experience')

#%%

processed_data = fill_nan_with_median(processed_data, 'experience')

#%%

processed_data = fill_nan_with_probability(processed_data, 'company_size')

#%%

processed_data = fill_nan_with_probability(processed_data, 'last_new_job')

#%%

processed_data = fill_nan_with_probability(processed_data, 'major_discipline')


processed_data =  fill_nan_with_max_appear(processed_data, 'education_level')



processed_data = fill_nan_with_probability(processed_data, 'enrolled_university')


processed_data = fill_nan_with_knn(processed_data)

logit_results = get_coef_vals(processed_data)['Coef']
print(logit_results)
print(get_qualification_index(processed_data))
