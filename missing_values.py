import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def drop_nan_by_thresh(df, thresh):
    dropped_df = df.copy()

    dropped_df = dropped_df.dropna(thresh=thresh, axis=0)

    rows_dropped = df.shape[0] - dropped_df.shape[0]
    nan_values_dropped = sum((df.isna().sum() - dropped_df.isna().sum()).tolist())
    print('Number Of rows dropped: ' + str(rows_dropped))
    print('Number Of nan values dropped: ' + str(nan_values_dropped))

    return dropped_df


def get_replacement_df():
    company_size = {
        '50-99': 3,
        '100-500': 4,
        '10000+': 8,
        'Oct-49': 2,
        '1000-4999': 6,
        '<10': 1,
        '500-999': 5,
        '5000-9999': 7
    }

    last_new_job = {
        '1': 1,
        '>4': 5,
        '2': 2,
        'never': 0,
        '4': 4,
        '3': 3
    }

    major_discipline = {
        'STEM': 0,
        'Humanities': 1,
        'Other': 2,
        'Business Degree': 3,
        'Arts': 4,
        'No Major': 5
    }
    experience = {'>20': '25', '<1': '0'}

    df = pd.DataFrame({'company_size before' : company_size.keys(),
                       'company_size after' : company_size.values(),
                       'last_new_job before' : last_new_job.keys(),
                       'last_new_job after' : last_new_job.values(),
                       'major_discipline before' : major_discipline.keys(),
                       'major_discipline after' : major_discipline.values(),
                       'experience before': experience.keys(),
                       'experience after': experience.values()
                       })

    return df

def replace_by_dict(df, col_name):
    replaced_df = df.copy()
    if col_name == 'company_size':
        replacement_dict = {
            '50-99': 3,
            '100-500': 4,
            '10000+': 8,
            'Oct-49': 2,
            '1000-4999': 6,
            '<10': 1,
            '500-999': 5,
            '5000-9999': 7
        }
    elif col_name == 'last_new_job':
        replacement_dict = {
            '1': 1,
            '>4': 5,
            '2': 2,
            'never': 0,
            '4': 4,
            '3': 3
        }

    elif col_name == 'major_discipline':
        replacement_dict = {
            'STEM': 0,
            'Humanities': 1,
            'Other': 2,
            'Business Degree': 3,
            'Arts': 4,
            'No Major': 5
        }

    elif col_name == 'experience':
        replacement_dict = {'>20': '25', '<1': '0'}
    else:
        replacement_dict = {}

    replaced_df[col_name] = replaced_df[col_name].replace(replacement_dict)

    values_before = df[col_name].value_counts().index
    values_after = replaced_df[col_name].value_counts().index
    changes_df = pd.DataFrame({'Values before change': values_before, 'Values after change': values_after})
    print(changes_df.sort_values(by='Values after change').to_string(index=False))

    return replaced_df


def fill_nan_with_probability(df, col_name):
    filled_df = df.copy()
    no_nan_data = filled_df.dropna(subset=[col_name])
    all_nan_data = filled_df[col_name].isna()
    prob = no_nan_data[col_name].value_counts(normalize=True)
    filled_df.loc[all_nan_data, col_name] = np.random.choice(prob.index, size=len(df[all_nan_data]), p=prob.values)

    f, axes = plt.subplots(1, 2)
    sns.boxplot(x='target', y=col_name, data=no_nan_data, ax=axes[0]).set_title('Before Inserting')
    sns.boxplot(x='target', y=col_name, data=filled_df, ax=axes[1]).set_title('After Inserting')
    plt.show()

    return filled_df


def fill_nan_with_max_appear(df, col_name):
    filled_df = df.copy()
    max_appear = df['education_level'].value_counts().idxmax()

    filled_df[col_name] = filled_df[col_name].fillna(max_appear)
    return filled_df


def fill_nan_with_median(df, col_name):
    filled_df = df.copy()
    filled_df[col_name] = filled_df[col_name].astype('float64')
    col_median = np.median(filled_df[col_name].dropna().values)
    filled_df[col_name] = filled_df[col_name].fillna(col_median)
    return filled_df


