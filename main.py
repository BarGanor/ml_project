import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = '/Users/barganor/Downloads/XY_train (1).csv'

data = pd.read_csv(path)

# Remove row if more than two nan in row
data = data.dropna(thresh=12, axis=0)


def replace_by_dict(df, col_name):
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
    else:
        replacement_dict = {}

    print(replacement_dict)
    df[col_name] = df[col_name].replace(replacement_dict)
    return df

def fill_nan_with_probability(df, col_name):
    filled_df = df.copy()
    no_nan_data = filled_df.dropna(subset=[col_name])
    all_nan_data = filled_df[col_name].isna()
    prob = no_nan_data[col_name].value_counts(normalize=True)
    filled_df.loc[all_nan_data, col_name] = np.random.choice(prob.index, size=len(data[all_nan_data]), p=prob.values)

    f, axes = plt.subplots(1, 2)
    sns.boxplot(x='target', y=col_name, data=no_nan_data, ax=axes[0]).set_title('Before Inserting')
    sns.boxplot(x='target', y=col_name, data=filled_df, ax=axes[1]).set_title('After Inserting')
    plt.show()

    return filled_df

def fill_nan_with_max_appear(df, col_name):
    filled_df = df.copy()
    max_appear = df['education_level'].value_counts().idxmax()

    filled_df[col_name].fillna(max_appear)
    return filled_df
processed_data = replace_by_dict(data, 'company_size')

fill_nan_with_probability(processed_data, 'company_size')