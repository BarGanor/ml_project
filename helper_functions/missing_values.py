import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import display
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def drop_nan_by_thresh(df, thresh):
    dropped_df = df.copy()
    dropped_df = dropped_df.dropna(thresh=thresh, axis=0)

    difference_df = pd.concat([df.isna().sum(), dropped_df.isna().sum()], axis=1)
    difference_df.columns = ['Before', 'After']
    difference_df.loc['Sum Of Missing Vals'] = difference_df.sum(axis=0)
    difference_df.loc['Numbers in Rows'] = [df.shape[0], dropped_df.shape[0]]
    difference_df['Difference'] = difference_df['Before'] - difference_df['After']
    display(difference_df)
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

    df = pd.DataFrame({'company_size before': company_size.keys(),
                       'company_size after': company_size.values(),
                       'last_new_job before': last_new_job.keys(),
                       'last_new_job after': last_new_job.values(),
                       'major_discipline before': major_discipline.keys(),
                       'major_discipline after': major_discipline.values(),
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

    elif col_name == 'company_type':
        replacement_dict = {
            'Pvt Ltd': 1,
            'Unknown': 2,
            'Funded Startup': 3,
            'Public Sector': 4,
            'Early Stage Startup': 5,
            'NGO': 6,
            'Other': 7,
        }
    elif col_name == 'enrolled_university':
        replacement_dict = {
            'no_enrollment': 0,
            'Full time course': 1,
            'Part time course': -1
        }

    elif col_name == 'gender':
        replacement_dict = {
            'Male': 1,
            'Female': 0,
            'Other': -1
        }

    elif col_name == 'relevent_experience':
        replacement_dict = {
            'Has relevent experience': 1,
            'No relevent experience': 0
        }

    elif col_name == 'experience':
        replacement_dict = {'>20': '25', '<1': '0'}

    elif col_name == 'education_level':
        replacement_dict = {'Graduate': 15, 'Masters': 17, 'High School': 12, 'Phd': 20, 'Primary School': 6}
    else:
        replacement_dict = {}

    replaced_df[col_name] = replaced_df[col_name].replace(replacement_dict)

    values_before = df[col_name].value_counts().index
    values_after = replaced_df[col_name].value_counts().index
    changes_df = pd.DataFrame({'Values before change': values_before, 'Values after change': values_after})
    print(changes_df.sort_values(by='Values after change').to_string(index=False))

    return replaced_df


def fill_nan_with_mice(df, col_name):
    imputer = IterativeImputer()
    filled_df = df.copy()

    filled_df[col_name] = imputer.fit_transform(filled_df[[col_name]])
    filled_df[col_name] = round(filled_df[[col_name]])

    if col_name == 'relevent_experience':
        filled_df[col_name] = filled_df[col_name].replace({0: 'No relevent experience', 1: 'Has relevent experience'})

    if col_name == 'enrolled_university':
        filled_df[col_name] = filled_df[col_name].replace({0: 'no_enrollment', 1: 'Full time course', -1: 'Part time course'})

    if col_name == 'gender':
        filled_df[col_name] = filled_df[col_name].replace({0: 'Female', 1: 'Male', -1: 'Other'})
    return filled_df


def fill_nan_with_probability(df, col_name):
    filled_df = df.copy()
    no_nan_data = filled_df.dropna(subset=[col_name])
    all_nan_data = filled_df[col_name].isna()
    prob = no_nan_data[col_name].value_counts(normalize=True)
    filled_df.loc[all_nan_data, col_name] = np.random.choice(prob.index, size=len(df[all_nan_data]), p=prob.values)

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(col_name)
    plt.tight_layout(pad=4.0)
    sns.boxplot(x='target', y=col_name, data=no_nan_data, ax=axes[0]).set_title('Before Inserting')
    sns.boxplot(x='target', y=col_name, data=filled_df, ax=axes[1]).set_title('After Inserting')
    plt.show()

    return filled_df


def fill_nan_with_max_appear(df, col_name):
    filled_df = df.copy()
    max_appear = df['education_level'].value_counts().idxmax()

    filled_df[col_name] = filled_df[col_name].fillna(max_appear)

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(col_name)
    plt.xticks(rotation=90)
    sns.barplot(x='index', y=col_name, data=df[col_name].value_counts().reset_index(), ax=axes[0]).set_title('Before')
    sns.barplot(x='index', y=col_name, data=filled_df[col_name].value_counts().reset_index(), ax=axes[1]).set_title('After')

    plt.tight_layout()
    axes[0].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='x', rotation=90)
    plt.show()

    return filled_df


def fill_nan_with_median(df, col_name):
    # fill the column
    filled_df = df.copy()
    filled_df[col_name] = filled_df[col_name].astype('float64')
    col_median = np.median(filled_df[col_name].dropna().values)
    filled_df[col_name] = filled_df[col_name].fillna(col_median)

    # Create Plot
    value_counts_filled = filled_df[col_name].value_counts()
    value_counts_orig = df[col_name].astype('float64').value_counts()

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(col_name)
    sns.kdeplot(x=value_counts_filled.index, y=value_counts_filled.values, ax=axes[0]).set_title('Before')
    sns.kdeplot(x=value_counts_orig.index, y=value_counts_orig.values, ax=axes[1]).set_title('After')
    plt.show()
    return filled_df


def plot_before_and_after(df_before, df_after):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('gender')
    sns.violinplot(x="gender", y="target", data=df_before, split=True, ax=axes[0]).set_title('Before')
    sns.violinplot(x="gender", y="target", data=df_after, split=True, ax=axes[1]).set_title('After')
    plt.tight_layout()
    plt.show()


def fit_the_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = KNeighborsClassifier(n_neighbors=8)
    model.fit(x_train, y_train)
    print('Model Score on Self: ' + str(model.score(x_test, y_test)))

    return model


def prepare_df_for_model(df, for_train=True):
    scaler = MinMaxScaler()
    if for_train:
        df = df.dropna()

    cat_vars = df[['relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'last_new_job', 'target', 'company_size']].copy()

    cat_dummies = pd.get_dummies(cat_vars, drop_first=True)
    prepared_df = df.copy()
    prepared_df = prepared_df.drop(columns=cat_vars.columns)
    gender = prepared_df['gender']
    prepared_df = prepared_df.drop(columns=['city', 'enrollee_id', 'company_type', 'city_development_index', 'gender'])
    prepared_df = pd.DataFrame(scaler.fit_transform(prepared_df), columns=prepared_df.columns, index=prepared_df.index)
    prepared_df = pd.concat([prepared_df, cat_dummies, gender], axis=1)
    return prepared_df


def get_fit_model(df):
    prepared_df = prepare_df_for_model(df.copy())
    x = prepared_df.drop(columns=['gender'])
    y = prepared_df['gender']
    model = fit_the_model(x, y)
    return model


def fill_gender_with_knn(df):
    filled_df = df.copy()
    all_are_nan = prepare_df_for_model(filled_df[pd.isna(filled_df['gender'])], for_train=False)
    non_are_nan = filled_df[pd.notna(filled_df['gender'])]

    model = get_fit_model(non_are_nan)
    all_are_nan['gender'] = model.predict(all_are_nan.drop(columns=['gender']))

    filled_df.loc[pd.isna(filled_df['gender']), 'gender'] = all_are_nan['gender']
    plot_before_and_after(df_before=df, df_after=filled_df)
    return filled_df


def fill_nan_with_unknown(df, col):
    df = df.copy()
    df[col] = df[col].fillna('Unknown')
    value_counts = df[col].value_counts()
    plt.xticks(rotation=90)
    sns.barplot(x=value_counts.index, y=value_counts.values)
    return df
