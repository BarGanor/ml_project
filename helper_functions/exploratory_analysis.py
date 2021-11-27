import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = '/Users/barganor/Downloads/XY_train (1).csv'

data = pd.read_csv(path)


def plot_cities(df, ax):
    cities_count = df['city'].value_counts()
    cities_count.hist(ax=ax).set_title('cities')


def plot_cities_development(df, ax):
    sns.histplot(df['city_development_index'], kde=True, ax=ax).set_title('city_development_index')


def plot_gender(df, ax):
    df['gender'].value_counts(dropna=False).plot.pie(ax=ax, title='gender')


def plot_relevant_experience(df, ax):
    relevant_experience = df['relevent_experience'].value_counts()
    relevant_experience.index = relevant_experience.index.map({'Has relevent experience	': 'Has', 'No relevent experience': 'Doesnt have'})
    relevant_experience.plot(kind='bar', title='relevant_experience', ax=ax)


def plot_education_level(df, ax):
    df['education_level'].value_counts().plot(kind='bar', ax=ax, title='education_level')


def plot_company_type(df, ax):
    df['company_type'].value_counts(dropna=False).plot(kind='bar', ax=ax, title='company_type')


def plot_major_discipline(df, ax):
    df['major_discipline'].value_counts(dropna=False).plot(kind='bar', ax=ax, title='major_discipline')


def plot_experience(df, ax):
    df['experience'].value_counts(dropna=False).plot(kind='bar', ax=ax).set_title('experience')


def plot_training_hours(df, ax):
    sns.kdeplot(df['training_hours'], ax=ax).set_title('training_hours')


def plot_company_size(df, ax):
    df['company_size'].value_counts(dropna=False).plot(kind='bar', ax=ax).set_title('company_size')


def plot_last_new_job(df, ax):
    sns.histplot(df['last_new_job'], ax=ax).set_title('last_new_job')


def plot_target(df, ax):
    df['target'].value_counts().plot.bar(ax=ax).set_title('target')


def get_all_plots(df):
    plot_lst = [
        plot_cities, plot_cities_development, plot_gender,
        plot_relevant_experience, plot_education_level,
        plot_company_type,
        plot_training_hours, plot_target,
        plot_experience, plot_major_discipline,
        plot_company_size, plot_last_new_job
    ]

    fig, axes = plt.subplots(3, ceil(len(plot_lst) / 3), figsize=(30, 20))
    axe = axes.ravel()

    for i in range(len(plot_lst)):
        plot_lst[i](df, ax=axe[i])
    fig.tight_layout(pad=3.0)


def get_statistical_description(df):
    statistical_description = df.describe().transpose()
    statistical_description['value_range'] = statistical_description['min'].astype(str) + ' - ' + statistical_description['max'].astype(str)
    statistical_description = statistical_description.drop(columns=['count', 'min', 'max', '25%', '75%'])
    return statistical_description


def get_job_seekers_by_gender(df):
    print('אחוז מחפשי העבודה לפי מגדר')
    return df[df['target'] == 0]['gender'].value_counts() / df['gender'].value_counts()

