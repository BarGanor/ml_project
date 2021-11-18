import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = '/Users/barganor/Downloads/XY_train (1).csv'

data = pd.read_csv(path)


# data.head()


def plot_cities(df, ax):
    cities_count = df['city'].value_counts()
    cities_count.hist(ax=ax).set_title('cities')


def plot_cities_development(df, ax):
    df.groupby('city')['city_development_index'].mean().plot.density(ax=ax, title='city_development_index')


def plot_gender(df, ax):
    df['gender'].value_counts(dropna=False).plot.pie(ax=ax, title='gender')


def plot_relevant_experience(df, ax):
    df['relevent_experience'].value_counts().plot(kind='bar', title='relevant_experience')


def plot_education_level(df, ax):
    df['education_level'].value_counts().plot(kind='bar', ax=ax, title='education_level')


def plot_company_type(df, ax):
    df['company_type'].value_counts(dropna=False).plot(kind='bar', ax=ax, title='company_type')


def plot_training_hours(df, ax):
    sns.kdeplot(df['training_hours'], ax=ax).set_title('training_hours')


def plot_target(df, ax):
    df['target'].value_counts().plot.bar(ax=ax).set_title('target')


def get_all_plots(df):
    plot_lst = [
        plot_cities, plot_cities_development, plot_gender,
        plot_relevant_experience, plot_education_level,
        plot_company_type,
        plot_training_hours, plot_target
    ]

    fig, axes = plt.subplots(2, int(len(plot_lst) / 2), figsize=(15, 15))

    axe = axes.ravel()

    for i in range(len(plot_lst)):
        plot_lst[i](df, ax=axe[i])

    axe[3].set_visible(False)


def get_statistical_description(df):
    statistical_description = df.describe().transpose()
    statistical_description['value_range'] = statistical_description['min'].astype(str) + ' - ' + statistical_description['max'].astype(str)
    statistical_description = statistical_description.drop(columns=['count', 'min', 'max', '25%', '75%'])
    return statistical_description


def get_job_seekers_by_gender(df):
    print('אחוז מחפשי העבודה לפי מגדר')
    return df[df['target'] == 0]['gender'].value_counts() / df['gender'].value_counts()


# get_all_plots(data)
