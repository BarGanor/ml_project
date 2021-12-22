from .exploratory_analysis import *
from .feature_selection import *
from .feature_extraction import *
from .feature_representation import *
from .missing_values import *
from .dimensionality_reduction import *


def get_processed_data(df):
    df = drop_nan(df)
    df = replace_values(df)
    df = fill_nan_values(df)
    df = extract_features(df)
    df = represent_data(df)
    df = select_features(df)
    df = reduce_dimensionality(df)
    return df


def drop_nan(df):
    return drop_nan_by_thresh(data, 12)


def replace_values(df):
    df = replace_by_dict(df, 'company_size')
    df = replace_by_dict(df, 'last_new_job')
    df = replace_by_dict(df, 'major_discipline')
    df = replace_by_dict(df, 'experience')
    df = replace_by_dict(df, 'company_type')
    return df


def fill_nan_values(df):
    df = fill_nan_with_median(df, 'experience')
    df = fill_nan_with_probability(df, 'company_size')
    df = fill_nan_with_probability(df, 'last_new_job')
    df = fill_nan_with_probability(df, 'major_discipline')
    df = fill_nan_with_max_appear(df, 'education_level')
    df = fill_nan_with_probability(df, 'enrolled_university')
    df = fill_nan_with_probability(df, 'company_type')
    df = fill_gender_with_knn(df)
    return df


def extract_features(df):
    df = get_qualification_index(df)
    df = get_relevant_experience_feature(df)
    return df


def represent_data(df):
    return get_df_represented(df)


def select_features(df):
    correlation_df = get_correlation_df(df)
    df = drop_unimportant_vars(df, correlation_df)

    # Choosing final selection
    selected_features = ['city_development_index', 'experience', 'qualification_score', 'enrollment', 'training_hours']
    df = drop_not_selected(df, selected_features)
    return df


def reduce_dimensionality(df):
    return get_pca_results(df)
