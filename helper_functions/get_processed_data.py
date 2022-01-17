from sklearn.experimental import enable_iterative_imputer
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
    ##df = reduce_dimensionality(df)
    return df


def drop_nan(df):
    return drop_nan_by_thresh(df, 12)


def replace_values(df):
    df = replace_by_dict(df, 'company_size')
    df = replace_by_dict(df, 'last_new_job')
    df = replace_by_dict(df, 'major_discipline')
    df = replace_by_dict(df, 'experience')
    df = replace_by_dict(df, 'company_type')
    df = replace_by_dict(df, 'education_level')
    df = replace_by_dict(df, 'enrolled_university')
    df = replace_by_dict(df, 'gender')
    df = replace_by_dict(df, 'relevent_experience')
    return df


def fill_nan_values(df):
    df = fill_nan_with_mice(df, 'experience')
    df =  fill_nan_with_probability(df, 'company_size')
    df = fill_nan_with_probability(df, 'last_new_job')
    df = fill_nan_with_probability(df, 'major_discipline')
    #df = fill_nan_with_mice(df, 'education_level')
    df = fill_nan_with_max_appear(df, 'education_level')
    df = fill_nan_with_mice(df, 'enrolled_university')
    df = fill_nan_with_probability(df, 'enrolled_university')
    df = fill_nan_with_probability(df, 'company_type')
    df = fill_nan_with_mice(df, 'gender')
    df = fill_nan_with_mice(df, 'relevent_experience')
    #df = fill_nan_with_probability(df, 'relevent_experience')
    return df



def fill_nan_values_good(df):
    df = fill_nan_with_mice(df, 'experience')
    df = fill_nan_with_mice(df, 'company_size')
    df = fill_nan_with_mice(df, 'last_new_job')
    df = fill_nan_with_mice(df, 'major_discipline')
    df = fill_nan_with_mice(df, 'education_level')
    df = fill_nan_with_mice(df, 'enrolled_university')
    df = fill_nan_with_mice(df, 'company_type')
    df = fill_nan_with_mice(df, 'gender')
    df = fill_nan_with_mice(df, 'relevent_experience')
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
    backtracking_results = get_backtracking_results(df)
    correlation_df.index = correlation_df.index.droplevel()
    selection_df = pd.concat([correlation_df, backtracking_results], axis=1)
    selected_features = list(selection_df[selection_df['RFE_ranking'] <= 6].index)
    df = drop_not_selected(df, selected_features)
    return df


def reduce_dimensionality(df):
    return get_pca_results(df)
