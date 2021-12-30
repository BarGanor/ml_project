import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .missing_values import replace_by_dict


def get_dummies(df, col_names):
    dummies = pd.get_dummies(df[col_names])
    dummies.columns = [col[col.rfind('_')+1:] for col in dummies.columns]
    dummies = dummies.drop(columns=['No relevent experience', 'Other', 'Part time course'])
    return dummies


def get_normalized(df, col_names):
    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df[col_names]), columns=col_names, index=df[col_names].index)

    return normalized_df


def get_df_represented(df):
    dummies = get_dummies(df, ['relevent_experience', 'gender', 'enrolled_university'])
    normalized = get_normalized(df, ['training_hours', 'experience', 'company_size', 'education_level', 'last_new_job', 'qualification_score', 'relevant_experience_years'])

    not_changed = df[['major_discipline', 'company_type', 'target', 'city_development_index', 'city', 'enrollee_id']]

    return pd.concat([not_changed, normalized, dummies], axis=1)

# dummies: relevant_experience, gender, enrolled_university
# normalization: training_hours, experience, company_size, education_level, last_new_job, qualification_score, relevant_experience_years
# not added: major_discipline, company_type, target , city_dev, city_id, enrolee_id
