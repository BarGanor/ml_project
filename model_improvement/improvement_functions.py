from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def get_normalized(df, col_names):
    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df[col_names]), columns=col_names, index=df[col_names].index)

    return normalized_df




def get_df_represented(df):
    dummies = get_dummies(df, ['relevent_experience', 'gender', 'enrolled_university'])
    normalized = get_normalized(df, ['training_hours', 'experience', 'company_size', 'education_level', 'last_new_job', 'qualification_score', 'relevant_experience_years'])

    not_changed = df[['major_discipline', 'company_type', 'target', 'city_development_index', 'city', 'enrollee_id']]

    return pd.concat([not_changed, normalized, dummies], axis=1)