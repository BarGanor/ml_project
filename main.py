#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
path = '/Users/barganor/Downloads/XY_train (1).csv'

data = pd.read_csv(path)
data.head()

#%% md

### Cities

#%%

cities_count = data['city'].value_counts()
cities_count.hist()

#%% md

### Cities development

#%%

data.groupby('city')['city_development_index'].mean().plot.density()

#%% md

### Gender

#%%

data['gender'].value_counts(dropna=False).plot.pie()

#%% md

### Relevant experience

#%%

data['relevent_experience'].value_counts().plot(kind='bar')

#%%

data['education_level'].value_counts().plot(kind='bar')

#%%



#%%

data['company_type'].value_counts(dropna=False).plot(kind='bar')

#%%

sns.kdeplot(data['training_hours'])

#%%

statistical_description = data.describe().transpose()
statistical_description['value_range'] = statistical_description['min'].astype(str) + ' - ' + statistical_description['max'].astype(str)
statistical_description = statistical_description.drop(columns=['count', 'min','max','25%','75%'])
statistical_description

#%%

data['target'].value_counts().plot.bar()

#%%

print('אחוז מחפשי העבודה לפי מגדר')
data[data['target']==0]['gender'].value_counts() / data['gender'].value_counts()

#%%

median_city_development = data.groupby('city')['city_development_index'].mean().describe()['50%']
job_seekers_in_good_cities = data[(data['city_development_index']>median_city_development) & (data['target'] == 0)]['target'].count()/ data['target'].count()
job_seekers_in_bad_cities = data[(data['city_development_index']<median_city_development) & (data['target'] == 0)]['target'].count()/ data['target'].count()

pd.DataFrame({'Looking for a job highly developed cities': job_seekers_in_good_cities, 'Looking for a job un-developed cities': job_seekers_in_bad_cities},index=['Percentage'])

#%% md

# Pre Processing

#%%

# Make copy of original data
processed_data = data.copy()

#%%

# Check for duplicated data
print(processed_data.shape)
print(processed_data.drop_duplicates().shape)

#%%

# Dealing with missing values
print(processed_data.isna().sum())

#%%



#%%

# Transform experience to floats and fill nan with the median.
replacement_dct = {'>20':'25','<1':'0'}
processed_data['experience'] = processed_data['experience'].replace(replacement_dct).astype('float64')

# Calculate median
exp_median = np.median(processed_data['experience'].dropna().values)

# Change nan values to calculated median
processed_data['experience'] = processed_data['experience'].fillna(exp_median)

#%%

processed_data = processed_data.dropna(thresh=12, axis=0)

#%%

print(processed_data.isna().sum())

#%%

size_replacement_dict = {
                        '50-99':3,
                        '100-500':4,
                        '10000+':8,
                        'Oct-49':2,
                        '1000-4999':6,
                        '<10':1,
                        '500-999':5,
                        '5000-9999':7
                        }

processed_data['company_size'] = processed_data['company_size'].replace(size_replacement_dict)



print(processed_data.shape)

#%%

print(processed_data.dropna(thresh=13, axis=0).isna().sum())

#%%



#%%

# Code that shoes that "company_size" is not important for forecasting
print(pd.DataFrame({'size':processed_data['company_size'].values, 'target': processed_data['target']}).groupby('size').sum() / pd.DataFrame({'size':processed_data['company_size'].values, 'target': processed_data['target']}).groupby('size').count())
