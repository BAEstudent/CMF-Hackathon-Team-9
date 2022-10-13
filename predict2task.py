#!/usr/bin/env python
# coding: utf-8

# # Predictions of optimal courier number

# ### Using XGBoost classifier for each unique area

# ### Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import xgboost as xgb

import imblearn
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler

import holidays as hld
import json
import requests as rq
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import Counter

# from xgboost import XGBClassifier

import warnings
warnings.simplefilter("ignore")


# In[2]:


def choose_region(df, area):
    #get values of particular area
    return df[df['delivery_area_id_x']==area]


# ## Feature creation
# Added:
# * day of week
# * hour
# * weekday
# * day_of_year
# * month
# * average particular day of month frequency in all observations
# * average particular day of week frequency in all observations
# * days until nearest holiday
# * days after nearest holiday
# * days until nearest non-working day
# * days after nearest non-working day
# * if a day is holiday
# * if a day is weekend

# In[3]:


def create_features(delays):
    
    delays = delays.reset_index()
    delays['dttm'] = pd.to_datetime(delays['dttm'])
    delays_dates = delays['dttm']
    delays['day'] = delays_dates.apply(lambda x: x.day)
    delays['hour'] = delays_dates.apply(lambda x: x.hour)
    delays['weekday'] = delays_dates.apply(lambda x: x.weekday())
    delays['day_of_year'] = delays_dates.apply(lambda x: x.dayofyear)
    delays['month'] = delays['day_of_year'] = delays_dates.apply(lambda x: x.month)
    day_freq_delays = delays['day'].value_counts(normalize=True).to_dict()
    delays['day_freq'] = delays['day'].map(day_freq_delays)
    weekday_freq_delays = delays['weekday'].value_counts(normalize=True).to_dict()
    delays['weekday_freq'] = delays['weekday'].map(weekday_freq_delays)
    
    
    delays_dates = delays['dttm']
    
    holidays = [date[0] for date in hld.Russia(years=2021).items()]

    delays_dates = delays_dates.to_frame().sort_values(by='dttm')

    df_holidays = pd.DataFrame({'holidays': holidays})
    df_holidays['holidays'] = pd.to_datetime(df_holidays['holidays'])

    delays_dates = pd.merge_asof(delays_dates, df_holidays, left_on='dttm', right_on='holidays', direction='forward')
    delays_dates = pd.merge_asof(delays_dates, df_holidays, left_on='dttm', right_on='holidays')

    delays_dates['days_until_holiday'] = delays_dates.pop('holidays_x').sub(delays_dates['dttm']).dt.days
    delays_dates['days_since_holiday'] = delays_dates['dttm'].sub(delays_dates.pop('holidays_y')).dt.days

    delays_dates =  delays_dates.drop_duplicates()
    delays = pd.merge(delays, delays_dates, how='left', on='dttm')
    delays_dates = delays['dttm']

    url = 'https://raw.githubusercontent.com/d10xa/holidays-calendar/master/json/consultant2021.json'
    response = rq.get(url)
    non_working = json.loads(response.text)

    for days in non_working:
        non_working[days] = pd.to_datetime(non_working.get(days)).date

    delays['is_holiday'] = delays_dates.apply(lambda x: x.date() in non_working['holidays']) * 1
    delays['is_weekend'] = ((delays['weekday']==5) | (delays['weekday']==6)) * 1
    delays_dates = delays_dates.to_frame().sort_values(by='dttm')

    sorted_nonworking = np.sort(np.concatenate((non_working['holidays'], non_working['nowork'])))

    df_non_working = pd.DataFrame({'non_working': sorted_nonworking})
    df_non_working['non_working'] = pd.to_datetime(df_non_working['non_working'])

    delays_dates = pd.merge_asof(delays_dates, df_non_working, left_on='dttm', right_on='non_working', direction='forward')
    delays_dates = pd.merge_asof(delays_dates, df_non_working, left_on='dttm', right_on='non_working')

    delays_dates['days_until_nonworking'] = delays_dates.pop('non_working_x').sub(delays_dates['dttm']).dt.days
    delays_dates['days_since_nonworking'] = delays_dates['dttm'].sub(delays_dates.pop('non_working_y')).dt.days

    delays_dates =  delays_dates.drop_duplicates()

    delays = pd.merge(delays, delays_dates, how='left', on='dttm')
    
    delays = delays.set_index('dttm')

    return delays


# ## Oversampling
# 
# Oversample given observation with synthetic data to get 50/50 target class distribution

# In[4]:


def oversampling(df, TARGET):
    #Get the negative to positive weight class estimation
    def est(df, TARGET):
        counter = Counter(df[TARGET])
        # estimate scale_pos_weight value
        return counter[0] / counter[1]
        
    print(f"Initial negative to positive classes weight disbalance: {est(df, TARGET):.3f}")
    
    ### PREVIOUSLY USED TO UNIFORMLY DISTRIBUTE COURIERS ACROSS DATA FOR BETTER CLASSIFICATION
    ### UNSUCCESFUL ATTEMPT 
    # con['partners_cnt'] = con['partners_cnt'].apply(round)
    # sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    # # Fit the model to generate the data.
    # oversampled_trainX, oversampled_trainY = sampler.fit_resample(con.drop('partners_cnt', axis=1), con['partners_cnt'])
    # oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)
    # con = oversampled_train.copy()
    
    sm = SMOTE(sampling_strategy='auto', random_state=42)

    # Fit the model to generate the data.
    oversampled_trainX, oversampled_trainY = sm.fit_resample(df.drop(TARGET, axis=1), (df[TARGET]>0.05)*1)
    oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)
    
    print(f"Final negative to positive classes weight disbalance: {est(oversampled_train, TARGET):.3f}")
    
    return oversampled_train


# In[5]:


def recreate_dates(df):
    #after resampling, the date index values are dropped and chanded to range index, here we recreate dates
    
    dt = lambda x: '0' + str(x) if len(str(x))==1 else str(x)
    df['month'] = df['month'].apply(dt)
    df['day'] = df['day'].apply(dt)
    df['hour'] = df['hour'].apply(dt)
    
    df['dttm'] = '2021'+'-'+df['month']+'-'+df['day']+' '+df['hour']
    tr_dates = lambda x: datetime.strptime(x, "%Y-%m-%d %H")
    df['dttm'] = df['dttm'].apply(tr_dates)
    df = df.set_index('dttm')
    
    df['month'] = df['month'].astype('int')
    df['day'] = df['day'].astype('int')
    df['hour'] = df['hour'].astype('int')
    
    
    return df


# ## Function for varying couriers
# Vary couriers number to get different values for delay and find the optimal num of couriers afterwards

# In[6]:


def vary_couriers(df, model, FEATURES):
    
    df_proxy = df.copy()
    df_lst = []
    for i in range(len(df_proxy)):
        proxy = pd.DataFrame([df_proxy.iloc[i]]*20)
        proxy["partners_cnt"] = range(1, 21)
        df_lst.append(proxy)
    
    df_exp = pd.concat(df_lst)
    df_exp['perc'] = df_exp['orders_cnt'] / df_exp['partners_cnt']
    
    if 'delay' in df_exp.columns:
        df_exp = df_exp.drop('delay', axis=1)
    
    preds = model.predict(df_exp[FEATURES])

    df_varied = pd.concat([df_exp.reset_index(), pd.DataFrame(preds, columns=['delay'])], axis=1)
    
    return df_varied


# ## Function for predictions
# Choose the best prediction to solve the task
# 
# As some predictions from varying the couriers might be definitely wrong (data distribution problems), function makes empirically received border for delay, so that we consider the 'perc' values higher than 2.5 as the definite delay. After that we sort values and get the best prediction that has lowest delay (0 or 1), num of couriers ('partners_cnt') and 'perc'.
# 
# If the model predicted that there would be only delays for particular datetime, function returns the result, closest to perc ~ 2, as we consider such delay predictions as a data distribution problems

# In[7]:


def choose_best_pred(con_cls_proxy):
    con_cls_proxy.loc[con_cls_proxy['perc']>2.5, 'delay'] = 1 
    
    predictions = []
    for i in con_cls_proxy['index'].unique():
        prx = con_cls_proxy[con_cls_proxy['index']==i].sort_values(by=['delay', 'partners_cnt', 'perc'])

        if prx['delay'].iloc[0] == 1:

            prx = prx.iloc[(prx['perc'] - 2).abs().argsort(),:]

        prediction = prx.iloc[[0]]
        predictions.append(prediction)
    
    return pd.concat(predictions)


# In[8]:


FEATURES = ['partners_cnt', 'orders_cnt', 
            'perc', 'month', 'day', 'hour', 'weekday', 
            'day_of_year', 'day_freq', 'weekday_freq',
            'days_until_holiday', 'days_since_holiday', 'is_holiday', 'is_weekend',
            'days_until_nonworking', 'days_since_nonworking']

TARGET = 'delay'


# In[9]:


#make predictions by loading the weights

def pred_cur(orders, dates, areas, FEATURES):
    preds_by_area = []

    for area in tqdm(areas):

        con = orders_pred[['date', f'area_id_{area}']].rename(columns={'date': 'dttm', f'area_id_{area}': 'orders_cnt'})
        con['orders_cnt'] = con['orders_cnt'].apply(np.ceil)
        reg = xgb.XGBClassifier()
        reg.load_model(f'..\\courier hackathon\\weights2task\\area_{area}.json')
        con = create_features(con)
        con_var = vary_couriers(con, reg, FEATURES)
        predictions = choose_best_pred(con_var)['partners_cnt'].values

        preds_by_area.append(pd.Series(predictions, name=f'area_id_{area}'))
    
    final_couriers = pd.concat(preds_by_area, axis=1)
    final_couriers.index = dates
    final_couriers.to_csv('couriers_pred.csv')
    return final_couriers


# In[10]:


orders = pd.read_csv('orders.csv')
areas = orders['delivery_area_id'].unique()
orders_pred = pd.read_csv('orders_pred.csv')
dates = orders_pred['date'].copy()
pred_cur(orders_pred, dates, areas, FEATURES)


# In[ ]:


# final_couriers.to_csv('couriers_pred.csv')


# -----
