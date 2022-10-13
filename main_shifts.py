import numpy as np
import pandas as pd
import cvxpy as cp
import Shifts
import predict2task

FEATURES = ['partners_cnt', 'orders_cnt',
            'perc', 'month', 'day', 'hour', 'weekday',
            'day_of_year', 'day_freq', 'weekday_freq',
            'days_until_holiday', 'days_since_holiday', 'is_holiday', 'is_weekend',
            'days_until_nonworking', 'days_since_nonworking']

orders = pd.read_csv('orders.csv')
areas = orders['delivery_area_id'].unique()
orders_pred = pd.read_csv('orders_pred.csv')
dates = orders_pred['date'].copy()
couriers_pred = predict2task.pred_cur(orders_pred, dates, areas, FEATURES)


#data = pd.read_csv("couriers_pred.csv")
data = couriers_pred.copy()


#data = data.drop(columns=['date'])
result = Shifts.get_all_shifts(data)

shifts_only = []
for i in result:
    shifts_only.append(i[0])

df = pd.DataFrame(shifts_only)

shifts = df.rename(columns={col.old:f"area_id_{i}" for i, col_old in enumerate(data.columns)})
shifts.index = pd.Index(pd.date_range('2021-12-01', '2021-12-07'), name='date')
