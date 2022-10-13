import numpy as np
import pandas as pd
import cvxpy as cp
import Shifts
import

data = pd.read_csv("couriers_pred.csv")
data = data.drop(columns=['date'])
result = Shifts.get_all_shifts(data)

print(result)