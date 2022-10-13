# CMF-Hackathon-Team-9

This is the Git repository for Team 9 hackathon solution. It consists of several files:
- `Sber_hack_final_part_1.ipynb`,
- `task 2, delay correction.ipynb`,
- `Hackathon_Problem_3_Solution.ipynb`,
which are the solutions for the hackathon taks from 1 to 3 respectively.

### Tasks Solutions

- Task 1 The num_orders_pred(orders_file) receives a file with the demand data as input, and returns a pandas DataFrame with date indices for the following week (starting from the last date in the input file) and columns consisting of predicted numbers of orders for each delivery area ('delivery_id_area') from 0 to 592.
 XGBoost library was used for learning, implementing time series cross-validation.

- Task 2. After receiving orders predictions from Task 1 function predict_cur(..) predicts the number of couriers for the given orders. Prediction works in a following way: for each area the boosting model was training, the weights for model loads locally, then algorithm varies the number of couriers to seek for the non-delay, minimal number of couriers and minimal orders per courier var ('perc'). If there is a good prediction, function return model prediction, if there is no good solution from the model, function return a prediction of such number of couriers, that the 'perc' nearly close to ~ 2. Value 2 were empirically founded during the working process on the algorithm and might be considered optimal

- Task 3 was solved using the mixed-integer programming, minimizing the total number of working hours. For details, please refer to `Task_3_Solution_Design.pdf`, which contains the description of the solution for this task.

### Other

There are also several other files. 
- `couriers distribution.ipynb` is the code used for preparing the dataset (addind new features and cleaning the data) for the ML part.

- `Shifts.py` is the module for solving the 3-rd task of the hackathon, consists of a function `find_shifts(Demand)`, which takes the input from the model used for task 2, and outputs two variables: a list of shift schedules, and the list of total couriers reuqired on each day.

- `Task_3_Solution_Design.pdf` is a pdf file, containing the discription of the task 3 solution.

- `weights2task` is a folder, containing model weights for each area for Task2
- `train2task.py` is a package, containing functions for training models in Task2
- `orders_pred.csv` is a resulting .csv file with predictions of number of orders from Task 1
- `couriers_pred.csv` is a resulting .csv file with predictions of couriers number from Task 2
