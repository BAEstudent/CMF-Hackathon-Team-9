# The funciton "find_shifts()" takes in a list of integers (the demand for couriers),
# and returns the matrix of optimal shifts and the total number of shifts.
import pandas as pd
import numpy as np
import cvxpy as cp

def find_shifts(demands):
    ## Lists of variables for each weekday
    Nums_of_couriers = []
    Shifts = []

    ## Parameters for the minimization problem (penalties for longer shits)
    alpha = 4
    beta = 3
    gamma = 2
    zeta = 1

    # Creating the decision variables and supplementary variables
    N = cp.Variable(shape=(5, 11), integer=True)
    ident_11 = np.ones((11, 1))
    ident_5 = np.ones((5,))

    for demand in demands:
        # Formulating the objective function and necessary constraints
        obj = cp.Minimize(ident_5 @ (N @ ident_11) + alpha * (cp.sum(N[4, :])) +
                          beta * (cp.sum(N[3, :])) + gamma * (cp.sum(N[2, :])) + zeta * (cp.sum(N[1, :])))

        cons = [
            N >= 0,
            cp.sum(N[:, 0]) >= demand.iloc[0],
            cp.sum(N[:, 0]) + cp.sum(N[:, 1]) >= demand.iloc[1],
            cp.sum(N[:, 0]) + cp.sum(N[:, 1]) + cp.sum(N[:, 2]) >= demand.iloc[2],
            cp.sum(N[:, 0]) + cp.sum(N[:, 1]) + cp.sum(N[:, 2]) + cp.sum(N[:, 3]) >= demand.iloc[3],
            cp.sum(N[1:, 0]) + cp.sum(N[:, 1]) + cp.sum(N[:, 2]) + cp.sum(N[:, 3]) + cp.sum(N[:3, 4]) >= demand.iloc[4],
            cp.sum(N[2:, 0]) + cp.sum(N[1:, 1]) + cp.sum(N[:, 2]) + cp.sum(N[:, 3]) +
            cp.sum(N[:3, 4]) + cp.sum(N[:2, 5]) >= demand.iloc[5],
            cp.sum(N[3:, 0]) + cp.sum(N[2:, 1]) + cp.sum(N[1:, 2]) + cp.sum(N[:, 3]) +
            cp.sum(N[:3, 4]) + cp.sum(N[:2, 5]) + cp.sum(N[:1, 6]) >= demand.iloc[6],
            N[4, 1] + cp.sum(N[3:, 2]) + cp.sum(N[2:, 3]) + cp.sum(N[1:3, 4]) +
            cp.sum(N[:2, 5]) + cp.sum(N[:1, 6]) + N[0, 7] >= demand.iloc[7],
            N[4, 1] + cp.sum(N[2:, 2]) + cp.sum(N[1:, 3]) +
            cp.sum(N[:3, 4]) + cp.sum(N[:2, 5]) + cp.sum(N[:1, 6]) + N[0, 7] >= demand.iloc[8],
            N[4, 2] + cp.sum(N[3:4, 3]) + cp.sum(N[2:3, 4]) + cp.sum(N[1:2, 5]) +
            cp.sum(N[0:1, 6]) + N[0, 7] >= demand.iloc[9],
            N[4, 3] + N[3, 4] + N[1, 6] + N[0, 7] >= demand.iloc[10],
        ]
        prob = cp.Problem(obj, cons)

        # Solving the problem and acquiring results
        prob.solve(solver=cp.GLPK_MI)

        Nums_of_couriers.append(np.matmul(ident_5, np.matmul(N.value, ident_11))[0])
        Shifts.append(N.value)

    return Shifts, Nums_of_couriers

def get_all_shifts(data):
    all_shifts = []
    for id in data:
        a, b =(find_shifts(chunks(data[id])))
        all_shifts.append([a,b])
    return(all_shifts)

def chunks(lst):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), 11):
        yield lst[i:i + 11]
