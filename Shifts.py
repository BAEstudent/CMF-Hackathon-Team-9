# The funciton "find_shifts()" takes in a list of integers (the demand for couriers),
# and returns the matrix of optimal shifts and the total number of shifts.

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
            cp.sum(N[:, 0]) >= demand[0],
            cp.sum(N[:, 0]) + cp.sum(N[:, 1]) >= demand[1],
            cp.sum(N[:, 0]) + cp.sum(N[:, 1]) + cp.sum(N[:, 2]) >= demand[2],
            cp.sum(N[:, 0]) + cp.sum(N[:, 1]) + cp.sum(N[:, 2]) + cp.sum(N[:, 3]) >= demand[3],
            cp.sum(N[1:, 0]) + cp.sum(N[:, 1]) + cp.sum(N[:, 2]) + cp.sum(N[:, 3]) + cp.sum(N[:3, 4]) >= demand[4],
            cp.sum(N[2:, 0]) + cp.sum(N[1:, 1]) + cp.sum(N[:, 2]) + cp.sum(N[:, 3]) +
            cp.sum(N[:3, 4]) + cp.sum(N[:2, 5]) >= demand[5],
            cp.sum(N[3:, 0]) + cp.sum(N[2:, 1]) + cp.sum(N[1:, 2]) + cp.sum(N[:, 3]) +
            cp.sum(N[:3, 4]) + cp.sum(N[:2, 5]) + cp.sum(N[:1, 6]) >= demand[6],
            N[4, 1] + cp.sum(N[3:, 2]) + cp.sum(N[2:, 3]) + cp.sum(N[1:3, 4]) +
            cp.sum(N[:2, 5]) + cp.sum(N[:1, 6]) + N[0, 7] >= demand[7],
            N[4, 1] + cp.sum(N[2:, 2]) + cp.sum(N[1:, 3]) +
            cp.sum(N[:3, 4]) + cp.sum(N[:2, 5]) + cp.sum(N[:1, 6]) + N[0, 7] >= demand[8],
            N[4, 2] + cp.sum(N[3:4, 3]) + cp.sum(N[2:3, 4]) + cp.sum(N[1:2, 5]) +
            cp.sum(N[0:1, 6]) + N[0, 7] >= demand[9],
            N[4, 3] + N[3, 4] + N[1, 6] + N[0, 7] >= demand[10],
        ]
        prob = cp.Problem(obj, cons)

        # Solving the problem and acquiring results
        prob.solve(solver=cp.GLPK_MI)

        Nums_of_couriers.append(np.matmul(ident_5, np.matmul(N.value, ident_11))[0])
        Shifts.append(N.value)

    return Shifts, Nums_of_couriers
