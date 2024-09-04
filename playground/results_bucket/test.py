import numpy as np
from itertools import combinations_with_replacement


def calc_design_matrix(X, order):
    X = np.asarray(X)
    n_samples, n_features = X.shape
    des = np.ones((n_samples, 1))

    for o in range(1, order + 1):
        for combo in combinations_with_replacement(range(n_features), o):
            term = np.prod([X[:, i] for i in combo], axis=0)
            des = np.hstack((des, term[:, np.newaxis]))

    return des


# Example usage
X_test = np.array([[1, 2], [3, 4], [5, 6]])  # Sample input
order_test = 2  # Specify the order
design_matrix_output = calc_design_matrix(X_test, order_test)

# Displaying the resulting design matrix
print(X_test)
print(X_test.shape[1])
print(design_matrix_output)
print(design_matrix_output.shape[1])

#%%
