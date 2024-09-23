import numpy as np
from scipy.integrate import quad


def b_spline_basis(t: float, k: int, i: int, knots: np.ndarray) -> float:
    """
    Recursive definition of B-spline basis functions.
    - t: The point at which to evaluate the B-spline.
    - k: The order of the B-spline (degree + 1).
    - i: The index of the B-spline basis function.
    - knots: The knot vector (array of knot positions).
    """
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    else:
        left_term = 0.0
        if knots[i + k] != knots[i]:
            left_term = ((t - knots[i]) / (knots[i + k] - knots[i])) * b_spline_basis(t, k - 1, i, knots)

        right_term = 0.0
        if knots[i + k + 1] != knots[i + 1]:
            right_term = ((knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1])) * b_spline_basis(t, k - 1, i + 1,
                                                                                                       knots)

        return left_term + right_term


def generate_knots(order: int, n_basis: int) -> np.ndarray:
    """
    Generate a knot vector with clamped ends.
    - order: The order of the B-spline.
    - n_basis: The number of basis functions.
    """
    # For clamped B-splines, the first and last knots are repeated order+1 times
    n_knots = n_basis + order + 1
    knots = np.concatenate(([0] * order, np.linspace(0, 1, n_knots - 2 * order), [1] * order))
    return knots


def J(order_X: int, n_basis_X: int, order_beta: int, n_basis_beta: int) -> np.ndarray:
    """
    Compute the matrix of integrals of B-spline basis products between X and beta.
    - order_X: The order of the B-spline for X.
    - n_basis_X: The number of basis functions for X.
    - order_beta: The order of the B-spline for beta.
    - n_basis_beta: The number of basis functions for beta.

    Returns:
    - A matrix of size (n_basis_X, n_basis_beta) where each element is the integral of
      the product of the corresponding basis functions for X and beta.
    """
    # Generate knot vectors for X and beta
    knots_X = generate_knots(order_X, n_basis_X)
    knots_beta = generate_knots(order_beta, n_basis_beta)

    # Initialize the result matrix
    result = np.zeros((n_basis_X, n_basis_beta))

    # Define the integration bounds (from 0 to 1)
    a, b = 0.0, 1.0

    # Iterate over all basis functions for X and beta
    for i in range(n_basis_X):
        for j in range(n_basis_beta):
            # Define the integrand: the product of the i-th basis function of X and the j-th basis function of beta
            def integrand(t):
                return b_spline_basis(t, order_X, i, knots_X) * b_spline_basis(t, order_beta, j, knots_beta)

            # Compute the integral using numerical integration
            integral_value, _ = quad(integrand, a, b)
            result[i, j] = integral_value

    return result


# # Example usage
# order_X = 0  # B-spline order for X (degree 0, piecewise constant)
# n_basis_X = 10  # Number of basis functions for X
# order_beta = 3  # B-spline order for beta (degree 3, cubic splines)
# n_basis_beta = 4  # Number of basis functions for beta
#
# integral_matrix = J(order_X, n_basis_X, order_beta, n_basis_beta)
# print(integral_matrix)

#%%
