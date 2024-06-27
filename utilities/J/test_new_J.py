import numpy as np
from scipy.integrate import quad


# TODO: Good idea to implement the following functions in the J object one day
def b_spline_basis(t, k, i, knots):
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


def polynomial(t, p):
    return t ** p


def inner_product(k, knot_num, p):
    knots = [0.] * k + list(np.linspace(0, 1, knot_num + 1)) + [1.] * k
    J = np.zeros((knot_num, p))
    interval = 1 / knot_num

    for i in range(knot_num):
        for j in range(p):
            result, _ = quad(lambda t: b_spline_basis(t, k, i, knots) * polynomial(t, j), knots[i + k],
                             knots[i + k + 1])
            J[i, j] = result

    return J


k = 0
knot_num = 100
p = 2

J = np.round(inner_product(k, knot_num, p), 3)
print(J)

# %%
