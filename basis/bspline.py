from .basis import Basis
import numpy as np


class BSplineBasis(Basis):
    def __init__(self, degree, total_knots_num):
        self.degree = degree
        self.order = degree + 1
        self.internal_knots_num = total_knots_num

        self.internal_knots = np.linspace(0, 1, self.internal_knots_num)[1:-1]
        self.lower_bound_knots = np.zeros(self.order)
        self.upper_bound_knots = np.ones(self.order)
        self.augmented_knots = np.concatenate((self.lower_bound_knots, self.internal_knots, self.upper_bound_knots))

        self.num_basis_functions = len(self.augmented_knots) - self.order

    def evaluate_basis_function(self, i, t):
        return self._evaluate_bspline_basis_function(i, self.degree, t)

    def _evaluate_bspline_basis_function(self, i, k, t):
        knots = self.augmented_knots
        if k == 0:
            if knots[i] <= t < knots[i + 1]:
                return 1.0
            elif t == knots[-1] and t == knots[i + 1]:
                return 1.0  # Handle special case at the end of the knot vector
            else:
                return 0.0
        else:
            denom1 = knots[i + k] - knots[i]
            term1 = 0.0
            if denom1 != 0:
                term1 = ((t - knots[i]) / denom1) * self._evaluate_bspline_basis_function(i, k - 1, t)

            denom2 = knots[i + k + 1] - knots[i + 1]
            term2 = 0.0
            if denom2 != 0:
                term2 = ((knots[i + k + 1] - t) / denom2) * self._evaluate_bspline_basis_function(i + 1, k - 1, t)

            return term1 + term2

    def get_basis_support(self, i):
        start = self.augmented_knots[i]
        end = self.augmented_knots[i + self.order]
        return start, end

    def num_basis(self):
        return self.num_basis_functions
