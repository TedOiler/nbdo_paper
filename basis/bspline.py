from .basis import Basis
import numpy as np


class BSplineBasis(Basis):
    def __init__(self, degree, num_basis_functions):
        self.degree = degree
        self.num_basis_functions = num_basis_functions
        self.knot_vector = self.generate_knot_vector()

    def generate_knot_vector(self):
        num_knots = self.num_basis_functions + self.degree + 1
        knot_vector = np.zeros(num_knots)
        knot_vector[:self.degree + 1] = 0.0
        knot_vector[-(self.degree + 1):] = 1.0
        num_internal_knots = num_knots - 2 * (self.degree + 1)
        if num_internal_knots > 0:
            internal_knots = np.linspace(0, 1, num_internal_knots + 2)[1:-1]
            knot_vector[self.degree + 1:-self.degree - 1] = internal_knots
        return knot_vector

    def evaluate_basis_function(self, i, t):
        return self._evaluate_bspline_basis_function(i, self.degree, t)

    def _evaluate_bspline_basis_function(self, i, k, t):
        knots = self.knot_vector
        if k == 0:
            if knots[i] <= t < knots[i + 1]:
                return 1.0
            elif t == knots[-1] and t == knots[i + 1]:
                return 1.0
            else:
                return 0.0
        else:
            denom1 = knots[i + k] - knots[i]
            if denom1 != 0:
                term1 = (t - knots[i]) / denom1 * self._evaluate_bspline_basis_function(i, k - 1, t)
            else:
                term1 = 0.0

            denom2 = knots[i + k + 1] - knots[i + 1]
            if denom2 != 0:
                term2 = (knots[i + k + 1] - t) / denom2 * self._evaluate_bspline_basis_function(i + 1, k - 1, t)
            else:
                term2 = 0.0

            return term1 + term2

    def get_basis_support(self, i):
        knots = self.knot_vector
        return knots[i], knots[i + self.degree + 1]

    def num_basis(self):
        return self.num_basis_functions
