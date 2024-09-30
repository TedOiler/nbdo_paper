from .basis import Basis
import numpy as np


class FourierBasis(Basis):
    def __init__(self, num_basis_functions):
        self.num_basis_functions = num_basis_functions

    def evaluate_basis_function(self, i, t):
        if i == 0:
            return 1.0  # Constant term
        elif i % 2 == 1:
            n = (i + 1) // 2
            return np.sin(2 * np.pi * n * t)
        else:
            n = i // 2
            return np.cos(2 * np.pi * n * t)

    def get_basis_support(self, i):
        return 0.0, 1.0

    def num_basis(self):
        return self.num_basis_functions
