from .basis import Basis


class PolynomialBasis(Basis):
    def __init__(self, degrees):
        self.degrees = degrees  # List of degrees for each basis function
        self.num_basis_functions = len(degrees)

    def evaluate_basis_function(self, i, t):
        return t ** self.degrees[i]

    def get_basis_support(self, i):
        return 0.0, 1.0  # Polynomials are defined over [0,1]

    def num_basis(self):
        return self.num_basis_functions
