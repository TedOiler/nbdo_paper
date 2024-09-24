import numpy as np
from scipy.integrate import quad
from basis.bspline import BSplineBasis
from basis.polynomial import PolynomialBasis
from basis.fourier import FourierBasis


class JMatrix:
    def __init__(self, basis1, basis2):
        self.basis1 = basis1
        self.basis2 = basis2

    def compute(self):
        num_basis1 = self.basis1.num_basis()
        num_basis2 = self.basis2.num_basis()
        J = np.zeros((num_basis1, num_basis2))

        for i in range(num_basis1):
            for j in range(num_basis2):
                # Determine overlapping support
                a1, b1 = self.basis1.get_basis_support(i)
                a2, b2 = self.basis2.get_basis_support(j)
                a = max(a1, a2)
                b = min(b1, b2)
                if a >= b:
                    J[i, j] = 0.0
                    continue

                def integrand(t):
                    return self.basis1.evaluate_basis_function(i, t) * self.basis2.evaluate_basis_function(j, t)

                integral_value, _ = quad(integrand, a, b, limit=1000)
                J[i, j] = integral_value

        return J
