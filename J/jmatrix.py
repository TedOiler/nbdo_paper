import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag
from basis.bspline import BSplineBasis
from basis.polynomial import PolynomialBasis
from basis.fourier import FourierBasis


class JMatrix:
    def __init__(self, basis_pairs):
        self.basis_pairs = basis_pairs

    @staticmethod
    def _compute_element(basis1, basis2):
        num_basis1 = basis1.num_basis()
        num_basis2 = basis2.num_basis()
        J = np.zeros((num_basis1, num_basis2))

        for i in range(num_basis1):
            for j in range(num_basis2):
                # Determine overlapping support
                a1, b1 = basis1.get_basis_support(i)
                a2, b2 = basis2.get_basis_support(j)
                a = max(a1, a2)
                b = min(b1, b2)
                if a >= b:
                    J[i, j] = 0.0
                    continue

                def integrand(t):
                    return basis1.evaluate_basis_function(i, t) * basis2.evaluate_basis_function(j, t)

                integral_value, _ = quad(integrand, a, b, limit=1000)
                J[i, j] = integral_value

        return J

    def compute(self):
        J_blocks = []
        for basis1, basis2 in self.basis_pairs:
            J_element = self._compute_element(basis1, basis2)
            J_blocks.append(J_element)

        J = block_diag(*J_blocks)
        return J
