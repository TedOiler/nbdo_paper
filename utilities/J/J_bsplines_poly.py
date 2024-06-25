import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag

from utilities.basis.basis import b_spline_basis, polynomial


def elements(n, p, l1=0, k=3, knots_num=None) -> float:
    if knots_num is None:
        knots = [0.] * k + list(np.linspace(0, 1, k + 1)) + [1.] * k
    else:
        knots = [0.] * k + list(np.linspace(0, 1, knots_num - k)) + [1.] * k

    b_spline = lambda t: b_spline_basis(t, k, l1, knots)
    return quad(lambda t: b_spline(t) * polynomial(t, p=p), 0, 1, full_output=True)[0]


def calc_basis_matrix(x_basis, b_basis, k=3, knots_num=None) -> np.ndarray:
    return np.array([[elements(n=x_basis, p=p, l1=l1, k=k, knots_num=knots_num) for p in range(b_basis)] for l1 in
                     range(x_basis)])


def Jcb(*matrices):
    return block_diag(*matrices)
