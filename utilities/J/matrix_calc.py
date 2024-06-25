import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag

from utilities.basis.basis import indicator, polynomial, b_spline_basis


def elements(n, p, l1=0) -> float:
    return quad(lambda t: (indicator(t, l=l1 / n, u=(l1 + 1) / n)) * (polynomial(t, p=p)), l1 / n, (l1 + 1) / n,
                full_output=True)[0]


def elements_f_on_f(k, knots, p):
    return quad(
        lambda t: (b_spline_basis(t, k, knots)) * (polynomial(t, p)),
        0, 1,
        full_output=True
    )[0]


def calc_basis_matrix(x_basis, b_basis) -> np.ndarray:
    return np.array([[elements(n=x_basis, p=p, l1=l1) for p in range(b_basis)] for l1 in range(x_basis)])


def calc_basis_matrix_f_on_f(k, x_basis, b_basis):
    return np.array(
        [
            [elements_f_on_f(k=k, knots=knots, p=p) for p in range(b_basis)]
            for knots in range(x_basis)
        ]
    )


def Jcb_f_on_f(*matrices):
    return block_diag(*matrices)


def Jcb(*matrices):
    return block_diag(*matrices)


def calc_Sigma(Kx, Ky, N, decay=0):
    In = np.eye(N)
    # Shape = (len(Kx) + 1) * Ky
    Shape = Ky
    inputs = np.linspace(0, 1, Shape)

    def exp_decay(dec_rate, x):
        return np.exp(-dec_rate * x)

    if decay == 0:
        sigma1 = np.diag(exp_decay(decay, inputs))
    elif decay == np.inf:
        elements = np.zeros(Shape)
        # elements[0] = 1 # not sure
        sigma1 = np.diag(elements)
        sigma1[0, 0] = 1
    else:
        sigma1 = np.diag(exp_decay(decay, inputs))
    return np.kron(In, sigma1)


def calc_J_CH(Kx, Kb):
    J_chs = [Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b) for x, b in zip(x_row, b_row)]) for x_row, b_row in
             zip(Kx, Kb)]
    bases = [1] + J_chs
    return block_diag(*bases)


def calc_I_theta(Ky):
    return np.eye(Ky,
                  dtype=float)  # Most likely correct expression, need to figure out the correct dimensions of all the matrices
    # return np.kron(np.ones(len(Kx) + 1), np.eye(Ky)) # second attempt to fix the error
    # return np.eye((len(Kx)+1)*Ky, dtype=float) # old return statement


def calc_I_N(N):
    return np.eye(N, dtype=float)
