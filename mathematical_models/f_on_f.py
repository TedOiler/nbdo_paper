from .base_model import BaseModel
import numpy as np
import os
import sys
from scipy.linalg import block_diag


class FunctionOnFunctionModel(BaseModel):
    def __init__(self, Kx, Kb, Kx_family, Ky, Kb_family='polynomial', k_degree=None, Sigma_decay=0):
        self.Kx_family = Kx_family
        self.Kx = Kx
        self.Kb_family = Kb_family
        self.Kb = Kb
        self.Ky = Ky
        self.k_degree = k_degree

        self.knots_num = self.Kx[0] + 1
        self.J_CH = self.compute_Jcb()
        self.Sigma_decay = Sigma_decay
        self.Sigma = self.compute_Sigma()

        self.J_cb = self.compute_Jcb()

    def compute_objective(self, Gamma_, N, Kx):
        Gamma = np.hstack((np.ones((N, 1)), Gamma_))
        Z = Gamma @ self.J_CH
        ZtZ_inv = np.linalg.inv(Z.T @ Z)

        # A-optimality: Trace of the covariance matrix
        value = np.trace(ZtZ_inv) * np.trace(self.Sigma)

        return value if np.isfinite(value) else np.nan

    def compute_objective_input(self, x, i, j, Gamma_, N, Kx):
        Gamma_[i, j] = x
        return self.compute_objective(Gamma_, N, Kx)

    def compute_objective_relative(self, Gamma, N, Kx, Sigma_new, objective_old):
        Gamma_mat = np.hstack((np.ones((N, 1)), Gamma))
        Z = Gamma_mat @ self.J_CH
        ZtZ_inv = np.linalg.inv(Z.T @ Z)

        # A-optimality: Trace of the covariance matrix
        objective_new = np.trace(ZtZ_inv) * np.trace(Sigma_new)

        # In practice, you might want to ensure 'value' is valid (e.g., not NaN or Inf) before returning
        return np.exp(np.log(objective_new) - np.log(objective_old)) if np.isfinite(objective_new) else np.nan

    def compute_Jcb(self):
        if self.Kx_family == 'step':
            sys.path.append(os.path.abspath("../utilities"))
            from utilities.J.matrix_calc import Jcb, calc_basis_matrix
            Jcb = Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b) for x, b in zip(self.Kx, self.Kb)])
            return block_diag(1, Jcb)

    def get_Jcb(self):
        return self.J_cb

    def compute_Sigma(self):
        inputs = np.linspace(0, 1, self.Ky)

        def exp_decay(dec_rate, x):
            return np.exp(-dec_rate * x)

        if self.Sigma_decay == 0:
            Sigma = np.diag(exp_decay(self.Sigma_decay, inputs))
            return Sigma
        elif self.Sigma_decay == np.inf:
            elements = np.zeros((len(self.Kx) + 1) * self.Ky)
            Sigma = np.diag(elements)
            Sigma[0, 0] = 1
            return Sigma
        else:
            Sigma = np.diag(exp_decay(self.Sigma_decay, inputs))
            return Sigma

    def get_Sigma(self):
        return self.Sigma
