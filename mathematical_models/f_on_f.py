from .base_model import BaseModel
import numpy as np
import os
import sys


class FunctionOnFunctionModel(BaseModel):
    def __init__(self, I_theta, I_N, J_CH, Sigma,
                 Kx, Kb, Kx_family, Kb_family='polynomial', k_degree=None, knots_num=None, ):
        self.Kx_family = Kx_family
        self.Kx = Kx
        self.Kb_family = Kb_family
        self.Kb = Kb
        self.k_degree = k_degree
        self.knots_num = self.Kx[0] + 1
        self.I_theta = I_theta
        self.I_N = I_N
        self.J_CH = J_CH
        self.Sigma = Sigma

        self.J_cb = self.compute_Jcb()

    def compute_objective(self, Gamma_, N, Kx):
        Gamma = np.hstack((np.ones((N, 1)), Gamma_))
        Z = Gamma @ self.J_CH
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        F1 = (Z @ ZtZ_inv).T
        F = np.kron(F1, self.I_theta)
        # Delta = np.kron(self.I_N, self.Sigma)
        Covar = F @ self.Sigma @ F.T

        # A-optimality: Trace of the covariance matrix
        value = np.trace(Covar)

        # In practice, you might want to ensure 'value' is valid (e.g., not NaN or Inf) before returning
        return value if np.isfinite(value) else np.nan

    def compute_objective_input(self, x, i, j, Gamma_, N, Kx):
        Gamma_[i, j] = x
        return self.compute_objective(Gamma_, N, Kx)

    def compute_objective_relative(self, Gamma, N, Kx, Sigma_new, objective_old):
        Gamma_mat = np.hstack((np.ones((N, 1)), Gamma))
        Z = Gamma_mat @ self.J_CH
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        F1 = (Z @ ZtZ_inv).T
        F = np.kron(F1, self.I_theta)
        # Delta = np.kron(self.I_N, self.Sigma)
        Covar_new = F @ Sigma_new @ F.T

        # A-optimality: Trace of the covariance matrix
        objective_new = np.trace(Covar_new)

        # In practice, you might want to ensure 'value' is valid (e.g., not NaN or Inf) before returning
        return np.exp(np.log(objective_new) - np.log(objective_old)) if np.isfinite(objective_new) else np.nan

    def compute_Jcb(self):
        if self.Kx_family == 'step':
            sys.path.append(os.path.abspath("../utilities"))
            from utilities.J.matrix_calc import Jcb, calc_basis_matrix
            return Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b) for x, b in zip(self.Kx, self.Kb)])

    def get_Jcb(self):
        return self.J_cb
