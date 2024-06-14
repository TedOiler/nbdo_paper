from .base_model import BaseModel
import numpy as np


class FunctionOnFunctionModel(BaseModel):
    def __init__(self, I_theta, I_N, J_CH, Sigma):
        self.I_theta = I_theta  # Identity matrix scaled by the size of Ky
        self.I_N = I_N  # Identity matrix scaled by the size of N
        self.J_CH = J_CH  # Integral of basis matrix from calc_J_CH
        self.Sigma = Sigma  # Error structure matrix from calc_Sigma

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
