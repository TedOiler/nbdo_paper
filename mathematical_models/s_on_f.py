from .base_model import BaseModel
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("../utilities"))
from utilities.matrix_calc import Jcb, calc_basis_matrix


class ScalarOnFunctionModel(BaseModel):
    def __init__(self, Kx, Kb):
        self.Kx = Kx
        self.Kb = Kb
        self.J_cb = self.compute_Jcb()

    def compute_objective(self, Model_mat, f_coeffs):
        ones = np.ones((Model_mat.shape[0], 1))
        Gamma = Model_mat[:, :f_coeffs]
        Zetta = np.concatenate((ones, Gamma @ self.J_cb), axis=1)
        Covar = Zetta.T @ Zetta

        try:
            P_inv = np.linalg.inv(Covar)
        except np.linalg.LinAlgError:
            return np.nan

        value = np.trace(P_inv)

        return value

    def compute_objective_input(self, x, i, j, Model_mat, f_coeffs):
        Model_mat[i, j] = x
        return self.compute_objective(Model_mat, f_coeffs)

    def compute_Jcb(self):
        return Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b) for x, b in zip(self.Kx, self.Kb)])

    def get_Jcb(self):
        return self.J_cb
