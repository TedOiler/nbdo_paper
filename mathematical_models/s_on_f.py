from .base_model import BaseModel
import numpy as np
import os
import sys


class ScalarOnFunctionModel(BaseModel):
    def __init__(self, Kx, Kb, Kx_family, Kb_family='polynomial', k_degree=None, knots_num=None):
        self.Kx_family = Kx_family
        self.Kx = Kx
        self.Kb_family = Kb_family
        self.Kb = Kb
        self.k_degree = k_degree
        self.knots_num = self.Kx[0] + 1

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
        if self.Kx_family == 'step':
            sys.path.append(os.path.abspath("../utilities"))
            from utilities.matrix_calc import Jcb, calc_basis_matrix
            return Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b) for x, b in zip(self.Kx, self.Kb)])
        if self.Kx_family == 'b-spline':
            sys.path.append(os.path.abspath("../../utilities"))
            from utilities.J_bsplines_poly import Jcb, calc_basis_matrix
            return Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b, k=self.k_degree, knots_num=self.knots_num)
                         for x, b in zip(self.Kx, self.Kb)])

    def get_Jcb(self):
        return self.J_cb
