from .base_model import BaseModel
import numpy as np
import os
import sys


class ScalarOnScalar(BaseModel):
    def __init__(self):
        pass

    def compute_objective(self, Model_mat):
        ones = np.ones((Model_mat.shape[0], 1))
        Zetta = np.concatenate((ones, Model_mat), axis=1)
        Covar = Zetta.T @ Zetta

        try:
            P_inv = np.linalg.inv(Covar)
        except np.linalg.LinAlgError:
            return np.nan

        value = np.trace(P_inv)
        return value

    def compute_objective_input(self, x, i, j, Model_mat):
        Model_mat[i, j] = x
        return self.compute_objective(Model_mat)

