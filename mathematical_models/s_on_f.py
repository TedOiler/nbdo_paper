from .base_model import BaseModel
from J.jmatrix import JMatrix
from basis.bspline import BSplineBasis
from basis.polynomial import PolynomialBasis
from basis.fourier import FourierBasis
import numpy as np
import tensorflow as tf


class ScalarOnFunctionModel(BaseModel):
    def __init__(self, bases_pairs=None):

        self.basis_pairs = bases_pairs
        self.J = self.compute_J()
        self.Kx = self.J.shape[0]
        self.Kb = self.J.shape[1]

    def __str__(self):
        return (f"ScalarOnFunctionModel(Kx={self.Kx}, Kb={self.Kb}, "
                f"Kx_family='{self.Kx_family}', Kb_family='{self.Kb_family}', "
                f"k_degree={self.k_degree}, knots_num={self.knots_num})")

    def __repr__(self):
        return (f"ScalarOnFunctionModel(Kx={self.Kx}, Kb={self.Kb}, "
                f"Kx_family='{self.Kx_family}', Kb_family='{self.Kb_family}', "
                f"k_degree={self.k_degree}, knots_num={self.knots_num})")

    def Covar(self, Gamma, library):
        if library == 'numpy':
            ones = np.ones((Gamma.shape[0], 1))
            Zetta = np.concatenate((ones, Gamma @ self.J), axis=1)
            Covar = Zetta.T @ Zetta
            return Covar
        elif library == 'tensorflow':
            batch_size = tf.shape(Model_mat)[0]
            ones = tf.ones((batch_size, f_coeffs, 1))
            X = tf.reshape(Model_mat, (-1, f_coeffs, n))
            Z = tf.concat([ones, tf.matmul(X, self.J)], axis=2)
            Covar = tf.matmul(Z, Z, transpose_a=True)
            return Covar

    def compute_objective(self, Gamma):
        Covar = self.Covar(Gamma, library='numpy')
        try:
            P_inv = np.linalg.inv(Covar)
        except np.linalg.LinAlgError:
            return np.nan

        value = np.trace(P_inv)

        return value

    def compute_objective_input(self, x, i, j, Gamma):
        Gamma[i, j] = x
        return self.compute_objective(Gamma)

    def compute_objective_tf(self, X, m, n):
        Covar = self.Covar(X, m, library='tensorflow')
        batch_size = tf.shape(X)[0]
        ones = tf.ones((batch_size, m, 1))
        X = tf.reshape(X, (-1, m, n))
        Z = tf.concat([ones, tf.matmul(X, self.J)], axis=2)
        Z_t_Z = tf.matmul(Z, Z, transpose_a=True)

        det = tf.linalg.det(Z_t_Z)
        epsilon = 1e-6  # TODO: affects results significantly!! need to find out how to set it dynamically.
        condition = tf.abs(det)[:, None, None] < epsilon

        diagonal = tf.linalg.diag_part(Z_t_Z) + epsilon
        Z_t_Z_epsilon = Z_t_Z + tf.linalg.diag(diagonal - tf.linalg.diag_part(Z_t_Z))
        Z_t_Z_regularized = tf.where(condition, Z_t_Z_epsilon, Z_t_Z)

        M = tf.linalg.inv(Z_t_Z_regularized)
        value = tf.linalg.trace(M)
        return tf.where(value < 0, tf.constant(1e10), value)

    def compute_objective_bo(self, X, m, n):
        ones = np.ones((m, 1)).reshape(-1, 1)
        X = np.array(X).reshape(m, n)
        Z = np.hstack((ones, X @ self.J))
        try:
            M = np.linalg.inv(Z.T @ Z)
        except np.linalg.LinAlgError:
            return 1e10
        result = np.trace(M)
        return 1e10 if result < 0 else result

    def compute_J(self):
        return JMatrix(self.basis_pairs).compute()

    def get_Jcb(self):
        return self.J
