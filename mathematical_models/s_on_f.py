from .base_model import BaseModel
import numpy as np
import os
import sys
import tensorflow as tf


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

    def compute_objective_tf(self, X, m, n):
        batch_size = tf.shape(X)[0]
        ones = tf.ones((batch_size, m, 1))
        X = tf.reshape(X, (-1, m, n))
        Z = tf.concat([ones, tf.matmul(X, self.J_cb)], axis=2)

        Z_t_Z = tf.matmul(Z, Z, transpose_a=True)
        det = tf.linalg.det(Z_t_Z)
        epsilon = 1e-6
        condition = tf.abs(det)[:, None, None] < epsilon

        diagonal = tf.linalg.diag_part(Z_t_Z) + epsilon
        Z_t_Z_epsilon = Z_t_Z + tf.linalg.diag(diagonal - tf.linalg.diag_part(Z_t_Z))
        Z_t_Z_regularized = tf.where(condition, Z_t_Z_epsilon, Z_t_Z)

        M = tf.linalg.inv(Z_t_Z_regularized)
        value = tf.linalg.trace(M)
        value = tf.where(value < 0, tf.constant(1e10), value)
        return value

    def compute_objective_tf_2(self, X, m, n):
        ones = tf.ones((m, 1))
        Gamma = X[:, :self.Kx[0]]
        J_cb = tf.constant(self.J_cb, dtype=tf.float32)
        Zetta = tf.concat([ones, tf.matmul(Gamma, J_cb)], axis=1)
        Covar = tf.matmul(tf.transpose(Zetta), Zetta)
        try:
            P_inv = tf.linalg.inv(Covar)
        except tf.errors.InvalidArgumentError:
            return tf.constant(1e10, dtype=tf.float32)

        value = tf.linalg.trace(P_inv)
        return value

    def compute_Jcb(self):
        if self.Kx_family == 'step':
            sys.path.append(os.path.abspath("../utilities"))
            from utilities.J.matrix_calc import Jcb, calc_basis_matrix
            return Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b) for x, b in zip(self.Kx, self.Kb)])
        if self.Kx_family == 'b-spline':
            sys.path.append(os.path.abspath("../../utilities"))
            from utilities.J.J_bsplines_poly import Jcb, calc_basis_matrix
            return Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b, k=self.k_degree, knots_num=self.knots_num)
                         for x, b in zip(self.Kx, self.Kb)])

    def get_Jcb(self):
        return self.J_cb
