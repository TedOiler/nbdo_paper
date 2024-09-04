from .base_model import BaseModel
import numpy as np
import os
import sys
from scipy.linalg import block_diag
import tensorflow as tf


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

        def compute_objective_tf(self, Gamma_, N, Kx):
            Gamma = tf.concat([tf.ones((N, 1)), Gamma_], axis=1)
        Z = tf.matmul(Gamma, self.J_CH)
        ZtZ = tf.matmul(Z, Z, transpose_a=True)

        try:
            ZtZ_inv = tf.linalg.inv(ZtZ)
            value = tf.linalg.trace(ZtZ_inv) * tf.linalg.trace(self.Sigma)
        except tf.errors.InvalidArgumentError:
            return tf.constant(np.nan)

        return tf.where(tf.math.is_finite(value), value, tf.constant(np.nan))

    def compute_objective_tf(self, Gamma_, N, Kx):
        Gamma = tf.concat([tf.ones((N, 1)), Gamma_], axis=1)
        Z = tf.matmul(Gamma, self.J_CH)
        ZtZ = tf.matmul(Z, Z, transpose_a=True)

        try:
            ZtZ_inv = tf.linalg.inv(ZtZ)
            value = tf.linalg.trace(ZtZ_inv) * tf.linalg.trace(self.Sigma)
        except tf.errors.InvalidArgumentError:
            return tf.constant(np.nan)

        return tf.where(tf.math.is_finite(value), value, tf.constant(np.nan))

    def compute_objective_bo(self, Gamma_, N, Kx):
        Gamma = np.hstack((np.ones((N, 1)), Gamma_))
        Z = Gamma @ self.J_CH

        try:
            ZtZ_inv = np.linalg.inv(Z.T @ Z)
            value = np.trace(ZtZ_inv) * np.trace(self.Sigma)
        except np.linalg.LinAlgError:
            return 1e10

        return 1e10 if value < 0 else value

    def compute_Jcb(self):
        if self.Kx_family == 'step':
            sys.path.append(os.path.abspath("../utilities"))
            from utilities.J.matrix_calc import Jcb, calc_basis_matrix
            Jcb = Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b) for x, b in zip(self.Kx, self.Kb)])
            return block_diag(1, Jcb)
        if self.Kx_family == 'b-spline':
            sys.path.append(os.path.abspath("../utilities"))
            from utilities.J.J_bsplines_poly import Jcb, calc_basis_matrix
            Jcb = Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b, k=self.k_degree, knots_num=self.knots_num)
                        for x, b in zip(self.Kx, self.Kb)])
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
