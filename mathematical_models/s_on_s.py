from .base_model import BaseModel
import numpy as np
import tensorflow as tf


class ScalarOnScalarModel(BaseModel):
    def __init__(self, Kx):
        self.Kx = Kx
        self.J_cb = self.compute_Jcb()

    def __str__(self):
        return f"ScalarOnScalarModel(Kx={self.Kx}, J_cb=Identity matrix of size {self.J_cb.shape})"

    def __repr__(self):
        return f"ScalarOnScalarModel(Kx={self.Kx}, J_cb=Identity matrix of size {self.J_cb.shape})"

    def Covar(self, Model_mat):
        ones = np.ones((Model_mat.shape[0], 1))
        Zetta = np.concatenate((ones, Model_mat), axis=1)
        Covar = Zetta.T @ Zetta
        return Covar

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
        return tf.where(value < 0, tf.constant(1e10), value)

    def compute_objective_bo(self, X, m, n):
        ones = np.ones((m, 1)).reshape(-1, 1)
        X = np.array(X).reshape(m, n)
        Z = np.hstack((ones, X @ self.J_cb))
        try:
            M = np.linalg.inv(Z.T @ Z)
        except np.linalg.LinAlgError:
            return 1e10
        result = np.trace(M)
        return 1e10 if result < 0 else result

    def compute_Jcb(self):
        return np.eye(self.Kx[0])

    def get_Jcb(self):
        return self.J_cb
