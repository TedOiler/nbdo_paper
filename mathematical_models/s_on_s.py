from .base_model import BaseModel
import numpy as np
import tensorflow as tf
from itertools import combinations_with_replacement


class ScalarOnScalarModel(BaseModel):
    def __init__(self, Kx, order=1):
        self.Kx = Kx
        self.J_cb = self.compute_Jcb()
        self.order = order

    def __str__(self):
        return f"ScalarOnScalarModel(Kx={self.Kx}, J_cb=Identity matrix of size {self.J_cb.shape})"

    def __repr__(self):
        return f"ScalarOnScalarModel(Kx={self.Kx}, J_cb=Identity matrix of size {self.J_cb.shape})"

    def calc_model_matrix(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        model_mat = np.ones((n_samples, 1))

        for o in range(1, self.order + 1):
            for combo in combinations_with_replacement(range(n_features), o):
                term = np.prod([X[:, i] for i in combo], axis=0)
                model_mat = np.hstack((model_mat, term[:, np.newaxis]))
        return model_mat

    def calc_covar_matrix(self, Model_mat):
        model_mat = self.calc_model_matrix(Model_mat)
        Covar = model_mat.T @ model_mat
        return Covar

    def compute_objective(self, Model_mat):
        Covar = self.calc_covar_matrix(Model_mat)

        try:
            P_inv = np.linalg.inv(Covar)
        except np.linalg.LinAlgError:
            return np.nan

        value = np.trace(P_inv)
        return value

    def compute_objective_input(self, x, i, j, Model_mat):
        Model_mat[i, j] = x
        return self.compute_objective(Model_mat)

# ---------------------------------------- Tensorflow ----------------------------------------

    def compute_objective_tf(self, X, m, n):
        batch_size = tf.shape(X)[0]
        ones = tf.ones((batch_size, m, 1))
        X = tf.reshape(X, (-1, m, n))
        print(X)
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
