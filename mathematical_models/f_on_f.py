from .base_model import BaseModel
from J.jmatrix import JMatrix
import numpy as np
from scipy.linalg import block_diag
import tensorflow as tf


class FunctionOnFunctionModel(BaseModel):
    def __init__(self, bases_pairs=None, Sigma_decay=None, Kb_t=2, const=True):
        self.basis_pairs = bases_pairs
        self.Sigma_decay = Sigma_decay
        self.const = const
        offset = 1 if self.const else 0
        self.J = self.compute_J()
        self.Kx = self.J.shape[0] - offset
        self.Kb_s = self.J.shape[1] - offset
        self.Kb = self.Kb_s  #  For consistency reasons need to make better (this is used in the check in cordex to check if the model is estimable)
        self.Kb_t = Kb_t  # L in mathematics

        self.Sigma = self.compute_Sigma()

    def _prepare_Gamma(self, Gamma_, N):
        if self.const:
            ones = np.ones((N, 1))
            Gamma = np.hstack((ones, Gamma_))
        else:
            Gamma = Gamma_
        return Gamma

    def compute_objective(self, Gamma_, N, Kx):
        Gamma = self._prepare_Gamma(Gamma_, N)
        Z = Gamma @ self.J
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        M = np.kron(ZtZ_inv, self.Sigma)

        # A-optimality: Trace of the covariance matrix
        value = np.trace(ZtZ_inv) * np.trace(self.Sigma)

        return value if np.isfinite(value) else np.nan

    def compute_objective_input(self, x, i, j, Gamma_, N, Kx):
        Gamma_[i, j] = x
        return self.compute_objective(Gamma_, N, Kx)

    def compute_objective_tf(self, Gamma_, N, Kx):
        # Convert self.J and self.Sigma to TensorFlow tensors
        J_tf = tf.convert_to_tensor(self.J, dtype=tf.float32)
        Sigma_tf = tf.convert_to_tensor(self.Sigma, dtype=tf.float32)

        # Reshape Gamma_ from (batch_size, N * Kx) to (batch_size, N, Kx)
        batch_size = tf.shape(Gamma_)[0]
        Gamma_reshaped = tf.reshape(Gamma_, [batch_size, N, Kx])

        # Prepare Gamma
        if self.const:
            ones = tf.ones([batch_size, N, 1], dtype=tf.float32)
            Gamma = tf.concat([ones, Gamma_reshaped], axis=2)  # Shape: (batch_size, N, Kx + 1)
        else:
            Gamma = Gamma_reshaped  # Shape: (batch_size, N, Kx)

        # Expand J_tf to match batch dimensions
        J_tf_expanded = tf.expand_dims(J_tf, axis=0)  # Shape: (1, Kx + offset, Kb_s)
        J_tf_expanded = tf.tile(J_tf_expanded, [batch_size, 1, 1])  # Shape: (batch_size, Kx + offset, Kb_s)

        # Compute Z for each design in the batch
        Z = tf.matmul(Gamma, J_tf_expanded)  # Shape: (batch_size, N, Kb_s)

        # Compute ZtZ for each design in the batch
        ZtZ = tf.matmul(Z, Z, transpose_a=True)  # Shape: (batch_size, Kb_s, Kb_s)

        # Compute the inverse and trace for each design
        def compute_design_objective(ZtZ_single):
            try:
                ZtZ_inv = tf.linalg.inv(ZtZ_single)
                value = tf.linalg.trace(ZtZ_inv) * tf.linalg.trace(Sigma_tf)
            except tf.errors.InvalidArgumentError:
                value = tf.constant(np.nan, dtype=tf.float32)
            return value

        # Map over the batch
        values = tf.map_fn(compute_design_objective, ZtZ, dtype=tf.float32)

        # Return the mean value over the batch
        return tf.reduce_mean(values)

    def compute_objective_bo(self, Gamma_, N, Kx):
        Gamma_ = Gamma_.reshape((N, Kx))
        value = self.compute_objective(Gamma_, N, Kx)
        return 1e10 if np.isnan(value) or value < 0 else value

    def compute_J(self):
        if self.const:
            return block_diag(np.array([[1]]), JMatrix(self.basis_pairs).compute())
        else:
            return JMatrix(self.basis_pairs).compute()

    def compute_Sigma(self):
        inputs = np.linspace(0, 1, self.Kb_t)
        if self.Sigma_decay == np.inf:
            elements = np.zeros((self.Kx + 1) * self.Kb_t)
            Sigma = np.diag(elements)
            Sigma[0, 0] = 1
            return Sigma
        else:
            decay = np.exp(-self.Sigma_decay * inputs)
            return np.diag(decay)
