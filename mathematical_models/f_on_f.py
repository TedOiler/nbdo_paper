from .base_model import BaseModel
from J.jmatrix import JMatrix
import numpy as np
from scipy.linalg import block_diag
import tensorflow as tf
from scipy.integrate import quad
from scipy.misc import derivative


class FunctionOnFunctionModel(BaseModel):
    def __init__(self, bases_pairs=None, Sigma_decay=None, Kb_t=2, const=True, lambda_s=None, lambda_t=None, R=None, S=None, J_HH=None, Sigma=None):
        self.basis_pairs = bases_pairs
        self.Sigma_decay = Sigma_decay
        self.const = const

        offset = 1 if self.const else 0
        self.J = self.compute_J()
        self.J_HH = J_HH
        self.Kx = self.J.shape[0] - offset
        self.Kb_s = self.J.shape[1] - offset
        self.Kb = self.Kb_s  # For consistency reasons need to make better (this is used in the check in cordex to check if the model is estimable)
        self.Kb_t = Kb_t  # L in mathematics

        self.Sigma = self.compute_Sigma()

        self.lambda_s = lambda_s  # penalty for s part of beta
        self.R = R
        self.lambda_t = lambda_t  # penalty for t part of beta
        self.S = S
        self.I_Kb_t = np.eye(self.Kb_t)

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

        if self.lambda_s and self.lambda_t is not None:

            f_lambda_x = np.kron(np.eye(self.Kb_t), Z.T @ Z)
            f_lambda_s = self.lambda_s * np.kron(np.eye(self.Kb_t), self.R)
            f_lambda_t = self.lambda_t * np.kron(self.S, self.J_HH)
            F_lambda = f_lambda_x + f_lambda_s + f_lambda_t
            F_lambda_inv = np.linalg.inv(F_lambda)

            # A-opt ---
            # value_1 = np.kron(self.Sigma, Z.T @ Z)
            # value_2 = F_lambda_inv @ value_1
            # value_3 = value_2 @ F_lambda_inv
            # value = np.trace(value_3)
            # ---------

            # D-opt ---
            # value = np.linalg.det(F_lambda_inv @ f_lambda_x @ F_lambda_inv)
            # ---------
            #
            nominator = np.linalg.det(Z.T @ Z)**((self.Kb_t + 1)/2)
            denom1 = np.kron(self.I_Kb_t, Z.T @ Z)
            denom2 = self.lambda_s * np.kron(self.I_Kb_t, self.R)
            denom3 = self.lambda_t * np.kron(self.S, self.J_HH)
            denominator = np.linalg.det(denom1 + denom2 + denom3)
            # value = np.exp(np.log(nominator) - np.log(denominator))
            value = np.exp(np.log(1) - np.log(denominator)) #  Point after meeting 17-2 where we said we can only try to maximize the denominator of proposition 2.

        else:
            ZtZ_inv = np.linalg.inv(Z.T @ Z)
            # M = np.kron(ZtZ_inv, self.Sigma)
            value = np.linalg.det(ZtZ_inv)
            # value = np.trace(ZtZ_inv) * np.trace(self.Sigma)

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
            # elements = np.zeros((self.Kx + 1) * self.Kb_t)
            elements = np.zeros(self.Kb_t)
            Sigma = np.diag(elements)
            Sigma[0, 0] = 1
            return Sigma
        else:
            decay = np.exp(-self.Sigma_decay * inputs)
            return np.diag(decay)

    def compute_R(self):
        R_blocks = []

        for _, b_base in self.basis_pairs:
            num_basis = b_base.num_basis()

            def second_derivative(i, t):
                return derivative(lambda x: b_base.evaluate_basis_function(i, x), t, dx=1e-6, n=2)

            R = np.zeros((num_basis, num_basis))
            for i in range(num_basis):
                for j in range(num_basis):
                    integrand = lambda t: second_derivative(i, t) * second_derivative(j, t)
                    R[i, j], _ = quad(integrand, 0, 1)

            R_blocks.append(R)

        # Create block diagonal matrix from all R_blocks
        R_combined = block_diag(*R_blocks)

        # Expand the final matrix to include the leading 1
        final_size = R_combined.shape[0] + 1
        R_with_one = np.zeros((final_size, final_size))
        R_with_one[0, 0] = 1
        R_with_one[1:, 1:] = R_combined

        return R_with_one

    def compute_S(self):
        return np.eye(self.Kb_t)

    def set_Sigma(self, Sigma):
        self.Sigma = Sigma

