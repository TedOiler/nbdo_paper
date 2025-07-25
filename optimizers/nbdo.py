import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.backend import clear_session
import gc
from scipy.stats import qmc

from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from matplotlib import pyplot as plt

from pathlib import Path
import sys

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from base_optimizer import BaseOptimizer
from mathematical_models.f_on_f import FunctionOnFunctionModel
from mathematical_models.s_on_f import ScalarOnFunctionModel
from mathematical_models.s_on_s import ScalarOnScalarModel


class NBDO:
    def __init__(self, model, latent_dim,
                 base=2, max_layers=None, alpha=0.0,
                 latent_space_activation='tanh', output_layer_activation='tanh'):
        self.model = model
        self.runs = None
        # self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.base = base
        self.max_layers = max_layers
        self.alpha = alpha
        self.latent_space_activation = latent_space_activation
        self.output_space_activation = output_layer_activation

        self.input_dim = None
        self.encoder = None
        self.latent = None
        self.decoder = None
        self.autoencoder = None
        self.num_layers = None

        self.input_layer = None
        self.output_layer = None

        self.train_set = None
        self.val_set = None
        self.history = None

        self.optimal_latent_var = None
        self.optimal_cr = None
        self.optimal_des = None
        self.search_history = None
        self.eval_history = None

    def __repr__(self):
        return f"NBDO(\n" \
               f"  model: {self.model.__class__.__name__},\n" \
               f"  max_layers: {self.max_layers},\n" \
               f"  latent_dim: {self.latent_dim},\n" \
               f"  input_dim: {self.input_dim},\n" \
               f"  num_layers: {self.num_layers},\n" \
               f"  train_set: {self.train_set.shape if self.train_set is not None else None},\n" \
               f"  val_set: {self.val_set.shape if self.val_set is not None else None}\n" \
               f"  base: {self.base},\n" \
               f"  alpha: {self.alpha},\n" \
               f"  latent_space_activation: {self.latent_space_activation},\n" \
               f"  output_space_activation: {self.output_space_activation},\n" \
               f")"

    def __str__(self):
        return f"NBDO Model Summary:\n" \
               f"  Model Type: {self.model.__class__.__name__}\n" \
               f"  --------------------------------------------\n" \
               f"  Max Dimension: {self.max_layers}\n" \
               f"  Input Dimension: {self.input_dim}\n" \
               f"  Latent Dimension: {self.latent_dim}\n" \
               f"  Number of Layers: {self.num_layers}\n" \
               f"  --------------------------------------------\n" \
               f"  Training Set Size: {self.train_set.shape[0] if self.train_set is not None else None}\n" \
               f"  Validation Set Size: {self.val_set.shape[0] if self.val_set is not None else None}\n" \
               f"  --------------------------------------------\n" \
               f"  Base: {self.base}\n" \
               f"  Alpha: {self.alpha}\n" \
               f"  Latent Space Activation: {self.latent_space_activation}\n" \
               f"  Output Space Activation: {self.output_space_activation}"

    def _build_encoder(self):

        self.num_layers = int(np.log(self.input_dim / self.latent_dim) / np.log(self.base))
        self.num_layers = min(self.num_layers, self.max_layers) if self.max_layers is not None else self.num_layers

        self.input_layer = Input(shape=(self.input_dim,))
        encoder = self.input_layer
        for layer in range(self.num_layers):
            n_neurons = int(self.input_dim / (self.base ** (layer + 1)))
            encoder = Dense(n_neurons, activation=LeakyReLU(alpha=self.alpha))(encoder)

        latent = Dense(self.latent_dim, activation=self.latent_space_activation, name='latent')(encoder)
        self.encoder = Model(self.input_layer, latent, name='encoder')

    def _build_decoder(self):

        latent_inputs = Input(shape=(self.latent_dim,))
        decoder = latent_inputs
        for layer in range(self.num_layers, 0, -1):
            n_neurons = int(self.input_dim / self.base ** layer)
            decoder = Dense(n_neurons, activation=LeakyReLU(alpha=self.alpha))(decoder)
        self.output_layer = Dense(self.input_dim, activation=self.output_space_activation)(decoder)
        self.decoder = Model(latent_inputs, self.output_layer, name='decoder')

    def _build_autoencoder(self):
        self._build_encoder()
        self._build_decoder()

        autoencoder_input = self.input_layer
        latent_representation = self.encoder(autoencoder_input)
        autoencoder_output = self.decoder(latent_representation)

        self.autoencoder = Model(autoencoder_input, autoencoder_output, name='autoencoder')

    def _get_custom_loss(self):
        if isinstance(self.model, ScalarOnFunctionModel):
            def custom_loss(y_true, y_pred):
                reconstruction_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
                m = self.runs
                n = self.model.Kx[0]
                objective_value = self.model.compute_objective_tf(y_pred, m, n)
                return objective_value

            return custom_loss
        elif isinstance(self.model, FunctionOnFunctionModel):
            def custom_loss(y_true, y_pred):
                m = self.runs
                n = self.model.Kx
                objective_value = self.model.compute_objective_tf(y_pred, m, n)
                return objective_value
            return custom_loss
        elif isinstance(self.model, ScalarOnScalarModel):
            def custom_loss(y_true, y_pred):
                reconstruction_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
                m = self.runs
                n = self.model.Kx[0]
                objective_value = self.model.compute_objective_tf(y_pred, m, n)
                return objective_value
            return custom_loss

    def compute_train_set(self, num_designs, runs, epsilon=1e-10, type='random'):
        def model_matrix_columns(n, p):
            from math import comb
            return comb(n + p, p)

        self.runs = runs
        design_matrix = []
        valid_count = 0

        # Determine number of raw input features (NOT the number of model matrix terms)
        if isinstance(self.model, ScalarOnScalarModel):
            n_features = self.model.Kx[0]
        elif isinstance(self.model, ScalarOnFunctionModel):
            n_features = self.model.Kx[0]
        elif isinstance(self.model, FunctionOnFunctionModel):
            n_features = self.model.Kx
        else:
            raise ValueError("Unsupported model type.")

        max_attempts = int(1e6)

        for attempt in range(max_attempts):
            if valid_count == num_designs:
                break

            # Generate candidate design of shape (runs, n_features)
            if type == 'random':
                candidate_matrix = np.random.uniform(-1, 1, size=(runs, n_features))
            elif type == 'LHC':
                sampler = qmc.LatinHypercube(d=n_features)
                lhs_sample = sampler.random(n=runs)
                candidate_matrix = 2 * lhs_sample - 1  # scale to [-1, 1]
            else:
                raise ValueError("type must be 'random' or 'LHC'")

            # Compute information matrix for validation
            try:
                if isinstance(self.model, ScalarOnScalarModel):
                    ZtZ = self.model.calc_covar_matrix(candidate_matrix)
                elif isinstance(self.model, ScalarOnFunctionModel):
                    Z = np.hstack((np.ones((runs, 1)), candidate_matrix @ self.model.J))
                    ZtZ = Z.T @ Z
                elif isinstance(self.model, FunctionOnFunctionModel):
                    Gamma = np.hstack((np.ones((runs, 1)), candidate_matrix))
                    Z = np.matmul(Gamma, self.model.J)
                    ZtZ = Z.T @ Z
                else:
                    continue

                # Accept design if well-conditioned or if explicitly ScalarOnScalar (always accepted)
                if np.linalg.det(ZtZ) > epsilon or isinstance(self.model, ScalarOnScalarModel):
                    design_matrix.append(candidate_matrix)
                    valid_count += 1

            except np.linalg.LinAlgError:
                continue

        if valid_count < num_designs:
            raise RuntimeError(f"Only found {valid_count} valid designs after {max_attempts} attempts.")

        # Stack and flatten: (num_designs, runs * n_features)
        reshaped_design_matrix = np.stack(design_matrix).reshape(num_designs, -1)
        self.train_set, self.val_set = train_test_split(reshaped_design_matrix,
                                                        test_size=0.2,
                                                        random_state=42)
        self.input_dim = self.train_set.shape[1]


    def fit(self, epochs, batch_size=32,
            patience=50, optimizer=tf.keras.optimizers.legacy.RMSprop()):
        self._build_autoencoder()
        custom_loss = self._get_custom_loss()
        self.autoencoder.compile(optimizer=optimizer, loss=custom_loss)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.autoencoder.build(input_shape=(None, self.input_dim))
        self.history = self.autoencoder.fit(self.train_set, self.train_set,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            validation_data=(self.val_set, self.val_set),
                                            callbacks=[early_stopping])

        return self.history

    def optimize(self, n_calls=10, acq_func='EI', acq_optimizer='sampling',
                 n_random_starts=5, verbose=True):

        def objective(latent_var):
            latent_var = np.array(latent_var).reshape(1, -1)
            decoded = self.decoder.predict(latent_var)
            if self.model.__class__.__name__ == 'ScalarOnFunctionModel':
                optimality = self.model.compute_objective_bo(X=decoded, m=self.runs, n=self.model.Kx[0])
                return optimality
            elif self.model.__class__.__name__ == 'FunctionOnFunctionModel':
                optimality = self.model.compute_objective_bo(decoded, self.runs, self.model.Kx)
                return optimality
            elif self.model.__class__.__name__ == 'ScalarOnScalarModel':
                optimality = self.model.compute_objective_bo(X=decoded, m=self.runs, n=self.model.Kx[0])
                return optimality

        dimensions = [(-1., 1.) for _ in range(self.latent_dim)]
        res = gp_minimize(objective, dimensions, n_calls=n_calls,
                          random_state=42, verbose=verbose, n_jobs=-1,
                          n_random_starts=n_random_starts, acq_func=acq_func, acq_optimizer=acq_optimizer)
        self.optimal_latent_var = res.x
        self.optimal_cr = res.fun
        self.optimal_des = self.decode(np.array(self.optimal_latent_var).reshape(1, -1))
        self.search_history = res.x_iters
        self.eval_history = res.func_vals
        clear_session()
        return self.optimal_cr, self.optimal_des

    def clear_memory(self):
        del self.autoencoder
        del self.encoder
        del self.decoder
        gc.collect()

    def encode(self, design):
        return self.encoder.predict(design.reshape(1, -1))

    def decode(self, latent):
        return self.decoder.predict(latent).reshape(self.runs, -1)
